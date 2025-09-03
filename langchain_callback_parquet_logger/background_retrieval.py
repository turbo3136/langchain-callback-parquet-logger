"""
Background response retrieval for OpenAI responses.

Retrieves completed responses from OpenAI's API for background/async requests,
with automatic logging to Parquet files, rate limiting, and checkpoint support.
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List
import warnings

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import openai
except ImportError:
    openai = None

from .logger import ParquetLogger


async def retrieve_background_responses(
    df: "pd.DataFrame",
    openai_client,
    logger: Optional[ParquetLogger] = None,
    response_id_col: str = "response_id",
    logger_custom_id_col: str = "logger_custom_id",
    batch_size: int = 50,
    max_retries: int = 3,
    timeout: float = 30.0,
    show_progress: bool = True,
    checkpoint_file: Optional[str] = None,
    return_results: bool = True,
) -> Optional["pd.DataFrame"]:
    """
    Retrieve background responses from OpenAI and log them to Parquet.
    
    Args:
        df: DataFrame containing response IDs to retrieve
        openai_client: Initialized OpenAI async client
        logger: Optional ParquetLogger instance for logging responses
        response_id_col: Column name containing response IDs (default: "response_id")
        logger_custom_id_col: Column name containing custom IDs (default: "logger_custom_id")
        batch_size: Number of concurrent requests (default: 50)
        max_retries: Maximum retries per request (default: 3)
        timeout: Timeout per request in seconds (default: 30.0)
        show_progress: Show progress bar (default: True)
        checkpoint_file: Optional path to checkpoint file for resume capability
        return_results: If False, don't keep results in memory (default: True)
        
    Returns:
        DataFrame with retrieval results if return_results=True, else None
        
    Example:
        >>> import openai
        >>> from langchain_callback_parquet_logger import ParquetLogger, retrieve_background_responses
        >>> 
        >>> client = openai.AsyncClient()
        >>> df = pd.DataFrame({
        ...     'response_id': ['resp_123', 'resp_456'],
        ...     'logger_custom_id': ['user-001', 'user-002']
        ... })
        >>> 
        >>> with ParquetLogger('./logs') as logger:
        ...     results = await retrieve_background_responses(df, client, logger=logger)
    """
    if pd is None:
        raise ImportError("pandas is required for background retrieval. Install with: pip install pandas")
    
    # Validate required columns
    if response_id_col not in df.columns:
        raise ValueError(f"Column '{response_id_col}' not found in DataFrame")
    if logger_custom_id_col not in df.columns:
        warnings.warn(f"Column '{logger_custom_id_col}' not found. Using empty string for custom IDs.")
        df = df.copy()
        df[logger_custom_id_col] = ""
    
    # Initialize progress tracking
    progress_bar = None
    if show_progress:
        try:
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    from tqdm.notebook import tqdm
                else:
                    from tqdm import tqdm
            except ImportError:
                from tqdm import tqdm
            progress_bar = tqdm(total=len(df), desc="Retrieving responses", position=0, leave=True)
        except ImportError:
            print(f"Retrieving {len(df)} responses...")
    
    # Load checkpoint if exists
    processed_ids = set()
    failed_ids = {}
    if checkpoint_file and Path(checkpoint_file).exists():
        try:
            checkpoint_df = pd.read_parquet(checkpoint_file)
            processed_ids = set(checkpoint_df['response_id'].values)
            if 'error' in checkpoint_df.columns:
                failed_ids = dict(zip(
                    checkpoint_df[checkpoint_df['error'].notna()]['response_id'],
                    checkpoint_df[checkpoint_df['error'].notna()]['error']
                ))
            if progress_bar:
                progress_bar.update(len(processed_ids))
            print(f"Resumed from checkpoint: {len(processed_ids)} already processed")
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint: {e}")
    
    # Prepare results storage
    results = [] if return_results else None
    checkpoint_batch = []
    
    # Rate limiting state
    rate_limit_reset = 0
    rate_limit_remaining = batch_size
    
    async def retrieve_single(row: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve a single response with retries and logging."""
        response_id = row[response_id_col]
        custom_id = row.get(logger_custom_id_col, "")
        
        # Skip if already processed
        if response_id in processed_ids:
            if progress_bar:
                progress_bar.update(1)
            if return_results:
                return {
                    'response_id': response_id,
                    'status': 'already_processed',
                    'openai_response': None,
                    'error': None
                }
            return None
        
        # Log attempt
        if logger:
            logger._add_entry({
                'timestamp': datetime.now(timezone.utc),
                'run_id': '',
                'logger_custom_id': custom_id,
                'event_type': 'background_retrieval_attempt',
                'provider': 'openai',
                'logger_metadata': logger.logger_metadata_json,
                'payload': json.dumps({
                    'response_id': response_id,
                    'attempt_time': datetime.now(timezone.utc).isoformat()
                })
            })
        
        # Retry loop
        last_error = None
        for attempt in range(max_retries):
            try:
                # Check rate limits
                nonlocal rate_limit_reset, rate_limit_remaining
                if rate_limit_remaining <= 0 and time.time() < rate_limit_reset:
                    wait_time = rate_limit_reset - time.time()
                    await asyncio.sleep(wait_time)
                
                # Make request with timeout
                response = await asyncio.wait_for(
                    openai_client.responses.retrieve(response_id),
                    timeout=timeout
                )
                
                # Parse rate limit headers if available
                if hasattr(response, 'headers'):
                    headers = response.headers
                    if 'x-ratelimit-remaining-requests' in headers:
                        rate_limit_remaining = int(headers['x-ratelimit-remaining-requests'])
                    if 'x-ratelimit-reset-after' in headers:
                        rate_limit_reset = time.time() + float(headers['x-ratelimit-reset-after'])
                
                # Log success
                response_data = response.model_dump() if hasattr(response, 'model_dump') else dict(response)
                
                if logger:
                    logger._add_entry({
                        'timestamp': datetime.now(timezone.utc),
                        'run_id': '',
                        'logger_custom_id': custom_id,
                        'event_type': 'background_retrieval_complete',
                        'provider': 'openai',
                        'logger_metadata': logger.logger_metadata_json,
                        'payload': json.dumps({
                            'response_id': response_id,
                            'openai_response': response_data,
                            'status': 'completed',
                            'retrieval_time': datetime.now(timezone.utc).isoformat()
                        })
                    })
                
                processed_ids.add(response_id)
                
                if progress_bar:
                    progress_bar.update(1)
                
                if return_results:
                    return {
                        'response_id': response_id,
                        'status': 'completed',
                        'openai_response': response_data,
                        'error': None
                    }
                return None
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                await asyncio.sleep(2 ** attempt)
            
            except Exception as e:
                # Check if it's a RateLimitError from OpenAI
                if openai and isinstance(e, openai.RateLimitError):
                    # Handle rate limit errors with exponential backoff and jitter
                    last_error = str(e)
                    delay = min(60, (2 ** attempt) * (1 + random.random() * 0.1))
                    await asyncio.sleep(delay)
                    continue
                
                # Handle other exceptions
                last_error = str(e)
                error_str = str(e)
                
                # Check for server errors (5xx)
                if any(error_str.startswith(f'5{x}') for x in '0123456789'):
                    # Retry with backoff for server errors
                    await asyncio.sleep(2 ** attempt)
                else:
                    # Don't retry for other client errors (4xx)
                    break
        
        # Log failure after all retries
        if logger:
            logger._add_entry({
                'timestamp': datetime.now(timezone.utc),
                'run_id': '',
                'logger_custom_id': custom_id,
                'event_type': 'background_retrieval_error',
                'provider': 'openai',
                'logger_metadata': logger.logger_metadata_json,
                'payload': json.dumps({
                    'response_id': response_id,
                    'error': last_error,
                    'error_type': 'retrieval_failed',
                    'attempts': max_retries
                })
            })
        
        failed_ids[response_id] = last_error
        processed_ids.add(response_id)
        
        if progress_bar:
            progress_bar.update(1)
        
        if return_results:
            return {
                'response_id': response_id,
                'status': 'failed',
                'openai_response': None,
                'error': last_error
            }
        return None
    
    # Process in batches
    rows = df.to_dict('records')
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        
        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[retrieve_single(row) for row in batch],
            return_exceptions=True
        )
        
        # Handle results
        for row, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                # Handle unexpected exceptions
                response_id = row[response_id_col]
                error_msg = str(result)
                failed_ids[response_id] = error_msg
                processed_ids.add(response_id)
                
                if progress_bar:
                    progress_bar.update(1)
                
                if return_results:
                    results.append({
                        'response_id': response_id,
                        'status': 'error',
                        'openai_response': None,
                        'error': error_msg
                    })
            elif result is not None and return_results:
                results.append(result)
            
            # Add to checkpoint batch
            if checkpoint_file:
                checkpoint_batch.append({
                    'response_id': row[response_id_col],
                    'processed': True,
                    'error': failed_ids.get(row[response_id_col])
                })
        
        # Save checkpoint periodically
        if checkpoint_file and len(checkpoint_batch) >= 100:
            save_checkpoint(checkpoint_file, checkpoint_batch)
            checkpoint_batch = []
    
    # Final checkpoint save
    if checkpoint_file and checkpoint_batch:
        save_checkpoint(checkpoint_file, checkpoint_batch)
    
    # Final flush of logger buffer
    if logger and hasattr(logger, 'flush'):
        logger.flush()
    
    # Clean up
    if progress_bar:
        progress_bar.close()
    
    # Print summary
    print(f"\nRetrieval complete: {len(processed_ids)} processed, {len(failed_ids)} failed")
    
    if return_results:
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        return results_df
    else:
        return None


def save_checkpoint(checkpoint_file: str, batch: List[Dict[str, Any]]):
    """Save checkpoint data to parquet file."""
    try:
        checkpoint_df = pd.DataFrame(batch)
        
        # Load existing checkpoint if exists
        if Path(checkpoint_file).exists():
            existing_df = pd.read_parquet(checkpoint_file)
            checkpoint_df = pd.concat([existing_df, checkpoint_df], ignore_index=True)
            # Remove duplicates, keeping last
            checkpoint_df = checkpoint_df.drop_duplicates(subset=['response_id'], keep='last')
        
        checkpoint_df.to_parquet(checkpoint_file, compression='snappy')
    except Exception as e:
        warnings.warn(f"Failed to save checkpoint: {e}")