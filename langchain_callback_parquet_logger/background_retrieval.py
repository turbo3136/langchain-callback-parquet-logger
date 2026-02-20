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

# OpenAI response statuses that mean "done, stop polling"
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
# Statuses that indicate the response is still being processed
_PENDING_STATUSES = {"in_progress", "queued", "processing"}


async def retrieve_background_responses(
    df: "pd.DataFrame",
    openai_client=None,
    logger: Optional[ParquetLogger] = None,
    response_id_col: str = "response_id",
    custom_id_col: str = "custom_id",
    batch_size: int = 50,
    max_retries: int = 3,
    timeout: float = 30.0,
    poll_interval: float = 30.0,
    max_poll_attempts: int = 40,
    show_progress: bool = True,
    checkpoint_file: Optional[str] = None,
    return_results: bool = True,
) -> Optional["pd.DataFrame"]:
    """
    Retrieve background responses from OpenAI and log them to Parquet.

    Polls each response until its status is "completed" (or a terminal failure state),
    logging each intermediate "in_progress"/"queued" check as a
    ``background_retrieval_pending`` event and the final outcome as either
    ``background_retrieval_complete`` or ``background_retrieval_error``.

    Args:
        df: DataFrame containing response IDs to retrieve
        openai_client: Initialized OpenAI async client
        logger: Optional ParquetLogger instance for logging responses
        response_id_col: Column name containing response IDs (default: "response_id")
        custom_id_col: Column name containing custom IDs (default: "custom_id")
        batch_size: Number of concurrent requests (default: 50)
        max_retries: Maximum retries per API call for transient errors such as
            rate limits, timeouts, and 5xx server errors (default: 3)
        timeout: Timeout per individual API request in seconds (default: 30.0)
        poll_interval: Seconds to wait between status checks when a response is
            still pending (default: 30.0)
        max_poll_attempts: Maximum number of times to poll a single response before
            giving up. Total max wait ≈ poll_interval × max_poll_attempts (default: 40)
        show_progress: Show progress bar (default: True)
        checkpoint_file: Optional path to checkpoint file for resume capability
        return_results: If False, don't keep results in memory (default: True)

    Returns:
        DataFrame with retrieval results if return_results=True, else None.
        Result columns: response_id, status, openai_response, error

    Example:
        >>> import openai
        >>> from langchain_callback_parquet_logger import ParquetLogger, retrieve_background_responses
        >>>
        >>> client = openai.AsyncClient()
        >>> df = pd.DataFrame({
        ...     'response_id': ['resp_123', 'resp_456'],
        ...     'custom_id': ['user-001', 'user-002']
        ... })
        >>>
        >>> with ParquetLogger('./logs') as logger:
        ...     results = await retrieve_background_responses(df, client, logger=logger)
    """
    if pd is None:
        raise ImportError("pandas is required for background retrieval. Install with: pip install pandas")

    if openai_client is None:
        if openai is None:
            raise ImportError("openai is required for background retrieval. Install with: pip install openai")
        openai_client = openai.AsyncOpenAI()

    # Suppress Pydantic serialization warnings globally
    warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic")

    # Validate required columns
    if response_id_col not in df.columns:
        raise ValueError(f"Column '{response_id_col}' not found in DataFrame")
    if custom_id_col not in df.columns:
        warnings.warn(f"Column '{custom_id_col}' not found. Using empty string for custom IDs.")
        df = df.copy()
        df[custom_id_col] = ""

    # Initialize progress tracking
    progress_bar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            progress_bar = tqdm(total=len(df), desc="Retrieving responses", position=0, leave=True)
        except Exception:
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

    # Rate limiting state (shared across concurrent tasks via nonlocal)
    rate_limit_reset = 0
    rate_limit_remaining = batch_size

    async def retrieve_single(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a single response: poll until terminal status, with error retries."""
        response_id = row[response_id_col]
        custom_id = row.get(custom_id_col, "")

        # Skip if already processed
        if response_id in processed_ids:
            if progress_bar:
                progress_bar.update(1)
                progress_bar.refresh()
            if return_results:
                return {
                    'response_id': response_id,
                    'status': 'already_processed',
                    'openai_response': None,
                    'error': None
                }
            return None

        def _log(event_type: str, payload: dict):
            if logger:
                logger._add_entry({
                    'timestamp': datetime.now(timezone.utc),
                    'run_id': '',
                    'parent_run_id': '',
                    'custom_id': custom_id,
                    'event_type': event_type,
                    'logger_metadata': logger.logger_metadata_json,
                    'payload': json.dumps(payload),
                })

        _log('background_retrieval_attempt', {
            'response_id': response_id,
            'attempt_time': datetime.now(timezone.utc).isoformat(),
        })

        # Outer polling loop — keeps going while response is pending
        for poll_attempt in range(max_poll_attempts):
            nonlocal rate_limit_reset, rate_limit_remaining

            # Check rate limits before making a request
            if rate_limit_remaining <= 0 and time.time() < rate_limit_reset:
                wait_time = rate_limit_reset - time.time()
                await asyncio.sleep(wait_time)

            # Inner error-retry loop — retries only on transient failures
            response = None
            last_error = None
            for error_attempt in range(max_retries):
                try:
                    response = await asyncio.wait_for(
                        openai_client.responses.retrieve(response_id),
                        timeout=timeout,
                    )
                    # Update rate limit state from response headers
                    if hasattr(response, 'headers'):
                        headers = response.headers
                        if 'x-ratelimit-remaining-requests' in headers:
                            rate_limit_remaining = int(headers['x-ratelimit-remaining-requests'])
                        if 'x-ratelimit-reset-after' in headers:
                            rate_limit_reset = time.time() + float(headers['x-ratelimit-reset-after'])
                    last_error = None
                    break  # successful API call

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {timeout}s"
                    await asyncio.sleep(2 ** error_attempt)

                except Exception as e:
                    last_error = str(e)
                    if openai and isinstance(e, openai.RateLimitError):
                        delay = min(60, (2 ** error_attempt) * (1 + random.random() * 0.1))
                        await asyncio.sleep(delay)
                    elif any(str(e).startswith(f'5{x}') for x in '0123456789'):
                        await asyncio.sleep(2 ** error_attempt)
                    else:
                        break  # non-retryable client error

            # If all error retries exhausted without a successful response, give up
            if response is None:
                _log('background_retrieval_error', {
                    'response_id': response_id,
                    'error': last_error,
                    'error_type': 'retrieval_failed',
                    'attempts': max_retries,
                })
                failed_ids[response_id] = last_error
                processed_ids.add(response_id)
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.refresh()
                if return_results:
                    return {'response_id': response_id, 'status': 'failed', 'openai_response': None, 'error': last_error}
                return None

            # Serialize the response object
            if hasattr(response, 'model_dump'):
                response_data = response.model_dump(mode='json', by_alias=False)
            elif hasattr(response, 'to_dict'):
                response_data = response.to_dict()
            elif hasattr(response, '__dict__'):
                response_data = {k: v for k, v in response.__dict__.items() if not k.startswith('_')}
            else:
                response_data = {'response': str(response)}

            # Check the response status — prefer the serialized dict, fall back to
            # the attribute, then default to 'completed' for backward compatibility
            # (guards against mock objects or responses with no status field).
            _s = response_data.get('status')
            if not isinstance(_s, str) or not _s:
                _s = getattr(response, 'status', None)
            response_status = _s if isinstance(_s, str) and _s else 'completed'

            if response_status == 'completed':
                _log('background_retrieval_complete', {
                    'response_id': response_id,
                    'openai_response': response_data,
                    'status': 'completed',
                    'retrieval_time': datetime.now(timezone.utc).isoformat(),
                    'poll_attempts': poll_attempt + 1,
                })
                processed_ids.add(response_id)
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.refresh()
                if return_results:
                    return {'response_id': response_id, 'status': 'completed', 'openai_response': response_data, 'error': None}
                return None

            elif response_status in _TERMINAL_STATUSES:
                # failed / cancelled / expired
                error_msg = f"Response status: {response_status}"
                _log('background_retrieval_error', {
                    'response_id': response_id,
                    'error': error_msg,
                    'error_type': response_status,
                    'openai_response': response_data,
                    'poll_attempts': poll_attempt + 1,
                })
                failed_ids[response_id] = error_msg
                processed_ids.add(response_id)
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.refresh()
                if return_results:
                    return {'response_id': response_id, 'status': response_status, 'openai_response': response_data, 'error': error_msg}
                return None

            else:
                # Still pending ("in_progress", "queued", etc.) — log and wait
                _log('background_retrieval_pending', {
                    'response_id': response_id,
                    'openai_status': response_status,
                    'poll_attempt': poll_attempt + 1,
                    'next_check_in': poll_interval,
                })
                await asyncio.sleep(poll_interval)

        # Exhausted max_poll_attempts without a terminal status
        error_msg = f"Exceeded max_poll_attempts ({max_poll_attempts}) — response never completed"
        _log('background_retrieval_error', {
            'response_id': response_id,
            'error': error_msg,
            'error_type': 'poll_timeout',
            'poll_attempts': max_poll_attempts,
        })
        failed_ids[response_id] = error_msg
        processed_ids.add(response_id)
        if progress_bar:
            progress_bar.update(1)
            progress_bar.refresh()
        if return_results:
            return {'response_id': response_id, 'status': 'poll_timeout', 'openai_response': None, 'error': error_msg}
        return None

    # Process in batches (controls max concurrency)
    rows = df.to_dict('records')

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]

        batch_results = await asyncio.gather(
            *[retrieve_single(row) for row in batch],
            return_exceptions=True,
        )

        for row, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                response_id = row[response_id_col]
                error_msg = str(result)
                failed_ids[response_id] = error_msg
                processed_ids.add(response_id)
                if progress_bar:
                    progress_bar.update(1)
                    progress_bar.refresh()
                if return_results:
                    results.append({
                        'response_id': response_id,
                        'status': 'error',
                        'openai_response': None,
                        'error': error_msg,
                    })
            elif result is not None and return_results:
                results.append(result)

            # Add to checkpoint batch
            if checkpoint_file:
                checkpoint_batch.append({
                    'response_id': row[response_id_col],
                    'processed': True,
                    'error': failed_ids.get(row[response_id_col]),
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

    print(f"\nRetrieval complete: {len(processed_ids)} processed, {len(failed_ids)} failed")

    if return_results:
        return pd.DataFrame(results)
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
