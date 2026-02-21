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
from .config import StorageConfig, JobConfig, RetrievalConfig
from .storage import create_storage, LocalStorage, S3Storage

# OpenAI response statuses that mean "done, stop polling"
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
# Statuses that indicate the response is still being processed
_PENDING_STATUSES = {"in_progress", "queued", "processing"}

# Background retrieval event types (all share the same run_id = response_id)
_BACKGROUND_EVENT_TYPES = {
    'background_retrieval_attempt',
    'background_retrieval_pending',
    'background_retrieval_complete',
    'background_retrieval_error',
}
# Terminal event types — a response with one of these as its latest event is done
_TERMINAL_RETRIEVAL_EVENT_TYPES = {
    'background_retrieval_complete',
    'background_retrieval_error',
}


def _query_pending_responses(storage) -> "pd.DataFrame":
    """Read parquet files from storage and find responses with non-terminal status.

    Uses ``run_id`` (= response_id) and ``event_type`` to determine which responses
    are still pending.  A response is pending when its latest logged event_type is
    NOT one of the terminal types (``background_retrieval_complete`` /
    ``background_retrieval_error``).

    Returns:
        DataFrame with columns: response_id, custom_id
    """
    if pd is None:
        raise ImportError(
            "pandas is required for auto-discovery. Install with: pip install pandas"
        )

    try:
        files = storage.list_files()
    except Exception as e:
        warnings.warn(f"Failed to list files from storage: {e}")
        return pd.DataFrame(columns=['response_id', 'custom_id'])

    if not files:
        return pd.DataFrame(columns=['response_id', 'custom_id'])

    frames = []
    for f in files:
        try:
            table = storage.read_table(Path(f))
            df = table.to_pandas()
            bg_df = df[df['event_type'].isin(_BACKGROUND_EVENT_TYPES)]
            if not bg_df.empty:
                frames.append(bg_df[['run_id', 'custom_id', 'event_type', 'timestamp']])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=['response_id', 'custom_id'])

    all_df = pd.concat(frames, ignore_index=True)

    # For each run_id (= response_id), find the latest event by timestamp
    latest = all_df.sort_values('timestamp').groupby('run_id').last().reset_index()

    # Keep only non-terminal entries (responses still pending)
    pending = latest[~latest['event_type'].isin(_TERMINAL_RETRIEVAL_EVENT_TYPES)]

    if pending.empty:
        return pd.DataFrame(columns=['response_id', 'custom_id'])

    return pd.DataFrame({
        'response_id': pending['run_id'].values,
        'custom_id': pending['custom_id'].values,
    })


async def retrieve_background_responses(
    df: Optional["pd.DataFrame"] = None,
    openai_client=None,
    logger: Optional[ParquetLogger] = None,
    storage_config: Optional[StorageConfig] = None,
    job_config: Optional[JobConfig] = None,
    retrieval_config: Optional[RetrievalConfig] = None,
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

    All log entries follow the standard SCHEMA (same as ParquetLogger) with
    ``run_id`` set to the OpenAI ``response_id`` for easy top-level querying.

    **Auto-discovery mode** (default when no ``df`` is supplied):
        Pass ``storage_config`` (and optionally ``job_config``) to point the retriever
        at the same storage location used by ``batch_process()``.  The retriever reads
        existing parquet files, finds every response whose latest logged event is
        non-terminal (``background_retrieval_attempt`` or ``background_retrieval_pending``),
        and retries only those.

    **Explicit mode** (pass a ``df`` directly):
        Provide a DataFrame with ``response_id`` and ``custom_id`` columns, as before.

    Args:
        df: Optional DataFrame containing response IDs to retrieve.  If omitted,
            ``storage_config`` must be provided and responses are auto-discovered
            from existing log files.
        openai_client: Initialized OpenAI async client (created automatically if None)
        logger: Optional ParquetLogger instance for logging.  Auto-created from
            ``storage_config`` when ``df`` is None and no logger is supplied.
        storage_config: StorageConfig used to locate existing log files for
            auto-discovery and to write new log entries.
        job_config: Optional JobConfig used to resolve the full storage path (same
            object passed to ``batch_process()``).
        retrieval_config: Optional RetrievalConfig controlling polling behavior and
            the auto-discovery source (``source="local"`` or ``source="s3"``).
            Defaults to ``RetrievalConfig()`` (local, conservative polling settings).
        response_id_col: Column name containing response IDs (default: "response_id")
        custom_id_col: Column name containing custom IDs (default: "custom_id")
        batch_size: Number of concurrent requests (default: 50)
        max_retries: Maximum retries per API call for transient errors (default: 3)
        timeout: Timeout per individual API request in seconds (default: 30.0)
        poll_interval: Seconds to wait between status checks (default: 30.0)
        max_poll_attempts: Maximum polls per response before giving up (default: 40)
        show_progress: Show progress bar (default: True)
        checkpoint_file: Optional path to checkpoint file for resume capability
        return_results: If False, don't keep results in memory (default: True)

    Returns:
        DataFrame with retrieval results if return_results=True, else None.
        Result columns: response_id, status, openai_response, error

    Example — auto-discovery::

        >>> results = await retrieve_background_responses(
        ...     storage_config=StorageConfig(output_dir="./my_logs"),
        ...     job_config=JobConfig(category="research"),
        ...     openai_client=client,
        ... )

    Example — explicit DataFrame::

        >>> df = pd.DataFrame({
        ...     'response_id': ['resp_123', 'resp_456'],
        ...     'custom_id': ['user-001', 'user-002']
        ... })
        >>> with ParquetLogger('./logs') as logger:
        ...     results = await retrieve_background_responses(df, client, logger=logger)
    """
    if pd is None:
        raise ImportError(
            "pandas is required for background retrieval. Install with: pip install pandas"
        )

    if openai_client is None:
        if openai is None:
            raise ImportError(
                "openai is required for background retrieval. Install with: pip install openai"
            )
        openai_client = openai.AsyncOpenAI()

    # Suppress Pydantic serialization warnings globally
    warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic")

    # Resolve retrieval configuration (source, polling params, etc.)
    rc = retrieval_config or RetrievalConfig()

    # --- Auto-discovery mode ---
    if df is None:
        if storage_config is None:
            raise ValueError(
                "Either 'df' or 'storage_config' must be provided. "
                "Pass a DataFrame with response IDs, or a StorageConfig to "
                "auto-discover pending responses from existing log files."
            )
        # Resolve storage path (reuse same logic as batch_process)
        from .batch import _build_storage_paths
        local_path, resolved_s3_config = _build_storage_paths(
            job_config or JobConfig(), storage_config
        )

        # Select a single backend based on retrieval_config.source
        if rc.source == "s3":
            if resolved_s3_config is None:
                raise ValueError(
                    "retrieval_config.source='s3' requires storage_config.s3_config to be set."
                )
            read_storage = S3Storage(resolved_s3_config)
        else:  # "local" (default)
            read_storage = LocalStorage(local_path)

        df = _query_pending_responses(read_storage)

        if df.empty:
            print("No pending responses found in storage.")
            return pd.DataFrame() if return_results else None

        print(f"Auto-discovered {len(df)} pending response(s) from storage.")

        # Auto-create logger pointing at the same full path if not supplied
        # (always write to all configured backends, regardless of read source)
        if logger is None:
            logger = ParquetLogger(log_dir=str(local_path), s3_config=resolved_s3_config)

    # --- Validate required columns ---
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

        def _log(event_type: str, data: dict):
            """Log a background retrieval event using the standard SCHEMA format.

            Entries match the structure produced by ParquetLogger._log_event():
            - run_id is set to response_id for top-level queryability
            - payload contains execution wrapper + data + raw sections
            """
            if logger:
                ts = datetime.now(timezone.utc)
                payload = {
                    "event_type": event_type,
                    "timestamp": ts.isoformat(),
                    "execution": {
                        "run_id": response_id,
                        "parent_run_id": "",
                        "custom_id": custom_id,
                        "tags": [],
                        "metadata": {}
                    },
                    "data": data,
                    "raw": {}
                }
                logger._add_entry({
                    'timestamp': ts,
                    'run_id': response_id,
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
