"""
Background response retrieval for OpenAI responses.

Retrieves completed responses from OpenAI's API for background/async requests,
with automatic logging to Parquet files, rate limiting, and checkpoint support.
"""

import asyncio
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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
from .config import StorageConfig, JobConfig, RetrievalConfig, ColumnConfig, S3Config
from .storage import create_storage, LocalStorage, S3Storage

# OpenAI response statuses that mean "done, stop polling"
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}
# Statuses that indicate the response is still being processed
_PENDING_STATUSES = {"in_progress", "queued", "processing"}

# Background retrieval event types (run_id = LangChain UUID; response_id stored in data.response_id)
_BACKGROUND_EVENT_TYPES = {
    'background_retrieval_queued',     # logged by Batch.run() when a background response is detected
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

    Uses ``event_type`` to determine which responses are still pending.
    A response is pending when its latest logged event_type is NOT one of the
    terminal types (``background_retrieval_complete`` / ``background_retrieval_error``).

    ``response_id`` is read from ``payload.data.response_id`` (current schema where
    ``run_id`` is the LangChain UUID). Events without ``data.response_id`` in the
    payload are excluded.

    Returns:
        DataFrame with columns: response_id, custom_id, run_id, parent_run_id, tags
    """
    import json as _json

    _EMPTY = ['response_id', 'custom_id', 'run_id', 'parent_run_id', 'tags']

    if pd is None:
        raise ImportError(
            "pandas is required for auto-discovery. Install with: pip install pandas"
        )

    try:
        files = storage.list_files()
    except Exception as e:
        warnings.warn(f"Failed to list files from storage: {e}")
        return pd.DataFrame(columns=_EMPTY)

    if not files:
        return pd.DataFrame(columns=_EMPTY)

    frames = []
    for f in files:
        try:
            table = storage.read_table(Path(f))
            df = table.to_pandas()
            bg_df = df[df['event_type'].isin(_BACKGROUND_EVENT_TYPES)]
            if not bg_df.empty:
                frames.append(bg_df[['run_id', 'parent_run_id', 'custom_id', 'event_type', 'timestamp', 'payload']])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=_EMPTY)

    all_df = pd.concat(frames, ignore_index=True)

    # Extract OpenAI response_id and tags from payload JSON.
    # run_id = LangChain UUID (top-level column); data.response_id = OpenAI resp_xxx.
    # Events without data.response_id are excluded (empty string → filtered below).
    def _extract_resp_id_and_tags(row) -> "pd.Series":
        try:
            parsed = _json.loads(row['payload'])
            resp_id = parsed.get('data', {}).get('response_id') or ''
            tags = parsed.get('execution', {}).get('tags', []) or []
        except Exception:
            resp_id = ''
            tags = []
        return pd.Series({'effective_response_id': resp_id, 'extracted_tags': tags})

    extracted = all_df.apply(_extract_resp_id_and_tags, axis=1)
    all_df['effective_response_id'] = extracted['effective_response_id']
    all_df['extracted_tags'] = extracted['extracted_tags']

    # Drop rows with no response_id (not background retrieval events we care about)
    all_df = all_df[all_df['effective_response_id'] != '']

    if all_df.empty:
        return pd.DataFrame(columns=_EMPTY)

    # For each response, find the latest event by timestamp
    latest = all_df.sort_values('timestamp').groupby('effective_response_id').last().reset_index()

    # Keep only non-terminal entries (responses still pending)
    pending = latest[~latest['event_type'].isin(_TERMINAL_RETRIEVAL_EVENT_TYPES)]

    if pending.empty:
        return pd.DataFrame(columns=_EMPTY)

    return pd.DataFrame({
        'response_id': pending['effective_response_id'].values,
        'custom_id': pending['custom_id'].values,
        'run_id': pending['run_id'].values,
        'parent_run_id': pending['parent_run_id'].values,
        'tags': pending['extracted_tags'].values,
    })


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    """Parse an S3 URI into (bucket, prefix).

    Examples::

        >>> _parse_s3_uri("s3://my-bucket/path/to/logs/")
        ('my-bucket', 'path/to/logs/')
        >>> _parse_s3_uri("s3://my-bucket")
        ('my-bucket', '')
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with 's3://'): {uri!r}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    if not bucket:
        raise ValueError(f"S3 URI has no bucket: {uri!r}")
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


async def retrieve_background_responses(
    df: Optional["pd.DataFrame"] = None,
    openai_client=None,
    logger: Optional[ParquetLogger] = None,
    storage_config: Optional[StorageConfig] = None,
    job_config: Optional[JobConfig] = None,
    retrieval_config: Optional[RetrievalConfig] = None,
    column_config: Optional[ColumnConfig] = None,
    response_parser=None,
    s3_path: Optional[str] = None,
) -> Optional["pd.DataFrame"]:
    """
    Retrieve background responses from OpenAI and log them to Parquet.

    Polls each response until its status is "completed" (or a terminal failure state),
    logging each intermediate "in_progress"/"queued" check as a
    ``background_retrieval_pending`` event and the final outcome as either
    ``background_retrieval_complete`` or ``background_retrieval_error``.

    **S3 resume mode** (simplest for cross-process resume):
        Pass ``s3_path="s3://bucket/prefix/"`` to auto-discover pending responses
        from S3 and write new retrieval events back to the same S3 location.
        No ``storage_config`` or ``job_config`` needed.

    **Auto-discovery mode** (default when no ``df`` is supplied):
        Pass ``storage_config`` (and optionally ``job_config``) to point the retriever
        at the same storage location used by ``batch_process()``.  The retriever reads
        existing parquet files, finds every response whose latest logged event is
        non-terminal (``background_retrieval_attempt``, ``background_retrieval_pending``,
        or ``background_retrieval_queued``), and retries only those.

    **Explicit mode** (pass a ``df`` directly):
        Provide a DataFrame with ``response_id`` and ``custom_id`` columns (or
        the column names configured in ``column_config``).

    Args:
        df: Optional DataFrame containing response IDs to retrieve.  If omitted,
            ``storage_config`` or ``s3_path`` must be provided and responses are
            auto-discovered from existing log files.
        openai_client: Initialized OpenAI async client (created automatically if None)
        logger: Optional ParquetLogger instance for logging.  Auto-created from
            ``storage_config`` or ``s3_path`` when no logger is supplied.
        storage_config: StorageConfig used to locate existing log files for
            auto-discovery and to write new log entries.
        job_config: Optional JobConfig used to resolve the full storage path (same
            object passed to ``Batch``).
        retrieval_config: Optional RetrievalConfig controlling all polling behavior
            (poll_interval, max_poll_attempts, batch_size, timeout, max_retries,
            show_progress, checkpoint_file, return_results, source).
            Defaults to ``RetrievalConfig()`` (local, conservative polling settings).
        column_config: Optional ColumnConfig specifying which DataFrame columns hold
            the response_id and custom_id values.  Defaults to ``ColumnConfig()``
            (columns named "response_id" and "custom_id").
        s3_path: Optional S3 URI (e.g. ``"s3://my-bucket/path/to/logs/"``).  When
            provided, pending responses are auto-discovered from that S3 location and
            new retrieval events are written back there.  Takes precedence over
            ``storage_config``-based discovery when ``df`` is None.

    Returns:
        DataFrame with retrieval results if rc.return_results=True, else None.
        Result columns: response_id, status, openai_response, error

    Example — S3 resume (cross-process)::

        >>> results = await retrieve_background_responses(
        ...     s3_path="s3://my-bucket/jobs/research/",
        ...     openai_client=client,
        ... )

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

    # Resolve configuration objects
    rc = retrieval_config or RetrievalConfig()
    cc = column_config or ColumnConfig()

    # --- S3 path shortcut (pre-process before standard auto-discovery) ---
    if s3_path is not None:
        import tempfile
        bucket, prefix = _parse_s3_uri(s3_path)
        s3_cfg = S3Config(bucket=bucket, prefix=prefix)
        if df is None:
            read_storage = S3Storage(s3_cfg)
            df = _query_pending_responses(read_storage)
            if df.empty:
                if rc.show_progress:
                    print("No pending responses found in S3.")
                return pd.DataFrame() if rc.return_results else None
            if rc.show_progress:
                print(f"Auto-discovered {len(df)} pending response(s) from S3.")
        if logger is None:
            logger = ParquetLogger(
                log_dir=tempfile.mkdtemp(prefix="bgretrieval_"),
                s3_config=s3_cfg,
            )

    # --- Auto-discovery mode ---
    if df is None:
        if storage_config is None:
            raise ValueError(
                "Either 'df' or 'storage_config' must be provided. "
                "Pass a DataFrame with response IDs, or a StorageConfig to "
                "auto-discover pending responses from existing log files."
            )
        # Resolve storage path (reuse same logic as Batch)
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
            if rc.show_progress:
                print("No pending responses found in storage.")
            return pd.DataFrame() if rc.return_results else None

        if rc.show_progress:
            print(f"Auto-discovered {len(df)} pending response(s) from storage.")

        # Auto-create logger pointing at the same full path if not supplied
        # (always write to all configured backends, regardless of read source)
        if logger is None:
            logger = ParquetLogger(log_dir=str(local_path), s3_config=resolved_s3_config)

    # --- Validate required columns ---
    if cc.response_id not in df.columns:
        raise ValueError(f"Column '{cc.response_id}' not found in DataFrame")
    if cc.custom_id not in df.columns:
        warnings.warn(f"Column '{cc.custom_id}' not found. Using empty string for custom IDs.")
        df = df.copy()
        df[cc.custom_id] = ""

    # Initialize manual progress tracking
    total_count = len(df)
    progress_interval = 1 if total_count <= 20 else max(1, total_count // 10)

    # Load checkpoint if exists
    processed_ids = set()
    failed_ids = {}
    if rc.checkpoint_file and Path(rc.checkpoint_file).exists():
        try:
            checkpoint_df = pd.read_parquet(rc.checkpoint_file)
            processed_ids = set(checkpoint_df['response_id'].values)
            if 'error' in checkpoint_df.columns:
                failed_ids = dict(zip(
                    checkpoint_df[checkpoint_df['error'].notna()]['response_id'],
                    checkpoint_df[checkpoint_df['error'].notna()]['error']
                ))
            if rc.show_progress:
                print(f"Resumed from checkpoint: {len(processed_ids)} already processed")
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint: {e}")

    # Prepare results storage
    results = [] if rc.return_results else None
    checkpoint_batch = []

    # Rate limiting state (shared across concurrent tasks via nonlocal)
    rate_limit_reset = 0
    rate_limit_remaining = rc.batch_size

    async def retrieve_single(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Retrieve a single response: poll until terminal status, with error retries."""
        response_id = row[cc.response_id]
        custom_id = row.get(cc.custom_id, "")

        # Use the original LangChain run_id/parent_run_id when available (captured by
        # _BackgroundRunIdCapture in Batch.run()) so that retrieval events share the
        # same top-level run_id column as llm_start/llm_end, enabling simple JOINs.
        # Fall back to response_id for backward compat when called without Batch.
        llm_run_id = row.get('run_id') or response_id
        llm_parent_run_id = row.get('parent_run_id') or ''
        tags = list(row.get('tags') or [])

        # Skip if already processed
        if response_id in processed_ids:
            if rc.return_results:
                return {
                    'response_id': response_id,
                    'status': 'already_processed',
                    'openai_response': None,
                    'error': None
                }
            return None

        def _log(event_type: str, data: dict, raw: Optional[dict] = None):
            """Log a background retrieval event using the standard SCHEMA format.

            Delegates to logger._log_event() so all entries go through the same
            code path as regular ParquetLogger events, including _safe_json_dumps().
            ``raw`` is optional; for completed events it should be the full response dict.
            """
            if logger:
                payload = {
                    "event_type": event_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "execution": {
                        "run_id": llm_run_id,
                        "parent_run_id": llm_parent_run_id,
                        "custom_id": custom_id,
                        "tags": tags,
                        "metadata": {}
                    },
                    "data": data,
                    "raw": raw or {}
                }
                logger._log_event(payload)

        _log('background_retrieval_attempt', {
            'response_id': response_id,
            'attempt_time': datetime.now(timezone.utc).isoformat(),
        })

        # Outer polling loop — keeps going while response is pending
        for poll_attempt in range(rc.max_poll_attempts):
            nonlocal rate_limit_reset, rate_limit_remaining

            # Check rate limits before making a request
            if rate_limit_remaining <= 0 and time.time() < rate_limit_reset:
                wait_time = rate_limit_reset - time.time()
                await asyncio.sleep(wait_time)

            # Inner error-retry loop — retries only on transient failures
            response = None
            last_error = None
            for error_attempt in range(rc.max_retries):
                try:
                    response = await asyncio.wait_for(
                        openai_client.responses.retrieve(response_id),
                        timeout=rc.timeout,
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
                    last_error = f"Timeout after {rc.timeout}s"
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
                    'attempts': rc.max_retries,
                }, raw={'error': last_error})
                failed_ids[response_id] = last_error
                processed_ids.add(response_id)
                if rc.return_results:
                    return {'response_id': response_id, 'status': 'failed', 'openai_response': None, 'error': last_error}
                return None

            # Extract output_text BEFORE model_dump — it's a computed property on the
            # OpenAI Response object that is not serialized by model_dump().
            # Storing it at the top level lets _parse_from_openai_response() find it
            # without navigating the nested output[].content[] array (which can have
            # null content items for reasoning outputs).
            output_text = getattr(response, 'output_text', None)

            # Serialize the response object
            if hasattr(response, 'model_dump'):
                response_data = response.model_dump(mode='json', by_alias=False)
            elif hasattr(response, 'to_dict'):
                response_data = response.to_dict()
            elif hasattr(response, '__dict__'):
                response_data = {k: v for k, v in response.__dict__.items() if not k.startswith('_')}
            else:
                response_data = {'response': str(response)}

            # Store output_text at top level for easy access
            if output_text:
                response_data['output_text'] = output_text

            # Check the response status — prefer the serialized dict, fall back to
            # the attribute, then default to 'completed' for backward compatibility
            # (guards against mock objects or responses with no status field).
            _s = response_data.get('status')
            if not isinstance(_s, str) or not _s:
                _s = getattr(response, 'status', None)
            response_status = _s if isinstance(_s, str) and _s else 'completed'

            if response_status == 'completed':
                # Apply structured output parser if provided (dict for Parquet serialization)
                parsed_output = None
                if response_parser is not None:
                    try:
                        parsed_output = response_parser(response_data)
                    except Exception:
                        pass

                _log('background_retrieval_complete', {
                    'response_id': response_id,
                    'openai_response': response_data,
                    'parsed_output': parsed_output,
                    'status': 'completed',
                    'retrieval_time': datetime.now(timezone.utc).isoformat(),
                    'poll_attempts': poll_attempt + 1,
                }, raw=response_data)
                processed_ids.add(response_id)
                if rc.return_results:
                    return {'response_id': response_id, 'status': 'completed', 'openai_response': response_data, 'parsed_output': parsed_output, 'error': None}
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
                }, raw=response_data)
                failed_ids[response_id] = error_msg
                processed_ids.add(response_id)
                if rc.return_results:
                    return {'response_id': response_id, 'status': response_status, 'openai_response': response_data, 'error': error_msg}
                return None

            else:
                # Still pending ("in_progress", "queued", etc.) — log and wait
                _log('background_retrieval_pending', {
                    'response_id': response_id,
                    'openai_status': response_status,
                    'poll_attempt': poll_attempt + 1,
                    'next_check_in': rc.poll_interval,
                })
                await asyncio.sleep(rc.poll_interval)

        # Exhausted max_poll_attempts without a terminal status
        error_msg = f"Exceeded max_poll_attempts ({rc.max_poll_attempts}) — response never completed"
        _log('background_retrieval_error', {
            'response_id': response_id,
            'error': error_msg,
            'error_type': 'poll_timeout',
            'poll_attempts': rc.max_poll_attempts,
        }, raw={'error': error_msg})
        failed_ids[response_id] = error_msg
        processed_ids.add(response_id)
        if rc.return_results:
            return {'response_id': response_id, 'status': 'poll_timeout', 'openai_response': None, 'error': error_msg}
        return None

    # Process in batches (controls max concurrency)
    rows = df.to_dict('records')
    completed_count = len(processed_ids)  # account for checkpoint-resumed items

    for i in range(0, len(rows), rc.batch_size):
        batch = rows[i:i + rc.batch_size]

        batch_results = await asyncio.gather(
            *[retrieve_single(row) for row in batch],
            return_exceptions=True,
        )

        for row, result in zip(batch, batch_results):
            if isinstance(result, Exception):
                response_id = row[cc.response_id]
                error_msg = str(result)
                failed_ids[response_id] = error_msg
                processed_ids.add(response_id)
                if rc.return_results:
                    results.append({
                        'response_id': response_id,
                        'status': 'error',
                        'openai_response': None,
                        'error': error_msg,
                    })
            elif result is not None and rc.return_results:
                results.append(result)

            # Add to checkpoint batch
            if rc.checkpoint_file:
                checkpoint_batch.append({
                    'response_id': row[cc.response_id],
                    'processed': True,
                    'error': failed_ids.get(row[cc.response_id]),
                })

        # Print progress after each gather batch completes
        completed_count += len(batch)
        if rc.show_progress and (completed_count % progress_interval == 0 or completed_count == total_count):
            print(f"  Retrieved {completed_count}/{total_count} responses ({100 * completed_count // total_count}%)")

        # Save checkpoint periodically
        if rc.checkpoint_file and len(checkpoint_batch) >= 100:
            save_checkpoint(rc.checkpoint_file, checkpoint_batch)
            checkpoint_batch = []

    # Final checkpoint save
    if rc.checkpoint_file and checkpoint_batch:
        save_checkpoint(rc.checkpoint_file, checkpoint_batch)

    # Final flush of logger buffer
    if logger and hasattr(logger, 'flush'):
        logger.flush()

    if rc.show_progress:
        print(f"\nRetrieval complete: {len(processed_ids)} processed, {len(failed_ids)} failed")

    if rc.return_results:
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
