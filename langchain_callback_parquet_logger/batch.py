"""Batch processing utilities for DataFrame operations with LangChain."""

import asyncio
import os
import re
import warnings
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

import pandas as pd
from langchain_core.runnables import RunnableLambda

from .logger import ParquetLogger
from .config import (
    JobConfig, StorageConfig, ProcessingConfig, ColumnConfig,
    S3Config, EventType, LLMConfig, RetrievalConfig
)
from .tagging import with_tags, extract_custom_id


@dataclass
class BatchOutcome:
    """Return value from Batch.run_and_retrieve() containing both processing and retrieval results."""
    batch_results: Optional[List]
    retrieval_results: Optional[pd.DataFrame]


# LLM class names known to support OpenAI's Responses API in background mode
_OPENAI_BACKGROUND_CLASSES = frozenset({'ChatOpenAI', 'AzureChatOpenAI'})
# Statuses indicating a response is still in-flight (not yet completed/failed)
_OPENAI_PENDING_STATUSES = frozenset({'in_progress', 'queued', 'processing'})


def _openai_response_id_extractor(result: Any, row: dict) -> Optional[str]:
    """Extract background response ID from a LangChain OpenAI result.

    Returns the response ID only when ``response_metadata['status']`` is a pending
    status (background mode). Returns None for synchronous completions so they are
    not inadvertently queued for retrieval polling.
    """
    meta = getattr(result, 'response_metadata', None)
    if not isinstance(meta, dict):
        return None
    if meta.get('status') not in _OPENAI_PENDING_STATUSES:
        return None
    return meta.get('id')


def _try_extract_id_from_exception(exc: BaseException) -> Optional[str]:
    """Extract a pending background response ID from a structured-output ValueError.

    When ``with_structured_output()`` wraps a background LLM, queued responses have
    no content, causing LangChain to raise ValueError.  The stringified exception
    contains the full response metadata (id + status), so we recover the ID via regex.
    Returns None for any other exception type or non-pending status.
    """
    msg = str(exc)
    id_match = re.search(r"\bid='(resp_[A-Za-z0-9]+)'", msg)
    status_match = re.search(r"'status':\s*'([^']+)'", msg)
    if id_match and status_match and status_match.group(1) in _OPENAI_PENDING_STATUSES:
        return id_match.group(1)
    return None


def _parse_from_openai_response(response_dict: dict, schema: Type) -> Optional[Any]:
    """Parse structured output from a serialized OpenAI Response dict.

    ``retrieve_background_responses()`` stores the response as
    ``model_dump(mode='json')``, so the JSON text for structured-output responses
    lives at ``output[0].content[0].text`` (type == 'output_text').
    Returns a validated Pydantic instance, or None on any failure.
    """
    if not response_dict or not isinstance(response_dict, dict):
        return None
    # Try convenience property first, then navigate the output array
    text: Optional[str] = response_dict.get('output_text')
    if not text:
        for item in response_dict.get('output', []):
            for content in (item.get('content') or []):
                if content.get('type') == 'output_text':
                    text = content.get('text')
                    break
            if text:
                break
    if not text:
        return None
    try:
        return schema.model_validate_json(text)
    except Exception:
        import json as _json
        try:
            return schema(**_json.loads(text))
        except Exception:
            return None


async def _batch_run(
    df: pd.DataFrame,
    llm: Any,
    prompt_col: str = "prompt",
    config_col: str = "config",
    tools_col: Optional[str] = "tools",
    max_concurrency: int = 10,
    show_progress: bool = True,
    return_exceptions: bool = True,
    return_results: bool = True,
    row_timeout: Optional[float] = None,
) -> Optional[list]:
    """
    Internal async batch processing engine for DataFrames.

    **Most users should use Batch instead**, which includes automatic logging.

    Args:
        df: DataFrame with prepared data
        llm: LangChain LLM (already configured with callbacks, structured output, etc.)
        prompt_col: Column name containing prompts (default: "prompt")
        config_col: Column name containing config dicts from with_tags() (default: "config")
        tools_col: Column name containing tools lists (default: "tools", set None to skip)
        max_concurrency: Maximum concurrent requests
        show_progress: Show progress bar (auto-detects notebook vs terminal)
        return_exceptions: Return exceptions instead of raising
        return_results: If False, don't keep results in memory (useful for huge DataFrames)
        row_timeout: Per-row timeout in seconds (None = no timeout). Useful with
            slow service tiers (e.g. OpenAI flex) to prevent indefinite hangs.

    Returns:
        List of results in same order as DataFrame rows, or None if return_results=False
    """
    # Suppress Pydantic serialization warnings globally so it covers
    # LangChain/OpenAI SDK internals calling model_dump() during ainvoke()
    warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic")

    rows = df.to_dict('records')
    total_count = len(rows)
    completed_count = 0
    # Print after every row for small batches, ~10% intervals for large ones
    progress_interval = 1 if total_count <= 20 else max(1, total_count // 10)

    async def process_row(row: dict) -> Any:
        """Process a single row from the DataFrame."""
        nonlocal completed_count

        # Build invoke kwargs from DataFrame columns
        invoke_kwargs = {"input": row.get(prompt_col)}

        # Add config if column exists
        if config_col in row and row[config_col] is not None:
            invoke_kwargs["config"] = row[config_col]

        # Add tools if column exists and not None
        if tools_col and tools_col in row and row[tools_col] is not None:
            invoke_kwargs["tools"] = row[tools_col]

        try:
            if row_timeout:
                result = await asyncio.wait_for(
                    llm.ainvoke(**invoke_kwargs),
                    timeout=row_timeout
                )
            else:
                result = await llm.ainvoke(**invoke_kwargs)
            return result
        except asyncio.TimeoutError:
            if return_exceptions:
                return TimeoutError(f"Row timed out after {row_timeout}s")
            raise TimeoutError(f"Row timed out after {row_timeout}s")
        except Exception as e:
            if return_exceptions:
                return e
            raise
        finally:
            completed_count += 1
            if show_progress and (completed_count % progress_interval == 0 or completed_count == total_count):
                print(f"  Processed {completed_count}/{total_count} rows ({100 * completed_count // total_count}%)")

    # Create runner and process batch
    runner = RunnableLambda(process_row)

    if return_results:
        # Normal mode: collect and return results
        results = await runner.abatch(
            rows,
            config={"max_concurrency": max_concurrency},
            return_exceptions=return_exceptions
        )
        return results
    else:
        # Memory-efficient mode: don't collect results
        await runner.abatch(
            rows,
            config={"max_concurrency": max_concurrency},
            return_exceptions=return_exceptions
        )
        return None


def _build_storage_paths(
    job_config: JobConfig,
    storage_config: StorageConfig,
) -> tuple:
    """Resolve local path and S3 config from job/storage configs.

    Shared by Batch and retrieve_background_responses() so both functions
    write to the same directory structure.

    Returns:
        (local_path, s3_config) where s3_config has its prefix already resolved
        to the full path (or None if S3 not configured).
    """
    version_safe = (job_config.version or 'unversioned').replace('.', '_')
    template_vars = {
        'job_category': job_config.category,
        'job_subcategory': job_config.subcategory or 'default',
        'environment': job_config.environment or 'production',
        'job_version': job_config.version,
        'job_version_safe': version_safe,
        'date': date.today().isoformat(),
    }

    local_path = Path(storage_config.output_dir) / storage_config.path_template.format(**template_vars)

    # Check for S3 bucket in environment if not configured — use a local variable,
    # never mutate the caller's storage_config object
    s3_config = storage_config.s3_config
    if not s3_config and os.environ.get('LANGCHAIN_S3_BUCKET'):
        s3_config = S3Config(bucket=os.environ['LANGCHAIN_S3_BUCKET'])

    resolved_s3_config = None
    if s3_config:
        import copy
        resolved_s3_config = copy.copy(s3_config)
        base_prefix = resolved_s3_config.prefix.rstrip('/')
        formatted_path = storage_config.path_template.format(**template_vars).lstrip('/')
        resolved_s3_config.prefix = f"{base_prefix}/{formatted_path}/" if base_prefix else f"{formatted_path}/"

    return local_path, resolved_s3_config


class Batch:
    """
    Orchestrates a complete batch processing job lifecycle.

    Resolves storage paths once, then provides ``run()``, ``retrieve()``, and
    ``run_and_retrieve()`` methods that share the same paths and configuration.
    Background response IDs captured during ``run()`` are automatically available
    to ``retrieve()``, enabling a seamless run-then-retrieve workflow.

    Args:
        job_config: Job metadata (category, subcategory, version, environment).
        storage_config: Storage paths and optional S3 configuration.
        processing_config: Concurrency, buffer size, and other processing settings.
        column_config: DataFrame column name mapping.

    Example — standard batch (no background processing)::

        >>> batch = Batch(
        ...     job_config=JobConfig(category="analysis", version="1.0"),
        ...     storage_config=StorageConfig(output_dir="./logs"),
        ... )
        >>> results = await batch.run(df, llm_config)

    Example — background processing (run then retrieve)::

        >>> def extract_response_id(result, row):
        ...     return getattr(result, 'response_metadata', {}).get('id')
        ...
        >>> batch = Batch(
        ...     job_config=JobConfig(category="analysis"),
        ...     storage_config=StorageConfig(output_dir="./logs"),
        ... )
        >>> outcome = await batch.run_and_retrieve(
        ...     df, llm_config,
        ...     openai_client=AsyncOpenAI(),
        ...     response_id_extractor=extract_response_id,
        ... )
        >>> batch_results = outcome.batch_results
        >>> retrieval_df = outcome.retrieval_results

    Example — inspect batch results before retrieval::

        >>> batch_results = await batch.run(df, llm_config,
        ...                                  response_id_extractor=extract_response_id)
        >>> print(f"Captured {len(batch.pending_response_ids)} background response(s)")
        >>> retrieval_df = await batch.retrieve(openai_client=AsyncOpenAI())
    """

    def __init__(
        self,
        job_config: Optional[JobConfig] = None,
        storage_config: Optional[StorageConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        column_config: Optional[ColumnConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ):
        self.job_config = job_config or JobConfig()
        self.storage_config = storage_config or StorageConfig()
        self.processing_config = processing_config or ProcessingConfig()
        self.column_config = column_config or ColumnConfig()
        self.retrieval_config = retrieval_config or RetrievalConfig()

        # Resolve storage paths once — shared by run() and retrieve()
        self.local_path, self.resolved_s3_config = _build_storage_paths(
            self.job_config, self.storage_config
        )
        self.local_path.mkdir(parents=True, exist_ok=True)

        # Background response IDs captured during run() — response_id → custom_id
        self._background_response_ids: Dict[str, str] = {}
        # Structured output schema from the last run() — applied during retrieve()
        self._last_structured_output: Optional[Type] = None

    @property
    def pending_response_ids(self) -> Dict[str, str]:
        """Background response IDs captured during the last run() call (response_id → custom_id)."""
        return dict(self._background_response_ids)

    def _build_logger_metadata(self, llm_config: LLMConfig, batch_size: int) -> dict:
        """Build the comprehensive logger metadata dict for a run."""
        return {
            # Complete batch-level configs
            'batch_config': {
                'job': asdict(self.job_config),
                'storage': {
                    'output_dir': self.storage_config.output_dir,
                    'path_template': self.storage_config.path_template,
                    's3': asdict(self.resolved_s3_config) if self.resolved_s3_config else None
                },
                'processing': asdict(self.processing_config),
                'column': asdict(self.column_config),
                'llm': llm_config.to_metadata_dict(),
            },

            # Batch execution metadata
            'batch_started_at': datetime.now(timezone.utc).isoformat(),
            'batch_size': batch_size,

            # Custom metadata from job_config (if any)
            **(self.job_config.metadata or {})
        }

    def _print_start(self, action: str, count: int) -> None:
        """Print job-start status lines."""
        print(f"{action} {count} rows...")
        print(f"📁 Local output: {self.local_path}")
        if self.resolved_s3_config:
            print(f"☁️  S3 upload: s3://{self.resolved_s3_config.bucket}/{self.resolved_s3_config.prefix}")

    def _print_end(self, action: str) -> None:
        """Print job-completion status lines."""
        print(f"✅ {action} complete!")
        print(f"📍 Local files: {self.local_path}")
        if self.resolved_s3_config:
            print(f"☁️  S3 location: s3://{self.resolved_s3_config.bucket}/{self.resolved_s3_config.prefix}")

    async def run(
        self,
        df: pd.DataFrame,
        llm_config: LLMConfig,
        response_id_extractor: Optional[Callable[[Any, dict], Optional[str]]] = None,
    ) -> Optional[List]:
        """
        Run batch processing with automatic Parquet logging.

        After _batch_run() completes, any result for which ``response_id_extractor``
        returns a non-None string is treated as a background (queued) response.
        Those response IDs are stored in ``self._background_response_ids`` and
        are automatically used by ``retrieve()``.

        Args:
            df: DataFrame with one row per LLM request.
            llm_config: LLM class and kwargs to use for the batch.
            response_id_extractor: Optional callable ``(result, row_dict) -> str | None``
                that extracts a background response ID from a result object.
                Return None for synchronous (non-background) results.

        Returns:
            List of results if processing_config.return_results=True, None otherwise.
        """
        # Always reset at the start of each run() so stale IDs from a prior call can
        # never leak into retrieve().  Calling run() twice with ChatOpenAI would
        # silently overwrite the first batch's IDs; running a regular (non-background)
        # LLM after a background LLM would leave old IDs in place indefinitely.
        self._background_response_ids = {}
        # Capture the structured output schema so retrieve() can parse responses.
        self._last_structured_output = llm_config.structured_output

        # Suppress Pydantic serialization warnings globally
        warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic")

        if self.column_config.prompt not in df.columns:
            raise ValueError(f"DataFrame missing required column: {self.column_config.prompt}")

        logger_metadata = self._build_logger_metadata(llm_config, len(df))

        if self.processing_config.show_progress:
            self._print_start("🚀 Starting processing of", len(df))

        # If ID extraction will be needed, hold results in memory regardless of
        # return_results — background results are small (just response_id objects)
        # and this prevents a silent failure when return_results=False.
        _will_extract_ids = (
            response_id_extractor is not None or
            llm_config.llm_class.__name__ in _OPENAI_BACKGROUND_CLASSES
        )
        _effective_return_results = self.processing_config.return_results or _will_extract_ids

        with ParquetLogger(
            log_dir=str(self.local_path),
            buffer_size=self.processing_config.buffer_size,
            logger_metadata=logger_metadata,
            partition_on=self.processing_config.partition_on,
            event_types=self.processing_config.event_types,
            s3_config=self.resolved_s3_config,
        ) as logger:
            llm = llm_config.create_llm(callbacks=[logger])
            results = await _batch_run(
                df, llm,
                prompt_col=self.column_config.prompt,
                config_col=self.column_config.config,
                tools_col=self.column_config.tools,
                max_concurrency=self.processing_config.max_concurrency,
                show_progress=self.processing_config.show_progress,
                return_exceptions=self.processing_config.return_exceptions,
                return_results=_effective_return_results,
                row_timeout=self.processing_config.row_timeout,
            )

            # Auto-detect extractor for known background-capable OpenAI LLM classes.
            # ID capture is intentionally inside the 'with' block so that
            # _background_response_ids is populated even when flush() raises an S3 error
            # on context exit — the LLM calls already succeeded, so the IDs are valid.
            if response_id_extractor is None and llm_config.llm_class.__name__ in _OPENAI_BACKGROUND_CLASSES:
                response_id_extractor = _openai_response_id_extractor

            # Warn when structured_output + background LLM are combined — the parser
            # cannot run until retrieval, so batch_results will contain ValueErrors.
            if (llm_config.structured_output
                    and llm_config.llm_class.__name__ in _OPENAI_BACKGROUND_CLASSES):
                warnings.warn(
                    "structured_output is set with a background-capable LLM. "
                    "Structured output parsing requires a completed response and cannot "
                    "run until retrieval. batch_results will contain ValueError exceptions "
                    "(normal for background mode) — retrieve() will apply the schema and "
                    "return a parsed_output column.",
                    UserWarning,
                    stacklevel=2,
                )

            if response_id_extractor and results:
                rows = df.to_dict('records')
                for result, row in zip(results, rows):
                    try:
                        if result is None:
                            continue
                        if isinstance(result, BaseException):
                            # For background mode with structured_output, LangChain's
                            # parser raises ValueError before we can read response_metadata.
                            # The exception message contains the ID and status, so recover
                            # the ID via regex rather than silently dropping it.
                            response_id = _try_extract_id_from_exception(result)
                        else:
                            response_id = response_id_extractor(result, row)
                        if response_id:
                            # Try direct column first; fall back to config tags
                            # (with_tags(custom_id='x') encodes it as 'logger_custom_id:x'
                            # in the tags list — the same extraction logger.py does for llm_end).
                            custom_id = (
                                str(row.get(self.column_config.custom_id) or '')
                                or extract_custom_id(
                                    (row.get(self.column_config.config) or {}).get('tags', [])
                                )
                            )
                            self._background_response_ids[response_id] = custom_id
                    except Exception as exc:
                        warnings.warn(
                            f"response_id_extractor raised {type(exc).__name__} for a row "
                            f"and was skipped: {exc}",
                            stacklevel=2,
                        )

        if self.processing_config.show_progress:
            self._print_end("Processing")

        # Respect the original return_results setting for the caller
        if not self.processing_config.return_results:
            return None
        return results

    async def retrieve(
        self,
        openai_client=None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve background responses and log them to the same Parquet location.

        Uses response IDs captured during ``run()`` if available, otherwise
        auto-discovers pending responses from the local Parquet files (useful
        for cross-process resume when a process restarts after a crash).

        Args:
            openai_client: OpenAI async client. Auto-creates ``AsyncOpenAI()`` if None.
            retrieval_config: Polling and execution settings.

        Returns:
            DataFrame with columns: response_id, status, openai_response, error
            (or None if retrieval_config.return_results=False).
        """
        from .background_retrieval import retrieve_background_responses, _query_pending_responses
        from .storage import LocalStorage, S3Storage

        rc = retrieval_config or self.retrieval_config

        # Build df from in-memory captured IDs, or auto-discover from storage
        if self._background_response_ids:
            df = pd.DataFrame([
                {
                    self.column_config.response_id: rid,
                    self.column_config.custom_id: cid,
                }
                for rid, cid in self._background_response_ids.items()
            ])
            count_desc = f"{len(df)} captured response(s)"
        else:
            # 'memory' source means use only in-memory IDs — skip storage discovery
            if rc.source == 'memory':
                return pd.DataFrame() if rc.return_results else None
            # Auto-discover from Parquet files using this batch's storage
            if rc.source == "s3" and self.resolved_s3_config:
                read_storage = S3Storage(self.resolved_s3_config)
            else:
                read_storage = LocalStorage(self.local_path)

            df = _query_pending_responses(read_storage)
            if df.empty:
                if rc.show_progress:
                    print("No pending responses found in storage.")
                return pd.DataFrame() if rc.return_results else None
            count_desc = f"{len(df)} auto-discovered response(s)"

        if rc.show_progress:
            print(f"🔍 Retrieving {count_desc}...")
            print(f"📁 Local output: {self.local_path}")
            if self.resolved_s3_config:
                print(f"☁️  S3 upload: s3://{self.resolved_s3_config.bucket}/{self.resolved_s3_config.prefix}")

        retrieval_metadata = {
            'batch_config': {
                'job': asdict(self.job_config),
                'storage': {
                    'output_dir': self.storage_config.output_dir,
                    'path_template': self.storage_config.path_template,
                    's3': asdict(self.resolved_s3_config) if self.resolved_s3_config else None,
                },
            },
            'retrieval_started_at': datetime.now(timezone.utc).isoformat(),
            **(self.job_config.metadata or {}),
        }

        # Build a response_parser callable so that parsing happens *during* retrieval
        # and parsed_output is included in the background_retrieval_complete Parquet log.
        # Using a factory function (_make_parser) avoids the closure-captures-variable bug.
        schema = self._last_structured_output
        response_parser = None
        if schema is not None:
            def _make_parser(s):
                def parser(response_dict: dict) -> Optional[dict]:
                    result = _parse_from_openai_response(response_dict, s)
                    return result.model_dump() if result is not None else None
                return parser
            response_parser = _make_parser(schema)

        with ParquetLogger(
            log_dir=str(self.local_path),
            buffer_size=1000,
            logger_metadata=retrieval_metadata,
            partition_on=self.processing_config.partition_on,
            s3_config=self.resolved_s3_config,
        ) as logger:
            results = await retrieve_background_responses(
                df=df,
                openai_client=openai_client,
                logger=logger,
                retrieval_config=rc,
                column_config=self.column_config,
                response_parser=response_parser,
            )

        # Convert dict parsed_output → Pydantic objects for caller convenience.
        # retrieve_background_responses() stores parsed_output as a JSON-serializable dict
        # (for Parquet logging); convert to a Pydantic instance here for the caller.
        if (rc.return_results
                and results is not None
                and not results.empty
                and schema is not None
                and 'parsed_output' in results.columns):
            results = results.copy()
            results['parsed_output'] = results['parsed_output'].apply(
                lambda d: schema.model_validate(d) if isinstance(d, dict) else None
            )

        if rc.show_progress:
            self._print_end("Retrieval")

        return results

    async def run_and_retrieve(
        self,
        df: pd.DataFrame,
        llm_config: LLMConfig,
        openai_client=None,
        retrieval_config: Optional[RetrievalConfig] = None,
        response_id_extractor: Optional[Callable[[Any, dict], Optional[str]]] = None,
    ) -> BatchOutcome:
        """
        Run batch processing then immediately retrieve any background responses.

        This is the primary method for the background-processing workflow.
        After the batch completes, background response IDs captured via
        ``response_id_extractor`` are polled automatically.

        Args:
            df: DataFrame with one row per LLM request.
            llm_config: LLM class and kwargs to use for the batch.
            openai_client: OpenAI async client for retrieval. Auto-created if None.
            retrieval_config: Polling and execution settings for retrieval.
            response_id_extractor: Callable ``(result, row_dict) -> str | None``
                that extracts a background response ID from a result object.

        Returns:
            BatchOutcome with:
                - ``batch_results``: list of LLM results (or None)
                - ``retrieval_results``: DataFrame of retrieved responses (or None)
        """
        batch_results = await self.run(
            df, llm_config, response_id_extractor=response_id_extractor
        )
        # If run() captured no background IDs (e.g. synchronous LLM), skip retrieve
        if not self._background_response_ids:
            if self.processing_config.show_progress:
                print(
                    "⚠️  No background response IDs captured — retrieval skipped.\n"
                    "   If you expected background responses, verify:\n"
                    "   - The model is configured for background mode (e.g. service_tier='flex')\n"
                    "   - ProcessingConfig(return_exceptions=True) so pending results are returned"
                )
            return BatchOutcome(batch_results=batch_results, retrieval_results=None)
        retrieval_results = await self.retrieve(
            openai_client=openai_client,
            retrieval_config=retrieval_config or self.retrieval_config,
        )
        return BatchOutcome(
            batch_results=batch_results,
            retrieval_results=retrieval_results,
        )
