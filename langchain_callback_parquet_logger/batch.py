"""Batch processing utilities for DataFrame operations with LangChain."""

import asyncio
import os
import warnings
from pathlib import Path
from datetime import date, datetime, timezone
from dataclasses import asdict
from typing import Any, Optional, TYPE_CHECKING, Type, List

import pandas as pd
from langchain_core.runnables import RunnableLambda

from .logger import ParquetLogger
from .config import (
    JobConfig, StorageConfig, ProcessingConfig, ColumnConfig,
    S3Config, EventType, LLMConfig, RetrievalConfig
)
from .tagging import with_tags


async def batch_run(
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
    Low-level async batch processing for DataFrames.

    **Most users should use batch_process() instead**, which includes automatic logging.
    Use batch_run() only when you need direct control over logging.

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

    Example:
        >>> # When you already have logging configured:
        >>> with ParquetLogger('./logs') as logger:
        >>>     llm.callbacks = [logger]
        >>>     results = await batch_run(df, llm, max_concurrency=100)
        >>>
        >>> # For most users, use batch_process() instead:
        >>> results = await batch_process(df)  # Logging handled automatically
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
            completed_count += 1
            if show_progress and (completed_count % progress_interval == 0 or completed_count == total_count):
                print(f"  Processed {completed_count}/{total_count} rows ({100 * completed_count // total_count}%)")
            return result
        except asyncio.TimeoutError:
            completed_count += 1
            if show_progress and (completed_count % progress_interval == 0 or completed_count == total_count):
                print(f"  Processed {completed_count}/{total_count} rows ({100 * completed_count // total_count}%)")
            if return_exceptions:
                return TimeoutError(f"Row timed out after {row_timeout}s")
            raise TimeoutError(f"Row timed out after {row_timeout}s")
        except Exception as e:
            completed_count += 1
            if show_progress and (completed_count % progress_interval == 0 or completed_count == total_count):
                print(f"  Processed {completed_count}/{total_count} rows ({100 * completed_count // total_count}%)")
            if return_exceptions:
                return e
            raise

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

    Shared by batch_process() and retrieve_batch_responses() so both functions
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

    # Check for S3 bucket in environment if not configured
    if not storage_config.s3_config and os.environ.get('LANGCHAIN_S3_BUCKET'):
        storage_config.s3_config = S3Config(bucket=os.environ['LANGCHAIN_S3_BUCKET'])

    resolved_s3_config = None
    if storage_config.s3_config:
        import copy
        resolved_s3_config = copy.copy(storage_config.s3_config)
        base_prefix = resolved_s3_config.prefix.rstrip('/')
        formatted_path = storage_config.path_template.format(**template_vars).lstrip('/')
        resolved_s3_config.prefix = f"{base_prefix}/{formatted_path}/" if base_prefix else f"{formatted_path}/"

    return local_path, resolved_s3_config


async def batch_process(
    df: pd.DataFrame,
    llm_config: LLMConfig,
    job_config: Optional[JobConfig] = None,
    storage_config: Optional[StorageConfig] = None,
    processing_config: Optional[ProcessingConfig] = None,
    column_config: Optional[ColumnConfig] = None,
) -> Optional[List]:
    """
    Batch process DataFrame through LLM with automatic logging to Parquet.

    Args:
        df: DataFrame with prepared data
        llm_config: LLM configuration including class, kwargs, and structured output
        job_config: Job metadata configuration
        storage_config: Storage configuration for output files
        processing_config: Processing configuration for batch operations
        column_config: DataFrame column name configuration

    Returns:
        List of results if processing_config.return_results=True, None otherwise

    Examples:
        >>> # Simple usage with LLMConfig
        >>> from langchain_openai import ChatOpenAI
        >>> await batch_process(
        ...     df,
        ...     llm_config=LLMConfig(
        ...         llm_class=ChatOpenAI,
        ...         llm_kwargs={'model': 'gpt-4', 'temperature': 0.7}
        ...     ),
        ...     job_config=JobConfig(category='analysis', version='2.0')
        ... )

        >>> # With structured output
        >>> from pydantic import BaseModel
        >>> class EmailInfo(BaseModel):
        ...     email: str
        ...     valid: bool
        >>>
        >>> await batch_process(
        ...     df,
        ...     llm_config=LLMConfig(
        ...         llm_class=ChatOpenAI,
        ...         llm_kwargs={'model': 'gpt-4'},
        ...         structured_output=EmailInfo
        ...     ),
        ...     job_config=JobConfig(category='email_validation')
        ... )
    """
    # Suppress Pydantic serialization warnings globally
    warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic")

    # Initialize configs with defaults if not provided
    job_config = job_config or JobConfig()
    storage_config = storage_config or StorageConfig()
    processing_config = processing_config or ProcessingConfig()
    column_config = column_config or ColumnConfig()

    # Validate DataFrame has required columns
    if column_config.prompt not in df.columns:
        raise ValueError(f"DataFrame missing required column: {column_config.prompt}")

    # LLM will be created inside the logger context with callbacks

    # Resolve local and S3 paths
    local_path, resolved_s3_config = _build_storage_paths(job_config, storage_config)
    local_path.mkdir(parents=True, exist_ok=True)

    # Build comprehensive logger metadata
    logger_metadata = {
        # Legacy flat fields (for backward compatibility in queries)
        'job_category': job_config.category,
        'job_subcategory': job_config.subcategory,
        'environment': job_config.environment,
        'job_description': job_config.description,
        'job_version': job_config.version,

        # Complete batch-level configs (NEW structure)
        'batch_config': {
            'job': asdict(job_config) if job_config else None,
            'storage': {
                'output_dir': storage_config.output_dir,
                'path_template': storage_config.path_template,
                's3': asdict(resolved_s3_config) if resolved_s3_config else None
            },
            'processing': asdict(processing_config) if processing_config else None,
            'column': asdict(column_config) if column_config else None,
            'llm': llm_config.to_metadata_dict(),
        },

        # Batch execution metadata
        'batch_started_at': datetime.now(timezone.utc).isoformat(),
        'batch_size': len(df),

        # Custom metadata from job_config (if any)
        **(job_config.metadata or {})
    }

    # Print status messages
    if processing_config.show_progress:
        print(f"🚀 Starting processing of {len(df)} rows...")
        print(f"📁 Local output: {local_path}")
        if resolved_s3_config:
            print(f"☁️  S3 upload: s3://{resolved_s3_config.bucket}/{resolved_s3_config.prefix}")

    # Process with context manager for automatic cleanup
    with ParquetLogger(
        log_dir=str(local_path),
        buffer_size=processing_config.buffer_size,
        logger_metadata=logger_metadata,
        partition_on=processing_config.partition_on,
        event_types=processing_config.event_types,
        s3_config=resolved_s3_config,
    ) as logger:
        # Create LLM with logger as callback
        llm = llm_config.create_llm(callbacks=[logger])

        # Run batch processing
        results = await batch_run(
            df, llm,
            prompt_col=column_config.prompt,
            config_col=column_config.config,
            tools_col=column_config.tools,
            max_concurrency=processing_config.max_concurrency,
            show_progress=processing_config.show_progress,
            return_exceptions=processing_config.return_exceptions,
            return_results=processing_config.return_results,
            row_timeout=processing_config.row_timeout,
        )

    # Print completion message
    if processing_config.show_progress:
        print("✅ Processing complete!")
        print(f"📍 Local files: {local_path}")
        if resolved_s3_config:
            print(f"☁️  S3 location: s3://{resolved_s3_config.bucket}/{resolved_s3_config.prefix}")

    return results


async def retrieve_batch_responses(
    df: pd.DataFrame,
    openai_client=None,
    job_config: Optional[JobConfig] = None,
    storage_config: Optional[StorageConfig] = None,
    retrieval_config: Optional[RetrievalConfig] = None,
    response_id_col: str = "response_id",
    custom_id_col: str = "custom_id",
) -> Optional[pd.DataFrame]:
    """
    Retrieve OpenAI background responses and log them to the same Parquet location
    as the original batch_process() call.

    Polls each response until its status is "completed" (or a terminal error/cancellation),
    logging each pending check and the final outcome. Pass the same ``job_config`` and
    ``storage_config`` you used in ``batch_process()`` to have all events land in one place.

    Args:
        df: DataFrame with response IDs (and optionally custom IDs) to retrieve
        openai_client: OpenAI async client. If None, creates ``AsyncOpenAI()`` automatically
            using ``OPENAI_API_KEY`` from the environment (same behaviour as LangChain's ChatOpenAI).
        job_config: Job metadata — use the same values as the original batch_process() call
        storage_config: Storage paths — use the same values as the original batch_process() call
        retrieval_config: Polling and execution settings (poll_interval, max_poll_attempts, etc.)
        response_id_col: Column name containing OpenAI response IDs (default: "response_id")
        custom_id_col: Column name containing your custom row IDs (default: "custom_id")

    Returns:
        DataFrame with columns: response_id, status, openai_response, error
        (or None if retrieval_config.return_results=False)

    Example:
        >>> from openai import AsyncOpenAI
        >>> import langchain_callback_parquet_logger as lcpl
        >>>
        >>> results = await lcpl.retrieve_batch_responses(
        ...     df,  # DataFrame with response_id and custom_id columns
        ...     openai_client=AsyncOpenAI(),
        ...     job_config=lcpl.JobConfig(
        ...         category=job_category,
        ...         subcategory=job_subcategory,
        ...         version=job_version,
        ...         environment="hex_notebook",
        ...     ),
        ...     storage_config=lcpl.StorageConfig(
        ...         output_dir="./batch_logs",
        ...         path_template="{job_category}/{job_subcategory}/v{job_version_safe}",
        ...         s3_config=lcpl.S3Config(bucket="my-bucket", on_failure="error"),
        ...     ),
        ...     retrieval_config=lcpl.RetrievalConfig(
        ...         poll_interval=30.0,
        ...         max_poll_attempts=40,
        ...         checkpoint_file="./retrieval_checkpoint.parquet",
        ...     ),
        ... )
    """
    from .background_retrieval import retrieve_background_responses

    job_config = job_config or JobConfig()
    storage_config = storage_config or StorageConfig()
    retrieval_config = retrieval_config or RetrievalConfig()

    local_path, resolved_s3_config = _build_storage_paths(job_config, storage_config)
    local_path.mkdir(parents=True, exist_ok=True)

    if retrieval_config.show_progress:
        print(f"🔍 Retrieving {len(df)} responses...")
        print(f"📁 Local output: {local_path}")
        if resolved_s3_config:
            print(f"☁️  S3 upload: s3://{resolved_s3_config.bucket}/{resolved_s3_config.prefix}")

    with ParquetLogger(
        log_dir=str(local_path),
        buffer_size=1000,
        partition_on="date",
        s3_config=resolved_s3_config,
    ) as logger:
        results = await retrieve_background_responses(
            df=df,
            openai_client=openai_client,
            logger=logger,
            response_id_col=response_id_col,
            custom_id_col=custom_id_col,
            batch_size=retrieval_config.batch_size,
            max_retries=retrieval_config.max_retries,
            timeout=retrieval_config.timeout,
            poll_interval=retrieval_config.poll_interval,
            max_poll_attempts=retrieval_config.max_poll_attempts,
            show_progress=retrieval_config.show_progress,
            checkpoint_file=retrieval_config.checkpoint_file,
            return_results=retrieval_config.return_results,
        )

    if retrieval_config.show_progress:
        print("✅ Retrieval complete!")
        print(f"📍 Local files: {local_path}")
        if resolved_s3_config:
            print(f"☁️  S3 location: s3://{resolved_s3_config.bucket}/{resolved_s3_config.prefix}")

    return results
