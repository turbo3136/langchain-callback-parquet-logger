"""Batch processing utilities for DataFrame operations with LangChain."""

import os
from pathlib import Path
from datetime import date
from typing import Any, Optional, TYPE_CHECKING, Type, List

import pandas as pd
from langchain_core.runnables import RunnableLambda

from .logger import ParquetLogger
from .config import (
    JobConfig, StorageConfig, ProcessingConfig, ColumnConfig,
    S3Config, EventType
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
    rows = df.to_dict('records')

    # Setup progress tracking if requested
    progress_bar = None
    if show_progress:
        try:
            # Try to use tqdm (auto-detects notebook vs terminal)
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    from tqdm.notebook import tqdm
                else:
                    from tqdm import tqdm
            except ImportError:
                from tqdm import tqdm
            progress_bar = tqdm(total=len(rows), desc="Processing batch")
        except ImportError:
            # Fall back to simple counter
            progress_bar = None
            print(f"Processing {len(rows)} rows...")

    async def process_row(row: dict) -> Any:
        """Process a single row from the DataFrame."""
        # Build invoke kwargs from DataFrame columns
        invoke_kwargs = {"input": row.get(prompt_col)}

        # Add config if column exists
        if config_col in row and row[config_col] is not None:
            invoke_kwargs["config"] = row[config_col]

        # Add tools if column exists and not None
        if tools_col and tools_col in row and row[tools_col] is not None:
            invoke_kwargs["tools"] = row[tools_col]

        try:
            result = await llm.ainvoke(**invoke_kwargs)
            if progress_bar:
                progress_bar.update(1)
            return result
        except Exception as e:
            if progress_bar:
                progress_bar.update(1)
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

        if progress_bar:
            progress_bar.close()

        return results
    else:
        # Memory-efficient mode: don't collect results
        await runner.abatch(
            rows,
            config={"max_concurrency": max_concurrency},
            return_exceptions=return_exceptions
        )

        if progress_bar:
            progress_bar.close()

        return None


async def batch_process(
    df: pd.DataFrame,
    llm: Any = None,
    job_config: Optional[JobConfig] = None,
    storage_config: Optional[StorageConfig] = None,
    processing_config: Optional[ProcessingConfig] = None,
    column_config: Optional[ColumnConfig] = None,
    llm_class: Optional[Type] = None,
    structured_output: Optional[Type] = None,
    llm_kwargs: Optional[dict] = None,
) -> Optional[List]:
    """
    Batch process DataFrame through LLM with automatic logging to Parquet.

    Args:
        df: DataFrame with prepared data
        llm: Pre-configured LLM instance (if None, will create one)
        job_config: Job metadata configuration
        storage_config: Storage configuration for output files
        processing_config: Processing configuration for batch operations
        column_config: DataFrame column name configuration
        llm_class: LLM class to instantiate if llm is None
        structured_output: Optional Pydantic model for structured output
        llm_kwargs: Keyword arguments for LLM initialization

    Returns:
        List of results if processing_config.return_results=True, None otherwise

    Examples:
        >>> # Simple usage with configs
        >>> await batch_process(
        ...     df,
        ...     job_config=JobConfig(category='analysis', version='2.0'),
        ...     storage_config=StorageConfig(s3_config=S3Config(bucket='my-bucket'))
        ... )

        >>> # With structured output
        >>> from pydantic import BaseModel
        >>> class EmailInfo(BaseModel):
        ...     email: str
        ...     valid: bool
        >>>
        >>> await batch_process(
        ...     df,
        ...     structured_output=EmailInfo,
        ...     job_config=JobConfig(category='email_validation')
        ... )
    """
    # Initialize configs with defaults if not provided
    job_config = job_config or JobConfig()
    storage_config = storage_config or StorageConfig()
    processing_config = processing_config or ProcessingConfig()
    column_config = column_config or ColumnConfig()

    # Validate DataFrame has required columns
    if column_config.prompt not in df.columns:
        raise ValueError(f"DataFrame missing required column: {column_config.prompt}")

    # Handle LLM creation
    if llm is None:
        if llm_class is None:
            # Try to use ChatOpenAI as default
            try:
                from langchain_openai import ChatOpenAI
                llm_class = ChatOpenAI
            except ImportError:
                raise ImportError(
                    "No LLM provided and ChatOpenAI not available. "
                    "Either provide 'llm' parameter or install langchain-openai"
                )

        # Initialize LLM
        llm = llm_class(**(llm_kwargs or {}))

    # Apply structured output if provided
    if structured_output:
        llm = llm.with_structured_output(structured_output)

    # Auto-detect provider
    provider = _detect_llm_provider(llm)

    # Format path templates with job metadata
    template_vars = {
        'job_category': job_config.category,
        'job_subcategory': job_config.subcategory,
        'environment': job_config.environment,
        'job_version': job_config.version,
        'date': date.today().isoformat(),
    }

    # Build local output path
    local_path = Path(storage_config.output_dir) / storage_config.path_template.format(**template_vars)
    local_path.mkdir(parents=True, exist_ok=True)

    # Check for S3 bucket in environment if not configured
    if not storage_config.s3_config and os.environ.get('LANGCHAIN_S3_BUCKET'):
        storage_config.s3_config = S3Config(bucket=os.environ['LANGCHAIN_S3_BUCKET'])

    # Format S3 prefix if using S3
    if storage_config.s3_config:
        # Format S3 prefix with template vars
        s3_prefix = storage_config.path_template.format(**template_vars)
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'
        storage_config.s3_config.prefix = s3_prefix

    # Build logger configuration
    logger_metadata = {
        'environment': job_config.environment,
        'job_category': job_config.category,
        'job_subcategory': job_config.subcategory,
        'job_description': job_config.description,
        'job_version': job_config.version,
        'provider': provider,
        **job_config.metadata
    }

    # Print status messages
    if processing_config.show_progress:
        print(f"🚀 Starting processing of {len(df)} rows...")
        print(f"📁 Local output: {local_path}")
        if storage_config.s3_config:
            print(f"☁️  S3 upload: s3://{storage_config.s3_config.bucket}/{storage_config.s3_config.prefix}")

    # Process with context manager for automatic cleanup
    with ParquetLogger(
        log_dir=str(local_path),
        buffer_size=processing_config.buffer_size,
        logger_metadata=logger_metadata,
        partition_on=processing_config.partition_on,
        event_types=processing_config.event_types,
        s3_config=storage_config.s3_config
    ) as logger:
        # Add logger to LLM callbacks
        if hasattr(llm, 'callbacks'):
            if llm.callbacks:
                llm.callbacks.append(logger)
            else:
                llm.callbacks = [logger]
        else:
            # Try to set callbacks attribute
            try:
                llm.callbacks = [logger]
            except AttributeError:
                raise ValueError(
                    "Cannot add callbacks to LLM. "
                    "Please provide an LLM that supports callbacks."
                )

        # Run batch processing
        results = await batch_run(
            df, llm,
            prompt_col=column_config.prompt,
            config_col=column_config.config,
            tools_col=column_config.tools,
            max_concurrency=processing_config.max_concurrency,
            show_progress=processing_config.show_progress,
            return_exceptions=processing_config.return_exceptions,
            return_results=processing_config.return_results
        )

    # Print completion message
    if processing_config.show_progress:
        print("✅ Processing complete!")
        print(f"📍 Local files: {local_path}")
        if storage_config.s3_config:
            print(f"☁️  S3 location: s3://{storage_config.s3_config.bucket}/{storage_config.s3_config.prefix}")

    return results


def _detect_llm_provider(llm: Any) -> str:
    """Auto-detect LLM provider from instance."""
    llm_class_name = llm.__class__.__name__
    provider_map = {
        'ChatOpenAI': 'openai',
        'ChatAnthropic': 'anthropic',
        'ChatCohere': 'cohere',
        'ChatGoogleGenerativeAI': 'google',
        'ChatVertexAI': 'google',
        'AzureChatOpenAI': 'azure',
    }
    return provider_map.get(llm_class_name, 'unknown')