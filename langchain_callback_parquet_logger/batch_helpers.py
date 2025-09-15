"""
Minimal batch processing helper for DataFrame operations with LangChain.
Focuses on simplicity - users prepare data in DataFrame columns.
"""

import asyncio
import os
from pathlib import Path
from datetime import date
from typing import Any, Optional, TYPE_CHECKING, Type, Dict, List, Literal

if TYPE_CHECKING:
    import pandas as pd

from langchain_core.runnables import RunnableLambda
from .logger import ParquetLogger
from . import with_tags


async def batch_run(
    df: "pd.DataFrame",
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
    Minimal batch runner for DataFrames with LLMs.
    
    Users prepare all data in DataFrame columns, this just handles async batching.
    
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
        >>> # Normal usage - keep results
        >>> results = await batch_run(df, llm, max_concurrency=100)
        >>> df['result'] = results
        >>> 
        >>> # Memory-efficient for huge DataFrames (results only in ParquetLogger)
        >>> with ParquetLogger('./logs') as logger:
        >>>     llm.callbacks = [logger]
        >>>     await batch_run(huge_df, llm, return_results=False)
        >>>     # Results are in parquet files, not memory
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
        # Process all rows but discard results (they're in ParquetLogger)
        await runner.abatch(
            rows,
            config={"max_concurrency": max_concurrency},
            return_exceptions=return_exceptions
        )
        
        if progress_bar:
            progress_bar.close()
        
        return None


async def batch_process(
    df: "pd.DataFrame",
    llm: Any = None,
    llm_class: Optional[Type] = None,
    structured_output: Optional[Type] = None,
    
    # Job metadata
    job_category: str = "batch_processing",
    job_subcategory: str = "default",
    job_description: str = "",
    job_version: str = "1.0.0",
    environment: str = "production",
    
    # Storage configuration - LOCAL OR S3
    output_dir: str = "./batch_logs",
    output_path_template: str = "{job_category}/{job_subcategory}",
    
    # S3 configuration (optional - only used if s3_bucket provided)
    s3_bucket: Optional[str] = None,
    s3_prefix_template: Optional[str] = None,
    s3_on_failure: Optional[Literal["error", "continue"]] = None,
    s3_retry_attempts: int = 3,
    
    # Logger configuration - ALL PARAMETERS EXPOSED
    provider: Optional[str] = None,
    buffer_size: int = 1000,
    partition_on: Optional[Literal["date"]] = "date",
    event_types: Optional[List[str]] = None,
    
    # LLM configuration
    llm_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    
    # Processing configuration - ALL PARAMETERS EXPOSED
    max_concurrency: int = 100,
    show_progress: bool = True,
    return_exceptions: bool = True,
    return_results: bool = False,
    
    # Column names
    prompt_col: str = "prompt",
    config_col: str = "config",
    tools_col: Optional[str] = "tools",
    
    # Additional customization
    extra_metadata: Optional[Dict[str, Any]] = None,
    logger_kwargs_override: Optional[Dict[str, Any]] = None,
    batch_kwargs_override: Optional[Dict[str, Any]] = None,
) -> Optional[List]:
    """
    Batch process DataFrame through LLM with automatic logging to Parquet.
    
    Supports three storage modes:
    1. Local only (default): Write Parquet files to local directory
    2. Local + S3: Write locally then upload to S3
    3. Memory efficient: Don't keep results in memory (return_results=False)
    
    Args:
        df: DataFrame with prepared data
        llm: Pre-configured LLM instance (if None, will create one)
        llm_class: LLM class to instantiate (default: ChatOpenAI if available)
        structured_output: Pydantic model for structured output
        
        job_category: Category for organizing logs (default: "batch_processing")
        job_subcategory: Subcategory for organizing logs (default: "default")
        job_description: Human-readable description of the job
        job_version: Version string for the job (default: "1.0.0")
        environment: Environment name (default: "production")
        
        output_dir: Base directory for local Parquet files (default: "./batch_logs")
        output_path_template: Template for organizing files locally
        
        s3_bucket: S3 bucket name (None = local only, string = also upload to S3)
        s3_prefix_template: S3 path template (defaults to output_path_template)
        s3_on_failure: How to handle S3 failures - "error" or "continue"
        s3_retry_attempts: Number of S3 upload retry attempts (default: 3)
        
        provider: LLM provider name (auto-detected if None)
        buffer_size: Number of entries before flushing to disk (default: 1000)
        partition_on: Partitioning strategy - "date" or None (default: "date")
        event_types: Event types to log (default: LLM events only)
        
        llm_kwargs: Keyword arguments for LLM initialization
        model_kwargs: Model-specific keyword arguments
        
        max_concurrency: Maximum concurrent LLM requests (default: 100)
        show_progress: Show progress bar (default: True)
        return_exceptions: Return exceptions instead of raising (default: True)
        return_results: Keep results in memory (default: False for efficiency)
        
        prompt_col: DataFrame column containing prompts (default: "prompt")
        config_col: DataFrame column containing configs (default: "config")
        tools_col: DataFrame column containing tools (default: "tools")
        
        extra_metadata: Additional metadata to include in logs
        logger_kwargs_override: Override any ParquetLogger parameters
        batch_kwargs_override: Override any batch_run parameters
        
    Returns:
        List of results if return_results=True, None otherwise
        
    Storage Configuration:
        Local storage is always used (even with S3 enabled).
        S3 upload is optional and happens after local write.
        
        output_dir: Base directory for local Parquet files
        output_path_template: Template for organizing files locally
        s3_bucket: If provided, also upload to this S3 bucket
        s3_prefix_template: S3 organization (defaults to output_path_template)
    
    Path Template Variables:
        {job_category}: From job_category parameter
        {job_subcategory}: From job_subcategory parameter
        {environment}: From environment parameter
        {job_version}: From job_version parameter
        {date}: Current date (YYYY-MM-DD)
    
    Examples:
        # Local only processing
        >>> await batch_process(df, job_category="validation")
        
        # Local with organized structure
        >>> await batch_process(
        ...     df,
        ...     job_category="emails",
        ...     output_dir="/data/batches",
        ...     output_path_template="{job_category}/v{job_version}"
        ... )
        
        # With S3 upload
        >>> await batch_process(
        ...     df,
        ...     job_category="production",
        ...     s3_bucket="my-bucket",
        ...     s3_prefix_template="ml/{job_category}/{date}/"
        ... )
        
        # With structured output
        >>> from pydantic import BaseModel
        >>> class EmailInfo(BaseModel):
        ...     email: str
        ...     valid: bool
        >>> 
        >>> await batch_process(
        ...     df,
        ...     structured_output=EmailInfo,
        ...     job_category="email_validation",
        ...     max_concurrency=1000
        ... )
        
        # Advanced with overrides
        >>> await batch_process(
        ...     df,
        ...     job_category="analysis",
        ...     event_types=['llm_start', 'llm_end', 'chain_start'],
        ...     logger_kwargs_override={'custom_param': 'value'}
        ... )
    """
    # Validate DataFrame has required columns
    missing_cols = []
    if prompt_col not in df.columns:
        missing_cols.append(prompt_col)
    # Only check for config_col if it's not None and not in columns
    if config_col and config_col not in df.columns:
        # config_col is optional, only error if explicitly specified but missing
        pass  # Don't add to missing_cols since it's optional
    # Only check for tools_col if it's explicitly specified and not in columns
    if tools_col and tools_col not in df.columns:
        # tools_col is optional, only error if explicitly specified but missing
        pass  # Don't add to missing_cols since it's optional
    
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
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
        
        # Initialize LLM with provided kwargs
        llm_init_kwargs = llm_kwargs or {}
        if model_kwargs:
            llm_init_kwargs['model_kwargs'] = model_kwargs
        
        llm = llm_class(**llm_init_kwargs)
    
    # Apply structured output if provided
    if structured_output:
        llm = llm.with_structured_output(structured_output)
    
    # Auto-detect provider if not specified
    if provider is None:
        # Try to detect from LLM class
        llm_class_name = llm.__class__.__name__
        provider_map = {
            'ChatOpenAI': 'openai',
            'ChatAnthropic': 'anthropic',
            'ChatCohere': 'cohere',
            'ChatGoogleGenerativeAI': 'google',
            'ChatVertexAI': 'google',
            'AzureChatOpenAI': 'azure',
        }
        provider = provider_map.get(llm_class_name, 'unknown')
    
    # Format path templates with job metadata
    template_vars = {
        'job_category': job_category,
        'job_subcategory': job_subcategory,
        'environment': environment,
        'job_version': job_version,
        'date': date.today().isoformat(),
    }
    
    # Build local output path
    local_path = Path(output_dir) / output_path_template.format(**template_vars)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Determine if S3 is being used
    use_s3 = s3_bucket is not None
    
    # Check for S3 bucket in environment if not provided
    if not use_s3 and os.environ.get('LANGCHAIN_S3_BUCKET'):
        s3_bucket = os.environ['LANGCHAIN_S3_BUCKET']
        use_s3 = True
    
    # Build logger configuration
    logger_config = {
        'log_dir': str(local_path),
        'buffer_size': buffer_size,
        'provider': provider,
        'partition_on': partition_on,
        'logger_metadata': {
            'environment': environment,
            'job_category': job_category,
            'job_subcategory': job_subcategory,
            'job_description': job_description,
            'job_version': job_version,
            'llm_kwargs': llm_kwargs or {},
            'model_kwargs': model_kwargs or {},
            **(extra_metadata or {})
        }
    }
    
    # Add event_types if specified
    if event_types is not None:
        logger_config['event_types'] = event_types
    
    # Add S3 configuration if using S3
    if use_s3:
        # Format S3 prefix
        s3_prefix_str = s3_prefix_template or output_path_template
        s3_prefix = s3_prefix_str.format(**template_vars)
        
        # Ensure prefix ends with /
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'
        
        # Auto-determine s3_on_failure if not specified
        if s3_on_failure is None:
            s3_on_failure = 'error' if environment in ['production', 'hex_notebook'] else 'continue'
        
        logger_config.update({
            's3_bucket': s3_bucket,
            's3_prefix': s3_prefix,
            's3_on_failure': s3_on_failure,
            's3_retry_attempts': s3_retry_attempts,
        })
    
    # Apply logger overrides
    if logger_kwargs_override:
        logger_config.update(logger_kwargs_override)
    
    # Build batch_run configuration
    batch_config = {
        'prompt_col': prompt_col,
        'config_col': config_col,
        'tools_col': tools_col,
        'max_concurrency': max_concurrency,
        'show_progress': show_progress,
        'return_exceptions': return_exceptions,
        'return_results': return_results,
    }
    
    # Apply batch overrides
    if batch_kwargs_override:
        batch_config.update(batch_kwargs_override)
    
    # Print status messages
    if show_progress:
        print(f"🚀 Starting processing of {len(df)} rows...")
        print(f"📁 Local output: {local_path}")
        if use_s3:
            print(f"☁️  S3 upload: s3://{s3_bucket}/{s3_prefix}")
    
    # Process with context manager for automatic cleanup
    with ParquetLogger(**logger_config) as logger:
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
        results = await batch_run(df, llm, **batch_config)
    
    # Print completion message
    if show_progress:
        print("✅ Processing complete!")
        print(f"📍 Local files: {local_path}")
        if use_s3:
            print(f"☁️  S3 location: s3://{s3_bucket}/{s3_prefix}")
    
    return results