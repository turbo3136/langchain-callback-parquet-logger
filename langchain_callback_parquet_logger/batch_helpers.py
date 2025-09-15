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
    job_config: Optional[Dict[str, Any]] = None,
    storage_config: Optional[Dict[str, Any]] = None,
    processing_config: Optional[Dict[str, Any]] = None,
    column_config: Optional[Dict[str, str]] = None,
    **kwargs  # For backward compatibility and advanced overrides
) -> Optional[List]:
    """
    Batch process DataFrame through LLM with automatic logging to Parquet.

    Args:
        df: DataFrame with prepared data
        llm: Pre-configured LLM instance (if None, will create one)
        job_config: Job metadata configuration dict with keys:
            - category: Category for organizing logs (default: "batch_processing")
            - subcategory: Subcategory (default: "default")
            - description: Human-readable description
            - version: Version string (default: "1.0.0")
            - environment: Environment name (default: "production")
            - metadata: Additional metadata dict
        storage_config: Storage configuration dict with keys:
            - output_dir: Base directory for local files (default: "./batch_logs")
            - path_template: Template for organizing files
            - s3_bucket: S3 bucket name (optional)
            - s3_prefix: S3 prefix template (optional)
            - s3_on_failure: "error" or "continue" (default: auto)
            - s3_retries: Number of retry attempts (default: 3)
        processing_config: Processing configuration dict with keys:
            - max_concurrency: Maximum concurrent requests (default: 100)
            - buffer_size: Entries before flushing (default: 1000)
            - show_progress: Show progress bar (default: True)
            - return_exceptions: Return exceptions (default: True)
            - return_results: Keep results in memory (default: False)
            - event_types: Event types to log (optional)
            - partition_on: "date" or None (default: "date")
            - provider: LLM provider name (auto-detected if None)
        column_config: Column name configuration dict with keys:
            - prompt: Column containing prompts (default: "prompt")
            - config: Column containing configs (default: "config")
            - tools: Column containing tools (default: "tools")
        **kwargs: Additional parameters for backward compatibility:
            - llm_class, structured_output, llm_kwargs, model_kwargs
            - Direct parameters like job_category, output_dir, etc.
            - logger_kwargs_override, batch_kwargs_override

    Returns:
        List of results if return_results=True, None otherwise

    Examples:
        # Simple usage with configs
        >>> await batch_process(
        ...     df,
        ...     job_config={'category': 'analysis', 'version': '2.0'},
        ...     storage_config={'s3_bucket': 'my-bucket'}
        ... )

        # Backward compatible usage
        >>> await batch_process(
        ...     df,
        ...     job_category="validation",
        ...     s3_bucket="my-bucket",
        ...     max_concurrency=1000
        ... )
    """
    # Handle backward compatibility - merge kwargs into configs
    job_config = job_config or {}
    storage_config = storage_config or {}
    processing_config = processing_config or {}
    column_config = column_config or {}

    # Map old parameter names to new config structure
    if 'job_category' in kwargs:
        job_config.setdefault('category', kwargs.pop('job_category'))
    if 'job_subcategory' in kwargs:
        job_config.setdefault('subcategory', kwargs.pop('job_subcategory'))
    if 'job_description' in kwargs:
        job_config.setdefault('description', kwargs.pop('job_description'))
    if 'job_version' in kwargs:
        job_config.setdefault('version', kwargs.pop('job_version'))
    if 'environment' in kwargs:
        job_config.setdefault('environment', kwargs.pop('environment'))
    if 'extra_metadata' in kwargs:
        job_config.setdefault('metadata', kwargs.pop('extra_metadata'))

    if 'output_dir' in kwargs:
        storage_config.setdefault('output_dir', kwargs.pop('output_dir'))
    if 'output_path_template' in kwargs:
        storage_config.setdefault('path_template', kwargs.pop('output_path_template'))
    if 's3_bucket' in kwargs:
        storage_config.setdefault('s3_bucket', kwargs.pop('s3_bucket'))
    if 's3_prefix_template' in kwargs:
        storage_config.setdefault('s3_prefix', kwargs.pop('s3_prefix_template'))
    if 's3_on_failure' in kwargs:
        storage_config.setdefault('s3_on_failure', kwargs.pop('s3_on_failure'))
    if 's3_retry_attempts' in kwargs:
        storage_config.setdefault('s3_retries', kwargs.pop('s3_retry_attempts'))

    if 'max_concurrency' in kwargs:
        processing_config.setdefault('max_concurrency', kwargs.pop('max_concurrency'))
    if 'buffer_size' in kwargs:
        processing_config.setdefault('buffer_size', kwargs.pop('buffer_size'))
    if 'show_progress' in kwargs:
        processing_config.setdefault('show_progress', kwargs.pop('show_progress'))
    if 'return_exceptions' in kwargs:
        processing_config.setdefault('return_exceptions', kwargs.pop('return_exceptions'))
    if 'return_results' in kwargs:
        processing_config.setdefault('return_results', kwargs.pop('return_results'))
    if 'event_types' in kwargs:
        processing_config.setdefault('event_types', kwargs.pop('event_types'))
    if 'partition_on' in kwargs:
        processing_config.setdefault('partition_on', kwargs.pop('partition_on'))
    if 'provider' in kwargs:
        processing_config.setdefault('provider', kwargs.pop('provider'))

    if 'prompt_col' in kwargs:
        column_config.setdefault('prompt', kwargs.pop('prompt_col'))
    if 'config_col' in kwargs:
        column_config.setdefault('config', kwargs.pop('config_col'))
    if 'tools_col' in kwargs:
        column_config.setdefault('tools', kwargs.pop('tools_col'))

    # Extract remaining kwargs for LLM and overrides
    llm_class = kwargs.pop('llm_class', None)
    structured_output = kwargs.pop('structured_output', None)
    llm_kwargs = kwargs.pop('llm_kwargs', None)
    model_kwargs = kwargs.pop('model_kwargs', None)
    logger_kwargs_override = kwargs.pop('logger_kwargs_override', None)
    batch_kwargs_override = kwargs.pop('batch_kwargs_override', None)

    # Set defaults for configs
    job_category = job_config.get('category', 'batch_processing')
    job_subcategory = job_config.get('subcategory', 'default')
    job_description = job_config.get('description', '')
    job_version = job_config.get('version', '1.0.0')
    environment = job_config.get('environment', 'production')
    extra_metadata = job_config.get('metadata', {})

    output_dir = storage_config.get('output_dir', './batch_logs')
    output_path_template = storage_config.get('path_template', '{job_category}/{job_subcategory}')
    s3_bucket = storage_config.get('s3_bucket')
    s3_prefix_template = storage_config.get('s3_prefix')
    s3_on_failure = storage_config.get('s3_on_failure')
    s3_retry_attempts = storage_config.get('s3_retries', 3)

    max_concurrency = processing_config.get('max_concurrency', 100)
    buffer_size = processing_config.get('buffer_size', 1000)
    show_progress = processing_config.get('show_progress', True)
    return_exceptions = processing_config.get('return_exceptions', True)
    return_results = processing_config.get('return_results', False)
    event_types = processing_config.get('event_types')
    partition_on = processing_config.get('partition_on', 'date')
    provider = processing_config.get('provider')

    prompt_col = column_config.get('prompt', 'prompt')
    config_col = column_config.get('config', 'config')
    tools_col = column_config.get('tools', 'tools')

    # Validate DataFrame has required columns
    if prompt_col not in df.columns:
        raise ValueError(f"DataFrame missing required column: {prompt_col}")
    
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