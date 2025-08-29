"""
Minimal batch processing helper for DataFrame operations with LangChain.
Focuses on simplicity - users prepare data in DataFrame columns.
"""

import asyncio
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from langchain_core.runnables import RunnableLambda


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