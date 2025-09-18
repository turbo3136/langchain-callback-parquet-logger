"""
Batch processing with v2.0 API - showing both low-level batch_run and high-level batch_process.
"""

import asyncio
import pandas as pd
from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_callback_parquet_logger import (
    ParquetLogger, with_tags, batch_run,
    S3Config
)


async def low_level_batch_processing(
    df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    log_dir: str = "./batch_logs",
    max_concurrency: int = 10,
    memory_efficient: bool = False,
    s3_config: Optional[S3Config] = None,
    logger_metadata: Optional[Dict[str, Any]] = None,
    custom_id_column: Optional[str] = None,
) -> Optional[List]:
    """
    Low-level batch processing using batch_run with manual logger setup.
    Use this when you need direct control over logging configuration.

    Args:
        df: DataFrame with a 'prompt' column (required)
        model: OpenAI model to use
        log_dir: Directory for parquet logs
        max_concurrency: Max parallel requests
        memory_efficient: If True, don't keep results in memory (for huge DataFrames)
        s3_config: Optional S3Config for log upload
        logger_metadata: Metadata to include in all logs
        custom_id_column: Column name to use for custom IDs (for tracking)

    Returns:
        List of results (or None if memory_efficient=True)
    """
    # Validate DataFrame
    if 'prompt' not in df.columns:
        raise ValueError("DataFrame must have a 'prompt' column")

    # Add config column with custom IDs if specified
    if custom_id_column and custom_id_column in df.columns:
        df['config'] = df[custom_id_column].apply(lambda x: with_tags(custom_id=str(x)))
    else:
        df['config'] = df.index.to_series().apply(lambda x: with_tags(custom_id=f"row-{x}"))

    # Setup logger with optional S3
    logger_config = {
        "log_dir": log_dir,
        "buffer_size": 100 if not memory_efficient else 500,
        "logger_metadata": logger_metadata or {"batch_size": len(df), "model": model},
        "event_types": ['llm_start', 'llm_end', 'llm_error'],  # v2.0 explicit event types
        "s3_config": s3_config  # v2.0 uses S3Config object
    }

    # Process with context manager
    with ParquetLogger(**logger_config) as logger:
        llm = ChatOpenAI(model=model, callbacks=[logger])

        print(f"📊 Processing {len(df)} rows with {model}")
        print(f"💾 Logs will be saved to: {log_dir}")
        if s3_config:
            print(f"☁️  Also uploading to S3: {s3_config.bucket}/{s3_config.prefix}")

        # Run batch processing
        results = await batch_run(
            df,
            llm,
            max_concurrency=max_concurrency,
            return_results=not memory_efficient,
            show_progress=True
        )

        if memory_efficient:
            print(f"✅ Processed {len(df)} rows (results in logs only)")
            return None
        else:
            print(f"✅ Processed {len(df)} rows with {len(results)} results")
            return results


def read_batch_results(log_dir: str = "./batch_logs") -> pd.DataFrame:
    """
    Read and parse batch processing results with v2.0 payload structure.

    Args:
        log_dir: Directory containing parquet logs

    Returns:
        DataFrame with parsed results
    """
    import json

    # Read all parquet files
    df_logs = pd.read_parquet(log_dir)

    # Filter to completed LLM calls only
    df_results = df_logs[df_logs['event_type'] == 'llm_end'].copy()

    # Parse v2.0 payload structure
    def parse_payload(payload_str):
        payload = json.loads(payload_str)

        # Handle different response structures
        response = payload.get('data', {}).get('outputs', {}).get('response', {})
        if isinstance(response, dict):
            content = response.get('content', '')
        else:
            content = str(response)

        # Extract usage safely
        usage = payload.get('data', {}).get('outputs', {}).get('usage', {})
        tokens = usage.get('total_tokens', 0) if usage else 0

        return {
            'custom_id': payload.get('execution', {}).get('custom_id', ''),
            'response': content,
            'llm_type': payload.get('data', {}).get('llm_type', ''),
            'tokens': tokens,
            'timestamp': payload.get('timestamp', '')
        }

    # Extract structured data
    parsed_data = df_results['payload'].apply(parse_payload)
    df_parsed = pd.DataFrame(list(parsed_data))

    # Combine with log metadata
    df_final = pd.concat([
        df_results[['run_id', 'logger_custom_id']].reset_index(drop=True),
        df_parsed
    ], axis=1)

    return df_final


async def main():
    """Example usage of low-level batch processing with v2.0 API."""

    # Example 1: Simple batch processing
    print("=" * 60)
    print("Example 1: Simple Batch Processing")
    print("=" * 60)

    df_simple = pd.DataFrame({
        'id': ['A001', 'A002', 'A003'],
        'prompt': [
            'What is machine learning?',
            'Explain quantum computing in one sentence',
            'What is the speed of light?'
        ]
    })

    results = await low_level_batch_processing(
        df_simple,
        custom_id_column='id',
        log_dir="./example1_logs"
    )

    if results:
        df_simple['response'] = [r.content if hasattr(r, 'content') else str(r) for r in results]
        print("\nResults:")
        print(df_simple[['id', 'prompt']].head())

    # Example 2: Large-scale memory-efficient processing
    print("\n" + "=" * 60)
    print("Example 2: Memory-Efficient Processing (Large Dataset)")
    print("=" * 60)

    # Simulate large dataset
    df_large = pd.DataFrame({
        'doc_id': [f'DOC-{i:05d}' for i in range(10)],
        'prompt': [f'Summarize document {i}: [long text here...]' for i in range(10)]
    })

    await low_level_batch_processing(
        df_large,
        custom_id_column='doc_id',
        log_dir="./example2_logs",
        memory_efficient=True,  # Don't keep results in memory
        max_concurrency=20,
        logger_metadata={'job': 'document_summarization', 'version': '2.0'}
    )

    # Read results from logs
    print("\nReading results from logs...")
    df_results = read_batch_results("./example2_logs")
    print(f"Found {len(df_results)} completed responses")
    print("\nSample results:")
    print(df_results.head()[['custom_id', 'response', 'tokens']].to_string())

    # Example 3: With S3 upload (requires AWS credentials)
    print("\n" + "=" * 60)
    print("Example 3: With S3 Upload (if configured)")
    print("=" * 60)

    # Check if user wants S3 (by setting S3_BUCKET env var)
    import os
    if os.getenv('S3_BUCKET'):
        df_s3 = pd.DataFrame({
            'task_id': ['T001', 'T002'],
            'prompt': ['Hello world', 'Goodbye world']
        })

        # Create S3Config for v2.0
        s3_config = S3Config(
            bucket=os.getenv('S3_BUCKET'),
            prefix="batch-runs/",
            on_failure="continue"  # Fail gracefully
        )

        await low_level_batch_processing(
            df_s3,
            custom_id_column='task_id',
            log_dir="./example3_logs",
            s3_config=s3_config,
            logger_metadata={'environment': 'production'}
        )
        print("✅ S3 upload configured (will fail gracefully if AWS credentials not set)")
    else:
        print("Skipping S3 example (S3_BUCKET env var not set)")

    # Example 4: Using high-level batch_process (recommended for most users)
    print("\n" + "=" * 60)
    print("Example 4: High-Level batch_process API (Recommended)")
    print("=" * 60)

    # Import the high-level API
    from langchain_callback_parquet_logger import (
        batch_process, JobConfig, StorageConfig, ProcessingConfig
    )

    df_highlevel = pd.DataFrame({
        'prompt': ['What is Python?', 'What is JavaScript?'],
        'config': [with_tags(custom_id='py'), with_tags(custom_id='js')]
    })

    # Use the simpler high-level API
    results = await batch_process(
        df_highlevel,
        job_config=JobConfig(category="languages"),  # Minimal config
        processing_config=ProcessingConfig(
            return_results=True,
            show_progress=True
        )
    )

    print(f"✅ High-level API processed {len(df_highlevel)} items")
    print("\nNote: Use batch_process() for simpler API, batch_run() for direct control")


if __name__ == "__main__":
    asyncio.run(main())