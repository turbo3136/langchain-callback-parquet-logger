"""
Example script demonstrating the batch_process function for automated batch processing.

This shows how to use batch_process with v2.0 API for both local-only and S3-enabled workflows.
"""

import asyncio
import pandas as pd
from pydantic import BaseModel
from langchain_callback_parquet_logger import (
    batch_process, with_tags,
    JobConfig, StorageConfig, ProcessingConfig, S3Config
)


# Example 1: Simple local-only processing
async def example_local_only():
    """Process a batch locally without S3."""

    # Prepare your data
    df = pd.DataFrame({
        'prompt': [
            'What is machine learning?',
            'Explain quantum computing',
            'How does blockchain work?',
            'What is the cloud?',
        ],
        'config': [
            with_tags(custom_id='ml_001'),
            with_tags(custom_id='qc_002'),
            with_tags(custom_id='bc_003'),
            with_tags(custom_id='cl_004'),
        ]
    })

    # Process locally - minimal config
    results = await batch_process(
        df,
        job_config=JobConfig(
            category="research",
            subcategory="tech_concepts",  # Optional
            description="Basic technology explanations"  # Optional
        ),
        storage_config=StorageConfig(
            output_dir="./batch_logs"
        ),
        processing_config=ProcessingConfig(
            return_results=True,  # Keep results in memory
            max_concurrency=10,
        )
    )

    # Add results to DataFrame
    if results:
        df['response'] = [r.content if hasattr(r, 'content') else str(r) for r in results]
        print(df[['prompt', 'response']].head())


# Example 2: Production with S3 upload
async def example_with_s3():
    """Process a batch with automatic S3 upload."""

    # Prepare data
    df = pd.DataFrame({
        'prompt': [
            f'Analyze sentiment: {text}'
            for text in [
                "This product is amazing!",
                "Terrible experience, would not recommend",
                "It's okay, nothing special",
                "Absolutely love it!",
            ]
        ],
        'config': [
            with_tags(custom_id=f'sentiment_{i:03d}', tags=['sentiment', 'analysis'])
            for i in range(4)
        ]
    })

    # Process with S3 upload - paths will mirror each other
    await batch_process(
        df,
        job_config=JobConfig(
            category="nlp",
            subcategory="sentiment_analysis",
            version="1.0.0",  # Optional
            environment="production"  # Optional
        ),
        storage_config=StorageConfig(
            output_dir="./temp_logs",
            path_template="{job_category}/{job_subcategory}/v{job_version_safe}/{date}",  # Version included in path
            s3_config=S3Config(
                bucket="my-ml-data",
                prefix="ml/"  # S3: ml/nlp/sentiment_analysis/v1_0_0/2024-01-15/
            )
        ),
        processing_config=ProcessingConfig(
            buffer_size=1000,
            max_concurrency=100,
            return_results=False,  # Don't keep in memory (data goes to S3)
        )
    )

    print(f"✅ Processed {len(df)} items and uploaded to S3")


# Example 3: Structured output with Pydantic
async def example_structured_output():
    """Process with structured output using Pydantic models."""

    class ProductReview(BaseModel):
        sentiment: str  # positive, negative, neutral
        confidence: float
        key_points: list[str]

    df = pd.DataFrame({
        'prompt': [
            'Review: ' + review for review in [
                "Great laptop, fast and reliable. Battery life could be better.",
                "Worst phone ever. Screen broke on day one.",
                "Decent headphones for the price. Sound quality is acceptable.",
            ]
        ],
        'config': [
            with_tags(custom_id=f'review_{i:03d}')
            for i in range(3)
        ]
    })

    results = await batch_process(
        df,
        structured_output=ProductReview,
        job_config=JobConfig(
            category="ecommerce",
            subcategory="review_analysis"
        ),
        storage_config=StorageConfig(
            output_dir="./structured_logs"
        ),
        processing_config=ProcessingConfig(
            event_types=['llm_start', 'llm_end'],  # Log specific events
            return_results=True,
        )
    )

    # Process structured results
    if results:
        for i, result in enumerate(results):
            if isinstance(result, ProductReview):
                print(f"Review {i}: Sentiment={result.sentiment}, Confidence={result.confidence:.2f}")


# Example 4: Custom LLM configuration
async def example_custom_llm():
    """Use custom LLM settings."""

    df = pd.DataFrame({
        'prompt': ['Translate to French: Hello world'] * 5,
        'config': [with_tags(custom_id=f'trans_{i:03d}') for i in range(5)]
    })

    await batch_process(
        df,
        job_config=JobConfig(
            category="translation",
            version="2.0.0"
        ),
        storage_config=StorageConfig(
            output_dir="./batch_logs",
            path_template="{job_category}/v{job_version_safe}/{date}"  # Using sanitized version (dots -> underscores)
        ),
        processing_config=ProcessingConfig(
            buffer_size=10,  # Flush frequently for testing
        ),
        llm_kwargs={
            'model': 'gpt-4',
            'temperature': 0.3,
            'max_tokens': 100,
        }
    )


# Example 5: Memory-efficient huge dataset processing
async def example_huge_dataset():
    """Process huge dataset without keeping results in memory."""

    # Simulate a large dataset
    huge_df = pd.DataFrame({
        'prompt': [f'Process item {i}' for i in range(10000)],
        'config': [with_tags(custom_id=f'item_{i:06d}') for i in range(10000)]
    })

    # Process in batches, write to parquet, don't keep in memory
    await batch_process(
        huge_df,
        job_config=JobConfig(
            category="bulk_processing",
            subcategory="large_dataset"
        ),
        storage_config=StorageConfig(
            output_dir="./bulk_logs",
            s3_config=S3Config(
                bucket="data-lake",
                prefix="bulk/"  # S3: bulk/bulk_processing/large_dataset/vunversioned/ (no version specified)
            )
        ),
        processing_config=ProcessingConfig(
            buffer_size=5000,  # Large buffer for efficiency
            max_concurrency=500,  # High concurrency
            return_results=False,  # DON'T keep 10k results in memory
            show_progress=True,  # Show real-time progress bar
        )
    )

    print("✅ Huge dataset processed - results in parquet files")


# Example 6: Minimal configuration
async def example_minimal():
    """Simplest possible usage with minimal config."""

    df = pd.DataFrame({
        'prompt': ['What is AI?', 'What is ML?'],
        'config': [with_tags(custom_id='q1'), with_tags(custom_id='q2')]
    })

    # Minimal config - just category is required
    results = await batch_process(
        df,
        job_config=JobConfig(category="test")
    )

    print(f"✅ Processed {len(df)} items with minimal config")


# Example 7: Environment-based configuration
async def example_environment_based():
    """Use environment variables for configuration."""
    import os

    # Set environment variable (normally this would be in your shell/container)
    os.environ['LANGCHAIN_S3_BUCKET'] = 'my-default-bucket'

    df = pd.DataFrame({
        'prompt': ['Test prompt'],
        'config': [with_tags(custom_id='test_001')]
    })

    # S3 bucket will be read from environment if not specified
    await batch_process(
        df,
        job_config=JobConfig(
            category="test",
            environment=os.environ.get('ENVIRONMENT', 'development')
        )
        # S3 bucket will be auto-configured from LANGCHAIN_S3_BUCKET env var
    )


if __name__ == "__main__":
    # Run the examples you want to test
    print("Running minimal example...")
    asyncio.run(example_minimal())

    print("\nRunning local-only example...")
    asyncio.run(example_local_only())

    # Uncomment to run other examples:
    # asyncio.run(example_with_s3())
    # asyncio.run(example_structured_output())
    # asyncio.run(example_custom_llm())
    # asyncio.run(example_huge_dataset())
    # asyncio.run(example_environment_based())