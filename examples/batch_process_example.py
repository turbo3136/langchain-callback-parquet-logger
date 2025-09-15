"""
Example script demonstrating the batch_process function for automated batch processing.

This shows how to use batch_process for both local-only and S3-enabled workflows.
"""

import asyncio
import pandas as pd
from pydantic import BaseModel
from langchain_callback_parquet_logger import batch_process, with_tags


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
    
    # Process locally
    results = await batch_process(
        df,
        job_category="research",
        job_subcategory="tech_concepts",
        job_description="Basic technology explanations",
        output_dir="./batch_logs",
        return_results=True,  # Keep results in memory
        max_concurrency=10,
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
    
    # Process with S3 upload
    await batch_process(
        df,
        job_category="nlp",
        job_subcategory="sentiment_analysis", 
        job_version="1.0.0",
        environment="production",
        output_dir="./temp_logs",
        s3_bucket="my-ml-data",  # Enable S3 upload
        s3_prefix_template="ml/{job_category}/{job_subcategory}/{date}/",
        buffer_size=1000,
        max_concurrency=100,
        return_results=False,  # Don't keep in memory (data goes to S3)
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
        job_category="ecommerce",
        job_subcategory="review_analysis",
        output_dir="./structured_logs",
        event_types=['llm_start', 'llm_end'],  # Log specific events
        return_results=True,
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
        job_category="translation",
        llm_kwargs={
            'model': 'gpt-4',
            'temperature': 0.3,
            'max_tokens': 100,
        },
        model_kwargs={
            'presence_penalty': 0.5,
        },
        output_path_template="{job_category}/v{job_version}/{date}",
        job_version="2.0.0",
        buffer_size=10,  # Flush frequently for testing
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
        job_category="bulk_processing",
        job_subcategory="large_dataset",
        output_dir="./bulk_logs",
        s3_bucket="data-lake",  # Optional: also upload to S3
        buffer_size=5000,  # Large buffer for efficiency
        max_concurrency=500,  # High concurrency
        return_results=False,  # DON'T keep 10k results in memory
        show_progress=True,  # Show progress bar
    )
    
    print("✅ Huge dataset processed - results in parquet files")


# Example 6: Environment-based configuration
async def example_environment_based():
    """Use environment variables for configuration."""
    import os
    
    # Set environment variable (normally this would be in your shell/container)
    os.environ['LANGCHAIN_S3_BUCKET'] = 'my-default-bucket'
    
    df = pd.DataFrame({
        'prompt': ['Test prompt'],
        'config': [with_tags(custom_id='test_001')]
    })
    
    # S3 bucket will be read from environment
    await batch_process(
        df,
        job_category="test",
        environment=os.environ.get('ENVIRONMENT', 'development'),
        # s3_bucket not specified - will use LANGCHAIN_S3_BUCKET
    )


if __name__ == "__main__":
    # Run the examples you want to test
    print("Running local-only example...")
    asyncio.run(example_local_only())
    
    # Uncomment to run other examples:
    # asyncio.run(example_with_s3())
    # asyncio.run(example_structured_output())
    # asyncio.run(example_custom_llm())
    # asyncio.run(example_huge_dataset())
    # asyncio.run(example_environment_based())