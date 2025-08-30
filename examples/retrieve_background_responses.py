"""
Example of retrieving background responses from OpenAI.

This example demonstrates how to retrieve completed responses for 
background/async OpenAI requests using the retrieve_background_responses function.
"""

import asyncio
import json
import pandas as pd
from datetime import datetime
import openai
from langchain_callback_parquet_logger import (
    ParquetLogger, 
    retrieve_background_responses
)


async def main():
    """Demonstrate background response retrieval."""
    
    # Example 1: Create sample data with response IDs
    # In practice, these would come from your logs where status='queued'
    df = pd.DataFrame({
        'response_id': [
            'resp_xxx',
            'resp_yyy',
            'resp_zzz'
        ],
        'logger_custom_id': [
            'user-123-req-001',
            'user-456-req-002', 
            'user-789-req-003'
        ],
        'original_prompt': [
            'What is the capital of France?',
            'Explain quantum computing',
            'Write a haiku about coding'
        ]
    })
    
    print("📋 Response IDs to retrieve:")
    print(df[['response_id', 'logger_custom_id']].to_string())
    print("\n" + "="*60 + "\n")
    
    # Initialize OpenAI client
    client = openai.AsyncClient()
    
    # Example 2: Retrieve with logging
    print("🔄 Retrieving responses with full logging...")
    
    with ParquetLogger('./retrieval_logs', buffer_size=10) as logger:
        results = await retrieve_background_responses(
            df,
            client,
            logger=logger,
            show_progress=True,
            checkpoint_file='./retrieval_checkpoint.parquet'
        )
    
    # Display results
    if results is not None:
        print("\n✅ Retrieval Results:")
        print("="*60)
        
        for idx, row in results.iterrows():
            print(f"\n📌 Response ID: {row['response_id'][:20]}...")
            print(f"   Status: {row['status']}")
            
            if row['status'] == 'completed' and row['openai_response']:
                # Extract content from OpenAI response
                response = row['openai_response']
                if 'choices' in response and response['choices']:
                    content = response['choices'][0].get('message', {}).get('content', 'N/A')
                    print(f"   Content: {content[:100]}...")
                if 'usage' in response:
                    print(f"   Tokens: {response['usage'].get('total_tokens', 'N/A')}")
            elif row['error']:
                print(f"   Error: {row['error']}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Memory-efficient mode for large datasets
    print("💾 Memory-efficient mode example (no results returned)...")
    
    # Simulate large DataFrame
    large_df = pd.DataFrame({
        'response_id': [f'resp_{i:06d}' for i in range(1000)],
        'logger_custom_id': [f'batch-{i//100}-item-{i:04d}' for i in range(1000)]
    })
    
    with ParquetLogger('./large_retrieval_logs', buffer_size=100) as logger:
        # This won't return results, only logs to parquet
        await retrieve_background_responses(
            large_df[:10],  # Just process first 10 for demo
            client,
            logger=logger,
            return_results=False,  # Don't keep in memory
            show_progress=True
        )
    
    print("\n📁 Logs saved to ./large_retrieval_logs/")
    
    # Example 4: Reading the logged results
    print("\n📊 Reading logged results from parquet:")
    print("="*60)
    
    # Read the retrieval logs
    log_df = pd.read_parquet('./retrieval_logs')
    
    # Filter for completed retrievals
    completed = log_df[log_df['event_type'] == 'background_retrieval_complete']
    print(f"Found {len(completed)} completed retrievals")
    
    # Parse payload to get response data
    if len(completed) > 0:
        first_complete = completed.iloc[0]
        payload = json.loads(first_complete['payload'])
        print(f"\nExample completed retrieval:")
        print(f"  Custom ID: {first_complete['logger_custom_id']}")
        print(f"  Response ID: {payload['response_id'][:20]}...")
        print(f"  Status: {payload['status']}")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
    
    # For Jupyter notebooks, use:
    # await main()