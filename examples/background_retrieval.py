"""
Background response retrieval from OpenAI.

Prerequisites:
1. Install: pip install "langchain-callback-parquet-logger[background]"
2. Have OpenAI API key configured
3. Have response IDs from background/async OpenAI requests
"""

import asyncio
import pandas as pd
import openai
from langchain_callback_parquet_logger import ParquetLogger, retrieve_background_responses


async def main():
    """Retrieve completed background responses from OpenAI."""
    
    # Example response IDs from background requests
    # In practice, get these from your logs where status='queued'
    df = pd.DataFrame({
        'response_id': [
            'batch_req_abc123',  # Replace with real response IDs
            'batch_req_def456',
            'batch_req_ghi789'
        ],
        'logger_custom_id': [
            'user-001-req-001',
            'user-001-req-002', 
            'user-002-req-001'
        ]
    })
    
    print("📋 Background Response Retrieval")
    print("=" * 60)
    print(f"Retrieving {len(df)} responses...")
    print(df.to_string())
    print()
    
    # Initialize OpenAI client
    client = openai.AsyncClient()
    
    # Retrieve responses with checkpoint support
    with ParquetLogger('./retrieval_logs') as logger:
        results = await retrieve_background_responses(
            df,
            client,
            logger=logger,
            show_progress=True,
            checkpoint_file='./checkpoint.parquet',  # Resume if interrupted
            max_concurrent=5,
            max_retries=3
        )
        
    if results:
        print(f"\n✅ Retrieved {len(results)} responses")
        df['response'] = results
        print("\nResults:")
        print(df[['logger_custom_id', 'response']].head())
    else:
        print("\n⚠️  Memory-efficient mode - results saved to logs only")
    
    # Read retrieval events from logs
    print("\n📊 Retrieval Statistics:")
    df_logs = pd.read_parquet('./retrieval_logs')
    
    attempts = len(df_logs[df_logs['event_type'] == 'background_retrieval_attempt'])
    completions = len(df_logs[df_logs['event_type'] == 'background_retrieval_complete'])
    errors = len(df_logs[df_logs['event_type'] == 'background_retrieval_error'])
    
    print(f"  Attempts: {attempts}")
    print(f"  Completed: {completions}")
    print(f"  Errors: {errors}")


if __name__ == "__main__":
    # Note: This requires real response IDs from OpenAI background requests
    # For testing, you can modify to use mock data
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"⚠️  Example requires real OpenAI response IDs: {e}")
        print("   Replace the response_ids in the DataFrame with actual values")