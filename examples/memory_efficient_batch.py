"""
Example showing memory-efficient batch processing for huge DataFrames.
When return_results=False, results are only saved to ParquetLogger, not kept in memory.
"""

import asyncio
import pandas as pd
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_callback_parquet_logger import ParquetLogger, with_tags, batch_run


async def main():
    """Demonstrate memory-efficient batch processing."""
    
    # Simulate a large DataFrame (in practice this could be millions of rows)
    print("Creating large DataFrame...")
    df = pd.DataFrame({
        'id': range(1000),  # Imagine this is 1M+ rows
        'text': [f"Text sample {i}" for i in range(1000)],
    })
    print(f"DataFrame has {len(df):,} rows")
    
    # Prepare columns
    df['prompt'] = df['text'].apply(lambda x: f"Summarize in 3 words: {x}")
    df['config'] = df['id'].apply(lambda x: with_tags(custom_id=f"batch-{x}"))
    
    # Configure logger with appropriate buffer for large batches
    with ParquetLogger(
        log_dir="./large_batch_logs",
        buffer_size=500,  # Larger buffer for efficiency
        provider="openai",
        logger_metadata={
            "batch_type": "large_scale_demo",
            "total_rows": len(df),
        }
    ) as logger:
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            callbacks=[logger]
        )
        
        print("\n" + "="*60)
        print("MEMORY-EFFICIENT BATCH PROCESSING")
        print("="*60)
        print("\nProcessing without keeping results in memory...")
        print("Results will be saved to ParquetLogger only.\n")
        
        # Process WITHOUT keeping results in memory
        result = await batch_run(
            df=df,
            llm=llm,
            max_concurrency=10,
            show_progress=True,
            return_results=False  # <-- KEY: Don't keep results in memory
        )
        
        print(f"\nReturn value: {result}")  # Will be None
        print("✅ Processing complete! Results are in parquet files.")
        
        # Show where logs are saved
        today = date.today()
        log_path = f"./large_batch_logs/date={today}/"
        print(f"📁 Results saved to: {log_path}")
        
        print("\n" + "="*60)
        print("MEMORY USAGE COMPARISON")
        print("="*60)
        print("""
With return_results=True (default):
  - DataFrame: Original memory usage
  - Results list: Additional memory for all responses
  - Total: 2x memory usage
  
With return_results=False (memory-efficient):
  - DataFrame: Original memory usage
  - Results: No additional memory (saved to parquet only)
  - Total: 1x memory usage
  
For 1M rows with 1KB responses:
  - Normal mode: ~2GB+ memory
  - Efficient mode: ~1GB memory (50% reduction)
""")
        
        # Show how to read results from parquet later
        print("="*60)
        print("READING RESULTS FROM PARQUET")
        print("="*60)
        print("""
To analyze results later:

import pandas as pd
import json

# Read the saved logs
df_logs = pd.read_parquet('./large_batch_logs')

# Filter to completion events with your custom IDs
completions = df_logs[
    (df_logs['event_type'] == 'llm_end') & 
    (df_logs['logger_custom_id'].str.startswith('batch-'))
]

# Extract results
for _, row in completions.iterrows():
    custom_id = row['logger_custom_id']
    payload = json.loads(row['payload'])
    response = payload['response']
    print(f"{custom_id}: {response}")
""")


if __name__ == "__main__":
    asyncio.run(main())