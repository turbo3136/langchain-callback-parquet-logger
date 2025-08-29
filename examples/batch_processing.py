"""
Batch processing example using the minimal batch_run helper with all advanced features.
Shows service_tier="flex", model_kwargs, structured output, and comprehensive logging.
"""

import asyncio
import pandas as pd
from typing import Optional
from datetime import date
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_callback_parquet_logger import ParquetLogger, with_tags, batch_run


# Define structured output schema for weather data
class WeatherData(BaseModel):
    """Structured weather information from web search."""
    location: str = Field(description="The city/location for the weather report")
    temperature: Optional[float] = Field(description="Temperature in Fahrenheit", default=None)
    conditions: Optional[str] = Field(description="Weather conditions", default=None)
    summary: str = Field(description="Brief weather summary")


async def main():
    """Demonstrate batch processing with all advanced features."""
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'city_id': ['NYC-001', 'LA-002', 'CHI-003', 'MIA-004', 'SEA-005'],
        'city': ['New York', 'Los Angeles', 'Chicago', 'Miami', 'Seattle'],
        'state': ['NY', 'CA', 'IL', 'FL', 'WA'],
    })
    
    print("📊 Processing weather data for cities:")
    print(df[['city_id', 'city', 'state']].to_string())
    print("\n" + "="*60 + "\n")
    
    # Step 1: Prepare DataFrame columns
    # Create prompts
    df['prompt'] = df.apply(
        lambda r: f"What is today's weather in {r['city']}, {r['state']}? "
                  f"Provide temperature and conditions.",
        axis=1
    )
    
    # Create config with custom IDs and tags
    df['config'] = df.apply(
        lambda r: with_tags(
            custom_id=r['city_id'],
            tags=['weather-batch', 'flex-tier', f"state-{r['state']}"]
        ),
        axis=1
    )
    
    # Add web search tool to all rows
    df['tools'] = [[{
        "type": "web_search",
        "search_context_size": "low"
    }]] * len(df)
    
    # Step 2: Setup logger with comprehensive metadata
    logger_config = {
        "log_dir": "./batch_logs",
        "buffer_size": 50,
        "provider": "openai",
        "partition_on": "date",
        "logger_metadata": {
            "batch_type": "weather_forecast",
            "service_tier": "flex",
            "background_processing": True,
            "api_version": "v2",
            "environment": "production",
            "team": "data-ops",
            "cost_center": "research",
        }
    }
    
    # Step 3: Configure LLM with all advanced features
    with ParquetLogger(**logger_config) as logger:
        # Initialize LLM with service_tier and model_kwargs
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            service_tier="flex",  # Using flex tier for cost optimization
            model_kwargs={
                "background": True,  # Background processing
                "prompt_cache_key": "weather-batch-v1",  # Cache key for prompt reuse
            },
            callbacks=[logger],  # Attach logger
            temperature=0,  # Consistent results
        )
        
        # Apply structured output
        structured_llm = llm.with_structured_output(WeatherData)
        
        # Step 4: Run batch processing with minimal helper
        print("🚀 Starting batch processing with flex tier...\n")
        
        results = await batch_run(
            df=df,
            llm=structured_llm,
            prompt_col='prompt',
            config_col='config',
            tools_col='tools',
            max_concurrency=3,  # Process 3 at a time
            show_progress=True,  # Show progress bar
            return_exceptions=True  # Don't fail whole batch on errors
        )
        
        # Step 5: Add results to DataFrame
        df['result'] = results
        
        # Step 6: Display results
        print("\n" + "="*60)
        print("WEATHER RESULTS")
        print("="*60 + "\n")
        
        for idx, row in df.iterrows():
            print(f"📍 {row['city']}, {row['state']} (ID: {row['city_id']})")
            
            result = row['result']
            if isinstance(result, Exception):
                print(f"   ❌ Error: {result}")
            else:
                print(f"   🌡️  Temperature: {result.temperature}°F")
                print(f"   ☁️  Conditions: {result.conditions}")
                print(f"   📝 Summary: {result.summary}")
            print()
        
        # Step 7: Save results to CSV
        df['temperature'] = df['result'].apply(
            lambda x: x.temperature if not isinstance(x, Exception) else None
        )
        df['conditions'] = df['result'].apply(
            lambda x: x.conditions if not isinstance(x, Exception) else None
        )
        
        output_df = df[['city_id', 'city', 'state', 'temperature', 'conditions']]
        output_df.to_csv('weather_results.csv', index=False)
        print(f"💾 Results saved to weather_results.csv")
        
        # Show where logs are saved
        today = date.today()
        print(f"📁 Logs saved to: {logger_config['log_dir']}/date={today}/")
        print("\n" + "="*60)
        print("✅ Batch processing complete!")
        print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
    # For notebooks use: await main()