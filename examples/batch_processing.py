"""
Simple batch processing example using OpenAI Responses API with web search and structured output.
This example shows how to use web search to get weather data and return it in a structured format.
"""

import asyncio
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_callback_parquet_logger import ParquetLogger


# ---------- 1) Progress tracking utility for batch operations ----------
class AsyncProgress:
    """Simple async-safe progress tracker for batch operations."""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.completed = 0
        self.desc = desc
        self._lock = asyncio.Lock()
    
    async def update(self):
        """Update progress display."""
        async with self._lock:
            self.completed += 1
            percent = (self.completed * 100) // self.total
            print(f"\r{self.desc}: {self.completed}/{self.total} ({percent}%) completed", 
                  end="" if self.completed < self.total else "\n", flush=True)


# ---------- 2) Define structured output schema for weather data ----------
class WeatherData(BaseModel):
    """Structured weather information from web search."""
    location: str = Field(description="The city/location for the weather report")
    temperature: Optional[float] = Field(description="Temperature in Fahrenheit for today", default=None)
    cloud_cover: Optional[str] = Field(description="Description of cloud cover (e.g., clear, partly cloudy, overcast)", default=None)
    precipitation_probability: Optional[float] = Field(description="Probability of precipitation as percentage (0-100)", default=None)
    summary: str = Field(description="Brief weather summary for the day")


# ---------- 3) Main async function ----------
async def main():
    # Create logger that saves to Parquet files with all parameters explicitly shown
    with ParquetLogger(
        log_dir="./llm_batch_logs",  # Directory to save parquet files
        buffer_size=50,               # Flush to disk after 50 log entries
        provider="openai"             # LLM provider name for tracking
    ) as logger:
        
        # Initialize the LLM with Responses API
        llm = ChatOpenAI(
            model="gpt-5-nano",  # Use a model that supports web search
            reasoning={"effort": "low"},  # low reasoning for testing
            use_responses_api=True,        # Use Responses API for web search support
            output_version="responses/v1", # Updated message shape for Responses API
            callbacks=[logger],             # Attach logger to track all interactions
        )
        
        # Apply structured output to the LLM
        structured_llm = llm.with_structured_output(WeatherData)
        
        # Create progress tracker for batch operations
        progress = AsyncProgress(total=5, desc="🔍 Fetching weather data")
        
        # Define function that processes each row with progress tracking
        async def process_row_with_progress(row: Dict[str, Any]) -> WeatherData:
            """Process a single input using web search and update progress."""
            # Pass web search tool directly - warning is harmless
            result = await structured_llm.ainvoke(
                input=row.get("input"),
                tools=[
                    {
                        "type": "web_search",
                        "search_context_size": "low",
                    },
                ],
            )
            await progress.update()  # Update progress after each completion
            return result
        
        # Create RunnableLambda wrapper for batch processing
        runner = RunnableLambda(process_row_with_progress)
        
        # Example weather queries for different cities
        rows = [
            {"input": "What is today's weather in New York City? Use the structured output requested."},
            {"input": "What is today's weather in Los Angeles? Use the structured output requested."},
            {"input": "What is today's weather in Chicago? Use the structured output requested."},
            {"input": "What is today's weather in Miami? Use the structured output requested."},
            {"input": "What is today's weather in Seattle? Use the structured output requested."},
        ]
        
        # Process all rows concurrently with max_concurrency in config
        # Progress will be shown as: "🔍 Fetching weather data: 3/5 (60%) completed"
        results = await runner.abatch(
            rows,
            config={"max_concurrency": 3},  # Limit concurrent requests
            return_exceptions=True,         # Return exceptions instead of raising
        )
        
        # Display structured results
        print("\n" + "=" * 80)
        print("WEATHER REPORT RESULTS")
        print("=" * 80)
        
        for i, result in enumerate(results, 1):
            if isinstance(result, Exception):
                print(f"\n❌ Query {i} Error: {result}")
            else:
                print(f"\n📍 {result.location}")
                print(f"   🌡️  Temperature: {result.temperature}°F" if result.temperature else "   🌡️  Temperature: N/A")
                print(f"   ☁️  Cloud Cover: {result.cloud_cover}" if result.cloud_cover else "   ☁️  Cloud Cover: N/A")
                print(f"   💧 Precipitation: {result.precipitation_probability}%" if result.precipitation_probability else "   💧 Precipitation: N/A")
                print(f"   📝 Summary: {result.summary}")
        
        print("\n" + "=" * 80)
        print(f"✅ Processed {len(results)} weather queries")
        print(f"📁 Logs saved to: ./llm_batch_logs/")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())