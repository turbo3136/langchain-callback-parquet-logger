"""
Simple batch processing example using OpenAI Responses API with web search and structured output.
This example shows how to use web search to get weather data and return it in a structured format.
"""

import asyncio
from typing import Any, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_callback_parquet_logger import ParquetLogger

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()


# ---------- 1) Define structured output schema for weather data ----------
class WeatherData(BaseModel):
    """Structured weather information from web search."""
    location: str = Field(description="The city/location for the weather report")
    temperature: Optional[float] = Field(description="Temperature in Fahrenheit for today", default=None)
    cloud_cover: Optional[str] = Field(description="Description of cloud cover (e.g., clear, partly cloudy, overcast)", default=None)
    precipitation_probability: Optional[float] = Field(description="Probability of precipitation as percentage (0-100)", default=None)
    summary: str = Field(description="Brief weather summary for the day")


# ---------- 2) Main async function ----------
async def main():
    # Create logger that saves to Parquet files
    with ParquetLogger(log_dir="./llm_batch_logs", buffer_size=50) as logger:
        
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
        
        # Define function that processes each row
        async def process_row(row: Dict[str, Any]) -> WeatherData:
            """Process a single input using web search."""
            # Pass web search tool directly - warning is harmless
            return await structured_llm.ainvoke(
                input=row.get("input"),
                tools=[
                    {
                        "type": "web_search",
                        "search_context_size": "low",
                    },
                ],
            )
        
        # Create RunnableLambda wrapper for batch processing
        runner = RunnableLambda(process_row)
        
        # Example weather queries for different cities
        rows = [
            {"input": "What is today's weather in New York City? Use the structured output requested."},
            {"input": "What is today's weather in Los Angeles? Use the structured output requested."},
            # {"input": "What is today's weather in Chicago? Use the structured output requested."},
            # {"input": "What is today's weather in Miami? Use the structured output requested."},
            # {"input": "What is today's weather in Seattle? Use the structured output requested."},
        ]
        
        # Process all rows concurrently with max_concurrency in config
        print("🔍 Fetching weather data using web search...\n")
        results = await runner.abatch(
            rows,
            config={"max_concurrency": 3},  # Limit concurrent requests
            return_exceptions=True,         # Return exceptions instead of raising
        )
        
        # Display structured results
        print("=" * 80)
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