"""
Basic usage example showing simple logging with the ParquetLogger.
"""

from langchain_callback_parquet_logger import ParquetLogger
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    # Create a logger instance with metadata
    logger = ParquetLogger(
        log_dir="./llm_logs",
        buffer_size=10,  # Flush every 10 messages
        provider="openai",
        logger_metadata={  # Logger-level metadata included in all logs
            "environment": "development",
            "service": "example-app",
            "version": "1.0.0"
        }
    )
    
    # Initialize LLM with the logger
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        callbacks=[logger]
    )
    
    # Example conversations
    questions = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What are the benefits of exercise?",
        "How do airplanes fly?",
    ]
    
    print("Starting conversations...\n")
    print("=" * 60)
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        
        try:
            # You can include a logger_custom_id with each request for tracking
            response = llm.invoke(
                question,
                metadata={"logger_custom_id": f"question-{i}"}  # Custom ID for this specific request
            )
            print(f"Response: {response.content[:200]}...")  # Show first 200 chars
        except Exception as e:
            print(f"Error: {e}")
    
    # Manually flush any remaining logs
    logger.flush()
    
    print("\n" + "=" * 60)
    print(f"✅ Completed {len(questions)} conversations")
    print(f"📁 Logs saved to: ./llm_logs/")
    print("=" * 60)


if __name__ == "__main__":
    main()