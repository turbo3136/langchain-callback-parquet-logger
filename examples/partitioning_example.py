"""
Example showing different partitioning options for the ParquetLogger.
"""

from langchain_callback_parquet_logger import ParquetLogger, with_tags
from langchain_community.chat_models.fake import FakeListChatModel


def main():
    print("=" * 60)
    print("ParquetLogger Partitioning Examples")
    print("=" * 60)
    
    # Example 1: Default behavior - date partitioning
    print("\n1. Date partitioning (default):")
    print("-" * 40)
    
    logger_with_date = ParquetLogger(
        log_dir="./logs_with_date_partitions",
        buffer_size=1,  # Flush immediately for demo
        partition_on="date"  # This is the default
    )
    
    llm1 = FakeListChatModel(
        responses=["Response with date partitioning"],
        callbacks=[logger_with_date]
    )
    
    response = llm1.invoke("Test message", config=with_tags(custom_id="test-date-partition"))
    print(f"Response: {response.content}")
    print(f"Files will be saved to: ./logs_with_date_partitions/date=YYYY-MM-DD/")
    
    # Example 2: No partitioning - files saved directly to log_dir
    print("\n2. No partitioning:")
    print("-" * 40)
    
    logger_no_partition = ParquetLogger(
        log_dir="./logs_no_partitions",
        buffer_size=1,  # Flush immediately for demo
        partition_on=None  # No partitioning
    )
    
    llm2 = FakeListChatModel(
        responses=["Response without partitioning"],
        callbacks=[logger_no_partition]
    )
    
    response = llm2.invoke("Test message", config=with_tags(custom_id="test-no-partition"))
    print(f"Response: {response.content}")
    print(f"Files will be saved directly to: ./logs_no_partitions/")
    
    # Example 3: Current directory without partitioning
    print("\n3. Current directory without partitioning:")
    print("-" * 40)
    
    logger_current_dir = ParquetLogger(
        log_dir=".",
        buffer_size=1,
        partition_on=None  # Save directly to current directory
    )
    
    llm3 = FakeListChatModel(
        responses=["Response in current directory"],
        callbacks=[logger_current_dir]
    )
    
    response = llm3.invoke("Test message", config=with_tags(custom_id="test-current-dir"))
    print(f"Response: {response.content}")
    print(f"Files will be saved directly to current directory")
    
    print("\n" + "=" * 60)
    print("Examples complete! Check the directories for parquet files.")
    print("=" * 60)


if __name__ == "__main__":
    main()