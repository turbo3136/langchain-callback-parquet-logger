# LangChain Callback Parquet Logger

A high-performance callback handler for logging LangChain LLM interactions to Parquet files with automatic partitioning, buffering, and batch processing support.

## Features

- 📊 **Parquet Format**: Efficient columnar storage for analytics
- 🚀 **Buffered Writing**: Configurable buffer size for optimal performance
- 📅 **Partitioning**: Optional daily partitioning for better organization
- 🏷️ **Custom Tracking**: Add custom IDs and metadata to your logs
- 🔄 **Batch Processing**: Simple helper for DataFrame batch operations
- 🔒 **Thread-Safe**: Safe for concurrent LLM calls

## Installation

```bash
pip install langchain-callback-parquet-logger
```

## Quick Start

```python
from langchain_callback_parquet_logger import ParquetLogger
from langchain_openai import ChatOpenAI

# Simple usage
llm = ChatOpenAI(model="gpt-4o-mini")
llm.callbacks = [ParquetLogger("./logs")]

response = llm.invoke("What is 2+2?")
```

## Core Features

### 1. Basic Logging

```python
# With context manager (recommended for notebooks)
with ParquetLogger('./logs') as logger:
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
    response = llm.invoke("Hello!")
# Logs automatically flushed on exit
```

### 2. Custom IDs and Metadata

```python
from langchain_callback_parquet_logger import ParquetLogger, with_tags

# Logger-level metadata (included in all logs)
logger = ParquetLogger(
    log_dir="./logs",
    logger_metadata={
        "environment": "production",
        "service": "api-gateway",
        "version": "2.1.0"
    }
)

# Request-level tracking with custom IDs
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
response = llm.invoke(
    "What is quantum computing?",
    config=with_tags(
        custom_id="user-123-req-456",
        tags=["production", "high-priority"]
    )
)
```

### 3. Batch Processing (v0.3.0+)

Process DataFrames through LLMs with minimal code:

```python
import pandas as pd
from langchain_callback_parquet_logger import batch_run, with_tags, ParquetLogger

# Prepare your data
df = pd.DataFrame({
    'id': ['001', '002', '003'],
    'question': ['What is AI?', 'Explain quantum computing', 'What is blockchain?']
})

# Add required columns
df['prompt'] = df['question']
df['config'] = df['id'].apply(lambda x: with_tags(custom_id=x))

# Configure LLM with advanced features
with ParquetLogger('./logs') as logger:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        service_tier="flex",  # Optional: optimize costs
        model_kwargs={"background": True},  # Optional: background processing
        callbacks=[logger]
    )
    
    # Run batch processing
    results = await batch_run(df, llm, max_concurrency=10, show_progress=True)
    df['answer'] = results
```

See [examples/batch_processing.py](examples/batch_processing.py) for advanced usage with structured outputs, web search tools, and more.

#### Memory-Efficient Mode for Huge DataFrames

For massive DataFrames, use `return_results=False` to avoid keeping results in memory:

```python
# Process huge DataFrame without memory overhead
with ParquetLogger('./logs') as logger:
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
    
    # Results saved to parquet only, not kept in memory
    await batch_run(huge_df, llm, return_results=False)
    
# Read results later from parquet files
df_logs = pd.read_parquet('./logs')
results = df_logs[df_logs['event_type'] == 'llm_end']
```

## Configuration Options

### ParquetLogger Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | str | "./llm_logs" | Directory for log files |
| `buffer_size` | int | 100 | Entries before auto-flush |
| `provider` | str | "openai" | LLM provider name |
| `logger_metadata` | dict | {} | Metadata for all logs |
| `partition_on` | str/None | "date" | "date" or None for no partitioning |

### batch_run Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | DataFrame with data |
| `llm` | LangChain LLM | required | Configured LLM instance |
| `prompt_col` | str | "prompt" | Column with prompts |
| `config_col` | str | "config" | Column with config dicts |
| `tools_col` | str/None | "tools" | Column with tools lists |
| `max_concurrency` | int | 10 | Max parallel requests |
| `show_progress` | bool | True | Show progress bar |
| `return_results` | bool | True | If False, don't keep results in memory |

## Reading Logs

### With Pandas
```python
import pandas as pd
import json

df = pd.read_parquet("./logs")
df['data'] = df['payload'].apply(json.loads)

# Analyze by custom ID
custom_requests = df[df['logger_custom_id'] != '']
print(f"Found {len(custom_requests)} tagged requests")
```

### With DuckDB
```python
import duckdb

conn = duckdb.connect()
df = conn.execute("""
    SELECT 
        logger_custom_id,
        event_type,
        timestamp,
        json_extract_string(payload, '$.usage.total_tokens') as tokens
    FROM read_parquet('./logs/**/*.parquet')
    WHERE logger_custom_id != ''
    ORDER BY timestamp DESC
""").df()
```

## Log Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamp | Event time (UTC) |
| `run_id` | string | Unique run ID |
| `logger_custom_id` | string | Your custom ID |
| `event_type` | string | llm_start/end/error |
| `provider` | string | LLM provider |
| `logger_metadata` | string | JSON metadata |
| `payload` | string | JSON event data |

## Important Notes

### Notebook Usage
In Jupyter/Colab, use one of these approaches for immediate writes:
- **Context manager** (recommended): `with ParquetLogger() as logger:`
- **Small buffer**: `ParquetLogger(buffer_size=1)`
- **Manual flush**: `logger.flush()`

### File Organization
```
logs/
├── date=2024-01-15/          # With partitioning (default)
│   └── logs_143022_123456.parquet
└── date=2024-01-16/
    └── logs_090122_345678.parquet
```

## Background Response Retrieval (v0.4.0+)

Retrieve completed responses from OpenAI's background/async requests:

```python
import pandas as pd
import openai
from langchain_callback_parquet_logger import retrieve_background_responses, ParquetLogger

# DataFrame with response IDs from background requests
df = pd.DataFrame({
    'response_id': ['resp_123...', 'resp_456...'],
    'logger_custom_id': ['user-001', 'user-002']
})

# Retrieve and log responses
client = openai.AsyncClient()
with ParquetLogger('./retrieval_logs') as logger:
    results = await retrieve_background_responses(
        df,
        client,
        logger=logger,
        show_progress=True,
        checkpoint_file='./checkpoint.parquet'  # Resume capability
    )
```

### Features
- **Automatic rate limiting** with exponential backoff
- **Checkpoint/resume** for interrupted retrievals
- **Memory-efficient mode** with `return_results=False`
- **Progress tracking** with tqdm
- **Structured logging** of attempts, completions, and errors

See [examples/retrieve_background_responses.py](examples/retrieve_background_responses.py) for detailed usage.

## Examples

- [`basic_usage.py`](examples/basic_usage.py) - Simple logging example
- [`batch_processing.py`](examples/batch_processing.py) - Advanced batch processing with all features
- [`simple_batch_example.py`](examples/simple_batch_example.py) - Before/after batch processing comparison
- [`memory_efficient_batch.py`](examples/memory_efficient_batch.py) - Memory-efficient processing for huge DataFrames
- [`partitioning_example.py`](examples/partitioning_example.py) - Partitioning strategies
- [`retrieve_background_responses.py`](examples/retrieve_background_responses.py) - Background response retrieval

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please submit a Pull Request.

## Support

For issues and questions, use [GitHub issues](https://github.com/turbo3136/langchain-callback-parquet-logger/issues).