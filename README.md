# LangChain Callback Parquet Logger

A high-performance callback handler for logging LangChain interactions to Parquet files with standardized payload structure.

## Features

- 📊 **Parquet Format**: Efficient columnar storage for analytics
- 🎯 **Standardized Structure**: Consistent payload format across all event types (v1.0.0+)
- 🚀 **Buffered Writing**: Configurable buffer size for optimal performance
- 📅 **Auto-Partitioning**: Daily partitioning for better data organization
- 🏷️ **Custom Tracking**: Add custom IDs and metadata to your logs
- 🔄 **Batch Processing**: Process DataFrames through LLMs efficiently
- ☁️ **S3 Upload**: Optional S3 upload for cloud storage
- 🔍 **Complete Event Support**: LLM, Chain, Tool, and Agent events

## Installation

```bash
pip install langchain-callback-parquet-logger
```

With optional features:
```bash
# S3 support
pip install "langchain-callback-parquet-logger[s3]"

# Background retrieval support
pip install "langchain-callback-parquet-logger[background]"
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
        "service": "api-gateway"
    }
)

# Request-level tracking
llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
response = llm.invoke(
    "What is quantum computing?",
    config=with_tags(custom_id="user-123-req-456")
)
```

### 3. Event Type Selection

```python
# Log all event types (v1.0.0+)
logger = ParquetLogger(
    './logs',
    event_types=['llm_start', 'llm_end', 'llm_error',
                 'chain_start', 'chain_end', 'chain_error',
                 'tool_start', 'tool_end', 'tool_error',
                 'agent_action', 'agent_finish']
)

# Default: Only LLM events for backward compatibility
logger = ParquetLogger('./logs')  # Only llm_start, llm_end, llm_error
```

### 4. Batch Processing

Process DataFrames through LLMs efficiently:

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

# Run batch processing
with ParquetLogger('./logs') as logger:
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[logger])
    results = await batch_run(df, llm, max_concurrency=10)
    df['answer'] = results
```

### 5. S3 Upload

For cloud storage and ephemeral environments:

```python
logger = ParquetLogger(
    log_dir="./logs",
    s3_bucket="my-llm-logs",
    s3_prefix="runs/",
    s3_on_failure="error"  # Fail fast for production
)
```

## Standardized Payload Structure (v1.0.0+)

All events now use a consistent structure for easier processing:

```python
{
    "event_type": "llm_start",
    "event_phase": "start",      # start/end/error/action/finish
    "event_component": "llm",    # llm/chain/tool/agent
    "timestamp": "2024-01-15T10:30:00Z",
    
    "execution": {
        "run_id": "uuid-string",
        "parent_run_id": "",      # Empty string if no parent
        "tags": [],
        "metadata": {},
        "custom_id": ""
    },
    
    "data": {
        "inputs": {               # All input data
            "prompts": [],        # LLM prompts
            "messages": [],       # Chat messages
            "inputs": {},         # Chain/tool inputs
            "input_str": "",      # Tool input string
            "action": {},         # Agent action
            "serialized": {}      # Serialized component
        },
        "outputs": {              # All output data
            "response": {},       # LLM response
            "outputs": {},        # Chain outputs
            "output": "",         # Tool output
            "finish": {},         # Agent finish
            "usage": {}           # Token usage
        },
        "error": {                # Error information
            "message": "",
            "type": "",
            "details": {},
            "traceback": []
        },
        "config": {               # Configuration
            "invocation_params": {},
            "model": "",
            "tools": [],
            "response_metadata": {}
        }
    },
    
    "raw": {                      # Complete raw data
        "kwargs": {},             # Full kwargs dict
        "primary_args": {}        # Main positional args
    }
}
```

## Reading Logs

### Basic Reading
```python
import pandas as pd
import json

# Read all logs
df = pd.read_parquet("./logs")

# Parse standardized payload (v1.0.0+)
for idx, row in df.iterrows():
    payload = json.loads(row['payload'])
    
    # Access standardized fields
    event_type = payload['event_type']
    prompts = payload['data']['inputs']['prompts']
    response = payload['data']['outputs']['response']
    usage = payload['data']['outputs']['usage']
    error_msg = payload['data']['error']['message']
```

### Query with DuckDB
```python
import duckdb

conn = duckdb.connect()
df = conn.execute("""
    SELECT 
        logger_custom_id,
        event_type,
        timestamp,
        json_extract_string(payload, '$.data.outputs.usage.total_tokens') as tokens,
        json_extract_string(payload, '$.data.config.model') as model
    FROM read_parquet('./logs/**/*.parquet')
    WHERE event_type = 'llm_end'
    ORDER BY timestamp DESC
""").df()
```

## Configuration Options

### ParquetLogger Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | str | "./llm_logs" | Directory for log files |
| `buffer_size` | int | 100 | Entries before auto-flush |
| `provider` | str | "openai" | LLM provider name |
| `logger_metadata` | dict | {} | Metadata for all logs |
| `partition_on` | str/None | "date" | "date" or None |
| `event_types` | list | ['llm_start', 'llm_end', 'llm_error'] | Events to log |
| `s3_bucket` | str/None | None | S3 bucket name |
| `s3_prefix` | str | "langchain-logs/" | S3 prefix |
| `s3_on_failure` | str | "error" | "error" or "continue" |

## Log Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamp | Event time (UTC) |
| `run_id` | string | Unique run ID |
| `parent_run_id` | string | Parent run ID (hierarchy tracking) |
| `logger_custom_id` | string | Your custom tracking ID |
| `event_type` | string | Event type |
| `provider` | string | LLM provider |
| `logger_metadata` | string | JSON metadata |
| `payload` | string | Standardized JSON payload (v1.0.0+) |

## Breaking Changes in v1.0.0

The payload structure has been completely standardized. If upgrading from earlier versions:

**Old structure (pre-v1.0.0):**
```python
payload = json.loads(row['payload'])
prompts = payload.get('prompts', [])  # Direct access, inconsistent
```

**New structure (v1.0.0+):**
```python
payload = json.loads(row['payload'])
prompts = payload['data']['inputs']['prompts']  # Nested, consistent
```

All fields now have non-null defaults, making processing more predictable.

## Examples

- [`basic_usage.py`](examples/basic_usage.py) - Simple logging
- [`batch_processing.py`](examples/batch_processing.py) - Batch operations
- [`memory_efficient_batch.py`](examples/memory_efficient_batch.py) - Large DataFrame processing
- [`retrieve_background_responses.py`](examples/retrieve_background_responses.py) - Background retrieval

## License

MIT License

## Contributing

Contributions welcome! Please submit a Pull Request.

## Support

For issues and questions, use [GitHub issues](https://github.com/turbo3136/langchain-callback-parquet-logger/issues).