# LangChain Parquet Logger

High-performance logging for LangChain - save all your LLM interactions to Parquet files for analysis.

## Quick Start (2 minutes)

### Install
```bash
pip install langchain-callback-parquet-logger

# With S3 support
pip install "langchain-callback-parquet-logger[s3]"
```

### Basic Usage
```python
from langchain_callback_parquet_logger import ParquetLogger
from langchain_openai import ChatOpenAI

# Add logger to any LangChain LLM
logger = ParquetLogger("./logs")
llm = ChatOpenAI(callbacks=[logger])

response = llm.invoke("What is 2+2?")
# Your logs are automatically saved to ./logs/
```

### Batch Processing
```python
import pandas as pd
from langchain_callback_parquet_logger import batch_process

# Your data
df = pd.DataFrame({
    'prompt': ['What is AI?', 'Explain quantum computing']
})

# Process it (logs automatically saved)
results = await batch_process(df)
```

That's it! Your logs are in Parquet format, ready for analysis.

## Core Features

### 1. Custom Tracking IDs

Track specific requests with custom IDs:

```python
from langchain_callback_parquet_logger import ParquetLogger, with_tags

logger = ParquetLogger("./logs")
llm = ChatOpenAI(callbacks=[logger])

# Add custom ID to track this specific request
response = llm.invoke(
    "What is quantum computing?",
    config=with_tags(custom_id="user-123-session-456")
)
```

### 2. Batch Processing (Complete Example)

```python
import pandas as pd
from langchain_callback_parquet_logger import (
    batch_process,
    with_tags,
    JobConfig,
    StorageConfig,
    S3Config
)

# Prepare your data
df = pd.DataFrame({
    'question': ['What is AI?', 'Explain DNA', 'What is quantum computing?'],
    'user_id': ['user1', 'user2', 'user3']
})

# Add required columns
df['prompt'] = df['question']  # Required column name
df['config'] = df['user_id'].apply(lambda x: with_tags(custom_id=x))

# Process with full configuration
results = await batch_process(
    df,
    job_config=JobConfig(
        category="research",
        subcategory="science",
        version="2.0.0"
    ),
    storage_config=StorageConfig(
        output_dir="./batch_logs",
        s3_config=S3Config(bucket="my-llm-logs")  # Optional S3 upload
    )
)

# Results are returned AND saved to Parquet files
df['answer'] = results
```

### 3. S3 Upload

For production and cloud environments:

```python
from langchain_callback_parquet_logger import ParquetLogger, S3Config

logger = ParquetLogger(
    log_dir="./logs",
    s3_config=S3Config(
        bucket="my-llm-logs",
        prefix="production/",
        on_failure="error"  # Fail fast in production
    )
)
```

### 4. Event Type Selection

Choose what events to log:

```python
# Default: Only LLM events
logger = ParquetLogger("./logs")

# Log everything
logger = ParquetLogger(
    "./logs",
    event_types=['llm_start', 'llm_end', 'llm_error',
                 'chain_start', 'chain_end', 'chain_error',
                 'tool_start', 'tool_end', 'tool_error']
)
```

## Reading Your Logs

```python
import pandas as pd
import json

# Read all logs
df = pd.read_parquet("./logs")

# Parse the payload
df['data'] = df['payload'].apply(json.loads)

# Analyze token usage
df['tokens'] = df['data'].apply(lambda x: x.get('data', {}).get('outputs', {}).get('usage', {}).get('total_tokens'))
```

## v2.0 Breaking Changes

If upgrading from v1.x:

### Old (v1.x)
```python
logger = ParquetLogger(
    log_dir="./logs",
    s3_bucket="my-bucket",
    s3_prefix="logs/",
    s3_on_failure="error"
)
```

### New (v2.0)
```python
from langchain_callback_parquet_logger import ParquetLogger, S3Config

logger = ParquetLogger(
    log_dir="./logs",
    s3_config=S3Config(
        bucket="my-bucket",
        prefix="logs/",
        on_failure="error"
    )
)
```

### batch_process changes:
- Now uses dataclass configs instead of 34 parameters
- Much simpler and cleaner API
- See batch processing example above

## Configuration Classes

### ParquetLogger
- `log_dir`: Where to save logs (default: "./llm_logs")
- `buffer_size`: Entries before auto-flush (default: 100)
- `s3_config`: Optional S3Config for uploads

### JobConfig
- `category`: Job category (default: "batch_processing")
- `subcategory`: Job subcategory (default: "default")
- `version`: Version string (default: "1.0.0")
- `environment`: Environment name (default: "production")

### StorageConfig
- `output_dir`: Local directory (default: "./batch_logs")
- `s3_config`: Optional S3Config for uploads

### S3Config
- `bucket`: S3 bucket name
- `prefix`: S3 prefix/folder (default: "langchain-logs/")
- `on_failure`: "error" or "continue" (default: "error")

## Advanced Usage

### Low-Level Batch Processing

If you need direct control over logging:

```python
from langchain_callback_parquet_logger import batch_run, ParquetLogger

# Setup your own logging
with ParquetLogger('./logs') as logger:
    llm = ChatOpenAI(callbacks=[logger])

    # Use low-level batch_run
    results = await batch_run(df, llm, max_concurrency=100)
```

### Context Manager (Notebooks)

For Jupyter notebooks, use context manager for immediate writes:

```python
with ParquetLogger('./logs', buffer_size=1) as logger:
    llm = ChatOpenAI(callbacks=[logger])
    response = llm.invoke("Hello!")
# Logs are guaranteed to be written
```

## Log Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | timestamp | Event time (UTC) |
| `run_id` | string | Unique run ID |
| `parent_run_id` | string | Parent run ID for nested calls |
| `logger_custom_id` | string | Your custom tracking ID |
| `event_type` | string | Event type (llm_start, llm_end, etc.) |
| `logger_metadata` | string | JSON metadata |
| `payload` | string | Full event data as JSON |

## Payload Structure

All events use a consistent JSON structure in the payload column:

```json
{
    "event_type": "llm_end",
    "timestamp": "2024-01-15T10:30:00Z",
    "execution": {
        "run_id": "uuid-here",
        "parent_run_id": "",
        "custom_id": "user-123"
    },
    "data": {
        "prompts": ["..."],
        "llm_type": "openai-chat",  // LangChain's native LLM type
        "response": {"content": "..."},
        "usage": {"total_tokens": 100}
    }
}
```

## Installation Options

```bash
# Basic
pip install langchain-callback-parquet-logger

# With S3 support
pip install "langchain-callback-parquet-logger[s3]"

# With background retrieval support (OpenAI)
pip install "langchain-callback-parquet-logger[background]"

# Everything
pip install "langchain-callback-parquet-logger[s3,background]"
```

## License

MIT

## Contributing

Pull requests welcome! Keep it simple.

## Support

[GitHub Issues](https://github.com/turbo3136/langchain-callback-parquet-logger/issues)