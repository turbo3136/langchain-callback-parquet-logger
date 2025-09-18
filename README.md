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

### 2. Batch Processing (Simple)

```python
import pandas as pd
from langchain_callback_parquet_logger import batch_process, with_tags

# Prepare your data
df = pd.DataFrame({
    'prompt': ['What is AI?', 'Explain DNA'],
    'config': [with_tags(custom_id='q1'), with_tags(custom_id='q2')]
})

# Process with automatic logging
results = await batch_process(df)
```

### 3. Batch Processing (Full Configuration)

```python
import pandas as pd
from langchain_callback_parquet_logger import (
    batch_process,
    with_tags,
    JobConfig,
    StorageConfig,
    ProcessingConfig,
    ColumnConfig,
    S3Config
)

# Prepare your data with custom column names
df = pd.DataFrame({
    'question': ['What is AI?', 'Explain DNA', 'What is quantum computing?'],
    'user_id': ['user1', 'user2', 'user3'],
    'tool_list': [[tool1, tool2], None, [tool3]]  # Optional tools
})

# Add config for each row (required)
df['run_config'] = df['user_id'].apply(lambda x: with_tags(
    custom_id=x,
    tags=['production', 'v2']
))

# Process with ALL configuration options
results = await batch_process(
    df,
    # LLM configuration
    llm_model='gpt-4',  # or pass existing LLM instance
    structured_output=None,  # or Pydantic model for structured responses

    # Job metadata configuration (all fields except category are optional)
    job_config=JobConfig(
        category="research",
        subcategory="science",  # Optional, defaults to None
        description="Analyzing scientific questions",  # Optional
        version="2.0.0",  # Optional
        environment="production",  # Optional
        metadata={"team": "data-science", "priority": "high"}  # Optional
    ),

    # Storage configuration
    storage_config=StorageConfig(
        output_dir="./batch_logs",
        path_template="{job_category}/{date}/{job_subcategory}",  # Custom path structure
        s3_config=S3Config(
            bucket="my-llm-logs",
            prefix="langchain-logs/",
            on_failure="continue",  # or "error" to fail on S3 errors
            retry_attempts=3
        )
    ),

    # Processing configuration
    processing_config=ProcessingConfig(
        max_concurrency=100,  # Parallel requests
        buffer_size=1000,  # Logger buffer size
        show_progress=True,  # Progress bar with real-time updates
        return_exceptions=True,  # Don't fail on single errors
        return_results=True,  # Set False for huge datasets to save memory
        event_types=['llm_start', 'llm_end', 'llm_error'],  # Events to log
        partition_on="date"  # Partition strategy
    ),

    # Column name configuration (if not using defaults)
    column_config=ColumnConfig(
        prompt="question",  # Your prompt column name
        config="run_config",  # Your config column name
        tools="tool_list"  # Your tools column name (optional)
    )
)

# Results are returned AND saved to Parquet files
df['answer'] = results
```

### 4. S3 Upload

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

### 5. Event Type Selection

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
- `subcategory`: Job subcategory (optional, default: None)
- `version`: Version string (optional, default: None)
- `environment`: Environment name (optional, default: None)
- `description`: Job description (optional, default: None)
- `metadata`: Additional metadata dict (optional, default: None)

### StorageConfig
- `output_dir`: Local directory (default: "./batch_logs")
- `path_template`: Path template for organizing files (default: "{job_category}/{job_subcategory}")
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
    },
    "raw": {
        // Complete dump of all callback arguments
        // Includes all kwargs plus positional args (serialized when possible)
        "response": {"generations": [...], "llm_output": {...}},
        "run_id": "uuid-here",
        "parent_run_id": "",
        // ... all other arguments passed to the callback
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