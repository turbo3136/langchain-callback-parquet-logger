# LangChain Callback Parquet Logger

A high-performance callback handler for logging LangChain LLM interactions to Parquet files. This package provides efficient, structured logging with automatic partitioning and buffering for production use.

## Features

- 📊 **Parquet Format**: Efficient columnar storage for analytics
- 🚀 **Buffered Writing**: Configurable buffer size for optimal performance
- 📅 **Daily Partitioning**: Automatic date-based file organization
- 🔄 **Thread-Safe**: Safe for concurrent LLM calls
- 📦 **Flexible Schema**: JSON payload for extensible logging
- 🔒 **Automatic Cleanup**: Ensures buffers flush on exit

## Installation

```bash
pip install langchain-callback-parquet-logger
```

## Quick Start

```python
from langchain_callback_parquet_logger import ParquetLogger
from langchain_openai import ChatOpenAI

# Just add the logger - works with ANY LangChain LLM
llm = ChatOpenAI(model="gpt-4")
llm.callbacks = [ParquetLogger("./logs")]

# Use the LLM normally
response = llm.invoke("What is 2+2?")
```

## Configuration

### Parameters

- `log_dir` (str, default: "./llm_logs"): Directory for log files
- `buffer_size` (int, default: 100): Number of entries before auto-flush
- `provider` (str, default: "openai"): LLM provider name for tracking

### Log Structure

Logs are saved as Parquet files with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | timestamp[us, tz=UTC] | Event timestamp |
| run_id | string | Unique run identifier |
| event_type | string | Event type (llm_start, llm_end, llm_error) |
| provider | string | LLM provider name |
| payload | string | JSON-encoded event data |

### File Organization

```
llm_logs/
├── date=2024-01-15/
│   ├── logs_143022_123456.parquet
│   └── logs_150331_789012.parquet
└── date=2024-01-16/
    └── logs_090122_345678.parquet
```

## Reading Logs

### With Pandas

```python
import pandas as pd
import json

# Read all parquet files in the log directory
df = pd.read_parquet("./logs")

# Parse JSON payloads
df['data'] = df['payload'].apply(json.loads)

# Analyze completions
completions = df[df['event_type'] == 'llm_end']
for _, row in completions.iterrows():
    data = row['data']
    if 'usage' in data:
        print(f"Run {row['run_id'][:8]}: {data['usage'].get('total_tokens', 0)} tokens")

# Get error rate
error_rate = len(df[df['event_type'] == 'llm_error']) / len(df) * 100
print(f"Error rate: {error_rate:.2f}%")
```

### With DuckDB

```python
import duckdb
import json

# Connect to DuckDB and query parquet files directly
conn = duckdb.connect()

# Read all logs
df = conn.execute("""
    SELECT * FROM read_parquet('./logs/**/*.parquet')
    ORDER BY timestamp DESC
""").df()

# Analyze by provider and event type
summary = conn.execute("""
    SELECT 
        provider,
        event_type,
        COUNT(*) as count,
        DATE(timestamp) as date
    FROM read_parquet('./logs/**/*.parquet')
    GROUP BY provider, event_type, DATE(timestamp)
    ORDER BY date DESC, provider
""").df()

print(summary)

# Extract specific fields from JSON payload
detailed = conn.execute("""
    SELECT 
        timestamp,
        run_id,
        json_extract_string(payload, '$.model') as model,
        json_extract_string(payload, '$.usage.total_tokens') as tokens
    FROM read_parquet('./logs/**/*.parquet')
    WHERE event_type = 'llm_end'
""").df()

print(f"Total tokens used: {detailed['tokens'].astype(float).sum()}")
```

### With PyArrow

```python
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import json

# Read using dataset API for better performance with partitioned data
dataset = ds.dataset("./logs", format="parquet", partitioning="hive")

# Convert to table with filters
table = dataset.to_table(filter=(ds.field("event_type") == "llm_end"))
df = table.to_pandas()

# Parse and analyze
for _, row in df.iterrows():
    payload = json.loads(row['payload'])
    print(f"Model: {payload.get('model_name', 'unknown')}")
    print(f"Tokens: {payload.get('usage', {}).get('total_tokens', 0)}")
```

## Context Manager Usage

```python
with ParquetLogger(log_dir="./logs") as logger:
    llm = ChatOpenAI(callbacks=[logger])
    llm.invoke("Process this message")
# Buffer automatically flushed on exit
```

## Advanced Usage

### Custom Buffer Size for Batch Processing

```python
# Large buffer for batch processing
logger = ParquetLogger(
    log_dir="./batch_logs",
    buffer_size=1000,  # Flush every 1000 entries
    provider="anthropic"
)
```

### Multiple Providers

```python
# Track different providers separately
openai_logger = ParquetLogger(log_dir="./logs", provider="openai")
anthropic_logger = ParquetLogger(log_dir="./logs", provider="anthropic")

openai_llm = ChatOpenAI(callbacks=[openai_logger])
anthropic_llm = ChatAnthropic(callbacks=[anthropic_logger])
```

## Examples

Check out the `examples/` directory for complete working examples:

- [`basic_usage.py`](examples/basic_usage.py) - Simple example showing fundamental logging capabilities
- [`batch_processing.py`](examples/batch_processing.py) - Advanced example with async batch processing, web search, and structured outputs

Run examples:
```bash
# Basic usage
python examples/basic_usage.py

# Batch processing with web search
python examples/batch_processing.py
```

## Performance Considerations

- **Buffer Size**: Larger buffers reduce I/O but use more memory
- **Compression**: Uses Snappy compression by default for balance of speed/size
- **Partitioning**: Daily partitions enable efficient querying and cleanup
- **Thread Safety**: Safe for concurrent use without performance penalty

## Development

### Install from source

```bash
git clone https://github.com/turbo3136/langchain-callback-parquet-logger.git
cd langchain-callback-parquet-logger
pip install -e .
```

### Running Tests

```bash
# No tests available yet
# Contributions welcome!
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please use the [GitHub issues page](https://github.com/turbo3136/langchain-callback-parquet-logger/issues).