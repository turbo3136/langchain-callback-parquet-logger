# CLAUDE.md

This file provides guidance to AI agents when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_background_retrieval.py -v

# Run specific test
python -m pytest tests/test_core.py::test_basic_logging -v

# Run with coverage
python -m pytest tests/ --cov=langchain_callback_parquet_logger
```

### Installation
```bash
# Install for development (includes test dependencies)
pip install -e ".[test]"

# Install with background retrieval support
pip install -e ".[background]"

# Install with S3 support
pip install -e ".[s3]"

# Install all optional dependencies
pip install -e ".[test,background,s3]"
```

### Package Building
```bash
# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

## Architecture Overview

This package provides a high-performance callback handler for logging LangChain LLM interactions to Parquet files with optional S3 upload support. The architecture consists of five main components:

### 1. Core Logging System (`logger.py`)
- **ParquetLogger**: Main callback handler that intercepts LangChain events
- **Enhanced Event Support (v0.5.0+)**:
  - LLM events: on_llm_start, on_llm_end, on_llm_error
  - Chain events: on_chain_start, on_chain_end, on_chain_error
  - Tool events: on_tool_start, on_tool_end, on_tool_error
  - Agent events: on_agent_action, on_agent_finish
  - Configurable via `event_types` parameter (defaults to LLM events only for backward compatibility)
- **Schema**: 7 columns with hierarchy support
  - timestamp, run_id, parent_run_id, logger_custom_id, event_type, logger_metadata, payload
  - `parent_run_id` enables tracking of execution hierarchy (chains → LLMs → tools)
  - LLM type is captured in the payload's `llm_type` field for llm_start events
- Buffers entries in memory (configurable size) before flushing to Parquet files
- Thread-safe with lock-based synchronization
- Supports daily partitioning (date=YYYY-MM-DD) or flat structure
- All event data stored as JSON strings in the payload column for flexibility
- **Raw Data Capture (v2.0+)**: Complete callback data captured in `raw` section of payload
  - All positional arguments and kwargs are serialized and stored
  - Uses `_serialize_any()` helper to try all serialization methods (model_dump, dict, to_dict, __dict__)
  - Ensures no information is lost while keeping `data` section clean and structured
- **S3 Support (v0.6.0+)**: Optional upload to S3 with retry logic and configurable failure handling

### 2. Batch Processing (`batch_helpers.py`)
- **batch_run()**: Async function for processing DataFrames of prompts through LLMs
- Expects DataFrame with columns: prompt, config (from with_tags()), optional tools
- Uses LangChain's RunnableLambda for async batching
- Memory-efficient mode (return_results=False) for huge DataFrames
- Progress tracking with tqdm (auto-detects notebook vs terminal)

### 3. Background Response Retrieval (`background_retrieval.py`)
- **retrieve_background_responses()**: Retrieves completed responses from OpenAI's background/async API
- Expects DataFrame with response_id and logger_custom_id columns
- Implements exponential backoff for rate limiting (429 errors)
- Checkpoint/resume capability via Parquet files
- Logs three event types: background_retrieval_attempt, background_retrieval_complete, background_retrieval_error
- Reuses ParquetLogger's _add_entry() method for consistent logging

### 4. S3 Integration
- **Optional Feature**: S3 upload only activates when `s3_bucket` parameter is provided
- **Failure Modes**: `error` mode for ephemeral environments (Hex.tech), `continue` mode for development
- **Retry Logic**: Exponential backoff with configurable attempts
- **Credential Chain**: Uses boto3's standard AWS credential resolution

### 5. Enhanced Batch Processing (`batch_process()` in `batch_helpers.py`)
- **Automated Batch Processing**: High-level function that combines batch processing with automatic logging
- **Storage Flexibility**: Supports local-only or local + S3 storage modes
- **Job Metadata**: Automatic organization with job categories, subcategories, versions, and environments
- **Path Templates**: Flexible path formatting with template variables
- **LLM Auto-Configuration**: Automatic LLM creation
- **Structured Output**: Built-in support for Pydantic models
- **Full Parameter Control**: Exposes all ParquetLogger and batch_run parameters
- **Override Support**: Escape hatches for advanced customization

## Key Design Patterns

### Tagging System
The `with_tags()` function in `__init__.py` creates config dictionaries with custom IDs and tags. Custom IDs are prefixed with "logger_custom_id:" and stored in the tags array, which ParquetLogger extracts during event handling.

### Buffer Management
ParquetLogger accumulates entries in memory until buffer_size is reached, then writes to Parquet. Flushing happens:
- When buffer reaches capacity
- On context manager exit (`__exit__`)
- Via explicit flush() call
- Via atexit handler
- In destructor as last resort

### Optional Dependencies
Package uses try/except imports to make features optional:
- batch_helpers requires pandas (always available if using batch features)
- background_retrieval requires openai, pandas, and tqdm
- S3 support requires boto3
- Import failures are silent, features simply won't be available

## Working with Parquet Files

The package writes Parquet files with the following structure:
- Default: `./logs/date=YYYY-MM-DD/logs_HHMMSS_microseconds.parquet`
- No partitioning: `./logs/logs_HHMMSS_microseconds.parquet`

### Basic Usage
```python
import pandas as pd
import json

df = pd.read_parquet("./logs")
# Parse JSON payload
df['data'] = df['payload'].apply(json.loads)
```

### Enhanced Batch Processing (v1.1.0+)
```python
from langchain_callback_parquet_logger import batch_process
import pandas as pd

# Prepare DataFrame
df = pd.DataFrame({
    'prompt': ['What is AI?', 'Explain quantum computing'],
    'config': [with_tags(custom_id='q1'), with_tags(custom_id='q2')]
})

# Local-only processing
await batch_process(
    df,
    job_category="research",
    job_subcategory="questions",
    output_dir="./batch_logs"
)

# With S3 upload
await batch_process(
    df,
    job_category="production",
    s3_bucket="my-data-lake",
    s3_prefix_template="ml/{job_category}/{date}/",
    max_concurrency=1000
)

# With structured output
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float

await batch_process(
    df,
    structured_output=Answer,
    job_category="structured_qa",
    event_types=['llm_start', 'llm_end', 'chain_start']
)
```

### Enhanced Event Logging (v0.5.0+)
```python
from langchain_callback_parquet_logger import ParquetLogger

# Log all event types including chains, tools, and agents
logger = ParquetLogger(
    './logs',
    event_types=['llm_start', 'llm_end', 'llm_error',
                 'chain_start', 'chain_end', 'chain_error',
                 'tool_start', 'tool_end', 'tool_error',
                 'agent_action', 'agent_finish']
)

# Use with LangChain components
chain = SomeChain(callbacks=[logger])
chain.run("input")
```

### Analyzing Execution Hierarchy
```python
# Read logs with hierarchy information
df = pd.read_parquet("./logs")

# Find all events related to a specific chain execution
chain_run_id = "abc-123"
chain_events = df[
    (df['run_id'] == chain_run_id) | 
    (df['parent_run_id'] == chain_run_id)
]

# Build execution tree
def get_children(df, parent_id):
    return df[df['parent_run_id'] == parent_id]

# Trace complete execution flow
root_events = df[df['parent_run_id'] == '']
for root in root_events.itertuples():
    print(f"Root: {root.event_type} ({root.run_id})")
    children = get_children(df, root.run_id)
    for child in children.itertuples():
        print(f"  └─ {child.event_type} ({child.run_id})")
```

## Version Management

Version is defined in two places that must be kept in sync:
- `langchain_callback_parquet_logger/__init__.py`: `__version__ = "X.Y.Z"`
- `pyproject.toml`: `version = "X.Y.Z"`

## Dependencies

Core dependencies (always required):
- pyarrow>=10.0.0
- langchain-core>=0.1.0

Optional dependency groups:
- `[test]`: pytest, pytest-asyncio, pytest-mock, pandas
- `[background]`: openai, pandas, tqdm
- `[s3]`: boto3