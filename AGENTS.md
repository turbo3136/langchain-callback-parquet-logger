# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Install all optional dependencies
pip install -e ".[test,background]"
```

### Package Building
```bash
# Build distribution
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

## Architecture Overview

This package provides a high-performance callback handler for logging LangChain LLM interactions to Parquet files. The architecture consists of three main components:

### 1. Core Logging System (`logger.py`)
- **ParquetLogger**: Main callback handler that intercepts LangChain events (on_llm_start, on_llm_end, on_llm_error)
- Uses a 7-column schema: timestamp, run_id, logger_custom_id, event_type, provider, logger_metadata, payload
- Buffers entries in memory (configurable size) before flushing to Parquet files
- Thread-safe with lock-based synchronization
- Supports daily partitioning (date=YYYY-MM-DD) or flat structure
- All event data stored as JSON strings in the payload column for flexibility

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
- Import failures are silent, features simply won't be available

## Working with Parquet Files

The package writes Parquet files with the following structure:
- Default: `./logs/date=YYYY-MM-DD/logs_HHMMSS_microseconds.parquet`
- No partitioning: `./logs/logs_HHMMSS_microseconds.parquet`

Reading logs:
```python
import pandas as pd
import json

df = pd.read_parquet("./logs")
# Parse JSON payload
df['data'] = df['payload'].apply(json.loads)
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