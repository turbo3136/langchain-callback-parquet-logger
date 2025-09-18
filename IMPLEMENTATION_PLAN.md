# Implementation Plan: LLMConfig and Enhanced Metadata Tracking

## Status: ✅ COMPLETED

All tasks have been successfully implemented and tested.

## Executive Summary

This plan outlines improvements to the langchain-callback-parquet-logger package v2.0.0 to:
1. Simplify and standardize LLM configuration through a new `LLMConfig` dataclass
2. Enhance metadata tracking to capture all batch and row-level configurations
3. Add custom ID descriptions for better context via the `with_tags` function
4. Rename `logger_custom_id` to `custom_id` in the schema for clarity

The changes focus on simplicity and complete observability while maintaining the existing two-level architecture (batch-level vs row-level).

## Architecture Overview

### Two-Level Separation
- **Batch Level**: Configuration for the entire job (storage, processing, job metadata, LLM config)
- **Row Level**: Per-row inputs (prompts, tools, configs) - already tracked in event payloads

### Key Principles
- No backward compatibility concerns (v2.0 already has breaking changes)
- Simplicity in both code and developer experience
- Complete tracking of all inputs and configurations
- Clear separation of concerns

## Detailed Implementation

### Understanding llm_kwargs vs model_kwargs

The distinction between these two parameter sets is important:

- **llm_kwargs**: Parameters for the LangChain LLM wrapper class constructor
  - These configure how LangChain interacts with the model API
  - Examples: `model='gpt-4'`, `api_key='...'`, `temperature=0.7`, `max_tokens=1000`, `timeout=30`
  - These are the arguments you'd pass directly when creating `ChatOpenAI(model='gpt-4', temperature=0.7)`

- **model_kwargs**: Additional parameters passed through to the underlying model API
  - These are model-specific parameters that get passed in the API request
  - Examples: `top_p=0.9`, `frequency_penalty=0.5`, `presence_penalty=0.5`, `logit_bias={...}`
  - LangChain passes these through via its `model_kwargs` parameter

Most users only need `llm_kwargs`. Use `model_kwargs` when you need fine-grained control over model-specific parameters not exposed as direct LLM constructor arguments.

### 1. Add LLMConfig Dataclass (`config.py`)

```python
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Type

@dataclass
class LLMConfig:
    """LLM configuration for batch processing.

    Understanding the kwargs distinction:
    - llm_kwargs: Arguments passed directly to the LLM class constructor
      Examples: model='gpt-4', temperature=0.7, api_key='...', max_tokens=1000
    - model_kwargs: Additional parameters passed to the underlying model API
      These get passed through to the model_kwargs parameter that most LangChain
      LLMs support. Examples: top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5

    Example usage:
        config = LLMConfig(
            llm_class=ChatOpenAI,
            llm_kwargs={'model': 'gpt-4', 'temperature': 0.7},  # OpenAI client args
            model_kwargs={'top_p': 0.9}  # Additional model parameters
        )
    """
    llm_class: Type  # The LangChain LLM class to instantiate (e.g., ChatOpenAI)
    llm_kwargs: Optional[Dict[str, Any]] = None  # Constructor arguments for the LLM class
    model_kwargs: Optional[Dict[str, Any]] = None  # Additional model parameters
    structured_output: Optional[Type] = None  # Optional Pydantic model for structured output

    def create_llm(self) -> Any:
        """Create the LLM instance from config.

        This combines llm_kwargs and model_kwargs appropriately:
        - llm_kwargs are passed directly to the LLM constructor
        - model_kwargs are passed as the 'model_kwargs' parameter
        """
        kwargs = (self.llm_kwargs or {}).copy()
        if self.model_kwargs:
            # Most LangChain LLMs accept a model_kwargs parameter
            # for additional model-specific parameters
            kwargs['model_kwargs'] = self.model_kwargs

        llm = self.llm_class(**kwargs)

        if self.structured_output:
            llm = llm.with_structured_output(self.structured_output)

        return llm

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata tracking.

        This ensures all configuration is logged for observability.
        """
        return {
            'class': self.llm_class.__name__,
            'module': self.llm_class.__module__,
            'llm_kwargs': self.llm_kwargs or {},
            'model_kwargs': self.model_kwargs or {},
            'structured_output': (
                self.structured_output.__name__
                if self.structured_output else None
            )
        }
```

**Rationale**:
- Factory pattern keeps LLM creation logic centralized
- `to_metadata_dict()` ensures clean serialization for logging
- Clear distinction between llm_kwargs (constructor args) and model_kwargs (API params)
- Matches existing config dataclass patterns

### 2. Update with_tags Function (`tagging.py`)

```python
def with_tags(
    custom_id: Optional[str] = None,
    custom_id_description: Optional[str] = None,  # NEW parameter
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a config dict with custom ID and optional description.

    Args:
        custom_id: Custom identifier for tracking this specific request
        custom_id_description: Human-readable description of what the custom_id represents
                               (e.g., "Customer support ticket ID", "Research question ID")
        tags: Additional tags to include
        metadata: Additional metadata dict to include

    Returns:
        Config dict for use with LangChain callbacks

    Example:
        config = with_tags(
            custom_id="ticket-12345",
            custom_id_description="Customer support ticket ID",
            tags=["production", "priority-high"]
        )
    """
    tags_list = tags or []

    if custom_id:
        # Add the custom ID with the standard prefix
        tags_list.append(f"{CUSTOM_ID_PREFIX}{custom_id}")

        # Add the description as a separate tag if provided
        if custom_id_description:
            tags_list.append(f"custom_id_description:{custom_id_description}")

    config = {"tags": tags_list}
    if metadata:
        config["metadata"] = metadata

    return config
```

**Note**: The custom_id_description is stored in the tags array and will be captured in the event payload under `payload.execution.tags`. This keeps all row-level context together and accessible for analysis.

### 3. Update Parquet Schema (`logger.py`)

```python
# Update schema definition (line ~22)
SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("run_id", pa.string()),
    ("parent_run_id", pa.string()),
    ("custom_id", pa.string()),  # RENAMED from logger_custom_id
    ("event_type", pa.string()),
    ("logger_metadata", pa.string()),
    ("payload", pa.string()),
])
```

**Update all references** (4 places in logger.py):
1. Line ~179: `'custom_id': payload["execution"]["custom_id"]`
2. Line ~369: `custom_id = pa.array([e["custom_id"] for e in buffer]`
3. Line ~378: Update array order to match new schema
4. Keep `CUSTOM_ID_PREFIX = "logger_custom_id:"` in config.py for backward compatibility with existing tags

### 4. Update batch_process Function (`batch.py`)

#### New Function Signature
```python
from dataclasses import asdict
from datetime import datetime, timezone

async def batch_process(
    df: pd.DataFrame,
    llm_config: LLMConfig,  # NEW: Single config object for LLM
    job_config: Optional[JobConfig] = None,
    storage_config: Optional[StorageConfig] = None,
    processing_config: Optional[ProcessingConfig] = None,
    column_config: Optional[ColumnConfig] = None,
) -> Optional[List]:
    """
    Batch process DataFrame through LLM with automatic logging to Parquet.

    Args:
        df: DataFrame with prepared data
        llm_config: LLM configuration including class, kwargs, and structured output
        job_config: Job metadata configuration
        storage_config: Storage configuration for output files
        processing_config: Processing configuration for batch operations
        column_config: DataFrame column name configuration

    Returns:
        List of results if processing_config.return_results=True, None otherwise
    """
```

#### LLM Initialization (Replace lines ~193-210)
```python
# Create LLM from config
llm = llm_config.create_llm()
```

#### Enhanced Metadata Structure (Replace lines ~237-248)
```python
# Build comprehensive logger metadata
logger_metadata = {
    # Legacy flat fields (for backward compatibility in queries)
    'job_category': job_config.category,
    'job_subcategory': job_config.subcategory,
    'environment': job_config.environment,
    'job_description': job_config.description,
    'job_version': job_config.version,

    # Complete batch-level configs (NEW structure)
    'batch_config': {
        'job': asdict(job_config) if job_config else None,
        'storage': {
            'output_dir': storage_config.output_dir,
            'path_template': storage_config.path_template,
            's3': asdict(storage_config.s3_config) if storage_config.s3_config else None
        },
        'processing': asdict(processing_config) if processing_config else None,
        'column': asdict(column_config) if column_config else None,
        'llm': llm_config.to_metadata_dict(),
    },

    # Batch execution metadata
    'batch_started_at': datetime.now(timezone.utc).isoformat(),
    'batch_size': len(df),

    # Custom metadata from job_config (if any)
    **(job_config.metadata or {})
}
```

**Note on Row-Level Data**: The config column data (including custom_id and custom_id_description) is automatically captured in each event's payload under `payload.execution.tags` and `payload.execution.metadata`. This maintains the separation between batch-level metadata (constant across all rows) and row-level data (varies per row).

### 5. Update Imports

#### In `batch.py`:
```python
from dataclasses import asdict  # Add this import
from datetime import datetime, timezone  # Add timezone
from .config import (
    JobConfig, StorageConfig, ProcessingConfig, ColumnConfig,
    S3Config, EventType, LLMConfig  # Add LLMConfig
)
```

#### In `tagging.py`:
```python
# Add these imports at the top
from typing import Optional, List, Dict, Any
from .config import CUSTOM_ID_PREFIX
```

#### In `__init__.py`:
```python
from .config import (
    S3Config,
    JobConfig,
    ProcessingConfig,
    StorageConfig,
    ColumnConfig,
    LLMConfig,  # Add to exports
    EventType,
)

from .tagging import with_tags  # Already exported, just ensure it's there

__all__ = [
    # ... existing exports ...
    "LLMConfig",  # Add to __all__
    "with_tags",  # Should already be in the list
]
```

### 6. Update README.md Examples

```python
# Basic usage with LLMConfig
from langchain_openai import ChatOpenAI
from langchain_callback_parquet_logger import (
    batch_process,
    LLMConfig,
    with_tags
)

df = pd.DataFrame({
    'prompt': ['What is AI?', 'Explain quantum computing'],
    'config': [
        with_tags(custom_id='q1', custom_id_description='Science FAQ question'),
        with_tags(custom_id='q2', custom_id_description='Science FAQ question')
    ]
})

# Simple LLM configuration
await batch_process(
    df,
    llm_config=LLMConfig(
        llm_class=ChatOpenAI,
        llm_kwargs={'model': 'gpt-4', 'temperature': 0.7}
    )
)

# With model_kwargs and structured output
from pydantic import BaseModel

class Answer(BaseModel):
    summary: str
    confidence: float

await batch_process(
    df,
    llm_config=LLMConfig(
        llm_class=ChatOpenAI,
        llm_kwargs={'model': 'gpt-4'},
        model_kwargs={'top_p': 0.9},  # Passed to model_kwargs parameter
        structured_output=Answer
    )
)

# Full configuration example with custom ID descriptions
# Prepare DataFrame with custom IDs and descriptions
df = pd.DataFrame({
    'question': ['What is dark matter?', 'How do vaccines work?'],
    'run_config': [
        with_tags(
            custom_id='research-001',
            custom_id_description='Astrophysics research question',
            tags=['physics', 'cosmology']
        ),
        with_tags(
            custom_id='research-002',
            custom_id_description='Medical research question',
            tags=['biology', 'medicine']
        )
    ]
})

await batch_process(
    df,
    llm_config=LLMConfig(
        llm_class=ChatAnthropic,
        llm_kwargs={'model': 'claude-3-opus-20240229', 'max_tokens': 1000},
        model_kwargs={'temperature': 0.5}
    ),
    job_config=JobConfig(
        category="research",
        subcategory="science",
        environment="production",
        metadata={"team": "data-science"}
    ),
    storage_config=StorageConfig(
        output_dir="./batch_logs",
        s3_config=S3Config(bucket="my-llm-logs")
    ),
    processing_config=ProcessingConfig(
        max_concurrency=100,
        show_progress=True
    ),
    column_config=ColumnConfig(
        prompt="question",
        config="run_config"
    )
)
```

## Implementation Summary

### Completed Tasks ✅

1. **Added LLMConfig dataclass** (`config.py`)
   - Created LLMConfig with llm_class, llm_kwargs, model_kwargs, structured_output
   - Added create_llm() factory method
   - Added to_metadata_dict() for serialization

2. **Updated with_tags function** (`tagging.py`)
   - Added custom_id_description parameter
   - Description stored in tags array as "custom_id_description:{value}"

3. **Updated Parquet schema** (`logger.py`)
   - Renamed column from "logger_custom_id" to "custom_id"
   - Updated all references throughout the file

4. **Refactored batch_process** (`batch.py`)
   - Replaced multiple LLM parameters with single llm_config: LLMConfig
   - Enhanced metadata tracking with comprehensive batch_config
   - Added batch execution metadata (started_at, batch_size)

5. **Updated imports** (`__init__.py`)
   - Added LLMConfig to exports

6. **Updated README examples**
   - All examples now use LLMConfig pattern
   - Added custom_id_description examples

7. **Fixed tests**
   - Updated test_batch_process.py to use LLMConfig API
   - Fixed test_core.py to use "custom_id" column name
   - Fixed background_retrieval.py to use "custom_id"

## Testing Checklist

1. **Unit Tests**: ✅ Fixed to use new API
2. **Integration Tests**: ✅ Updated for LLMConfig parameter
3. **Migration Testing**: ✅ Column rename verified

## Benefits of This Approach

1. **Simplicity**: Single LLMConfig object instead of multiple parameters
2. **Clarity**: Clear distinction between llm_kwargs and model_kwargs
3. **Observability**: Complete tracking of all configurations in metadata
4. **Consistency**: Follows existing dataclass pattern
5. **Flexibility**: Custom ID descriptions travel with the data in tags
6. **Type Safety**: Clear types for all configuration options

## Data Flow and Storage

### Batch-Level Data (Constant across all rows)
Stored in `logger_metadata` column:
- All job, storage, processing, column, and LLM configurations
- Batch execution metadata (start time, batch size)
- Custom metadata from job_config

### Row-Level Data (Varies per row)
Stored in `payload` column for each event:
- **Prompts**: In `payload.data.prompts` (llm_start events)
- **Custom ID**: In `payload.execution.custom_id`
- **Tags** (including custom_id_description): In `payload.execution.tags`
- **Row metadata**: In `payload.execution.metadata`
- **Tools**: In `payload.data.tools` (when provided)

This separation ensures efficient storage (batch config stored once) while maintaining complete per-row context.

### Accessing Custom ID Descriptions in Analysis
```python
import pandas as pd
import json

# Read logs
df = pd.read_parquet("./logs")
df['data'] = df['payload'].apply(json.loads)

# Extract custom_id and its description
df['custom_id'] = df['data'].apply(lambda x: x['execution']['custom_id'])
df['tags'] = df['data'].apply(lambda x: x['execution']['tags'])

# Parse custom_id_description from tags
def get_custom_id_description(tags):
    for tag in tags:
        if tag.startswith('custom_id_description:'):
            return tag.replace('custom_id_description:', '')
    return None

df['custom_id_description'] = df['tags'].apply(get_custom_id_description)
```

## Summary

This implementation simplifies LLM configuration through the LLMConfig dataclass while ensuring complete observability. The key improvements are:

1. **LLMConfig**: Centralizes all LLM-related settings with clear documentation of the llm_kwargs vs model_kwargs distinction
2. **Enhanced with_tags**: Adds custom_id_description directly where it's used, keeping row-level context together
3. **Comprehensive metadata**: All configurations are tracked in a structured, nested format
4. **Schema simplification**: Renaming logger_custom_id to custom_id improves clarity

The changes maintain the clean two-level architecture while providing developers with a simpler, more intuitive API for batch processing with complete observability.