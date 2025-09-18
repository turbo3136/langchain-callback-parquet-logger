# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-09-18

### Added
- **New LLMConfig dataclass** for cleaner LLM configuration
  - Separates `llm_kwargs` (constructor args) from `model_kwargs` (API params)
  - Includes `create_llm()` factory method and `to_metadata_dict()` for metadata tracking
  - Built-in support for `structured_output` with Pydantic models
- **Enhanced with_tags function**
  - New `custom_id_description` parameter for better context
  - Descriptions stored in tags as "custom_id_description:{value}"
- **Version path sanitization**
  - `job_version_safe` template variable for filesystem-safe version paths (dots replaced with underscores)
  - Comprehensive tests for version path validation in both local and S3 storage
- **Complete data capture** in `raw` payload section
  - `_serialize_any()` helper for comprehensive object serialization
  - All callback arguments now captured in raw (positional + kwargs)
  - Tests for raw data capture verification
- **Enhanced metadata tracking**
  - Comprehensive batch-level metadata in nested `batch_config` structure
  - All dataclass configs tracked and visible
  - Row-level context preserved through processing

### Changed (BREAKING)
- **BREAKING**: Removed `provider` column from Parquet schema (8 columns → 7 columns)
- **BREAKING**: Provider detection has been completely removed in favor of using LangChain's native `_llm_type`
- **BREAKING**: `batch_process` API completely redesigned with dataclass configs:
  - Now uses `llm_config: LLMConfig` parameter instead of multiple parameters (`llm`, `llm_class`, `llm_kwargs`, `structured_output`)
  - All configs now use dataclasses: `JobConfig`, `StorageConfig`, `ProcessingConfig`, `ColumnConfig`
- **BREAKING**: Default `StorageConfig.path_template` now includes version: `"{job_category}/{job_subcategory}/v{job_version_safe}"`
- **BREAKING**: Schema column renamed from `logger_custom_id` to `custom_id`
- **BREAKING**: S3 configuration now via `S3Config` dataclass in `StorageConfig.s3_config`
- **BREAKING**: JobConfig fields now optional (default to None instead of strings)
  - Only `category` is required
  - `subcategory`, `description`, `version`, `environment`, `metadata` all optional
- The `llm_type` is now captured in the event payload data for `llm_start` events
- Schema now contains: timestamp, run_id, parent_run_id, custom_id, event_type, logger_metadata, payload
- Version defaults to "unversioned" instead of "1.0.0" when not specified in JobConfig
- S3 paths now properly mirror local structure (base prefix + path_template)

### Removed
- Removed `DEFAULT_PROVIDER` constant from config
- Removed `_detect_provider()` method from ParquetLogger
- Removed `_detect_llm_provider()` function from batch processing
- Removed all provider detection logic and mappings

### Migration Guide
To migrate from v1.x to v2.0:

1. **batch_process API changes**:
   ```python
   # Old (v1.x)
   await batch_process(
       df,
       llm=llm_instance,  # or llm_class with llm_kwargs
       structured_output=MyModel
   )

   # New (v2.0)
   await batch_process(
       df,
       llm_config=LLMConfig(
           llm_class=ChatOpenAI,
           llm_kwargs={'model': 'gpt-4'},
           model_kwargs={'top_p': 0.9},  # Additional API params
           structured_output=MyModel
       ),
       job_config=JobConfig(category="my_job"),  # Only category required
       storage_config=StorageConfig(...),  # Optional
       processing_config=ProcessingConfig(...)  # Optional
   )
   ```

2. **S3 configuration changes**:
   ```python
   # Old (v1.x)
   logger = ParquetLogger(
       log_dir="./logs",
       s3_bucket="my-bucket",
       s3_prefix="logs/",
       s3_on_failure="error"
   )

   # New (v2.0)
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

3. **Schema changes**:
   - Column renamed: `logger_custom_id` → `custom_id`
   - `provider` column removed (8 columns → 7 columns)
   - To get LLM type, parse from payload:
   ```python
   import json
   df = pd.read_parquet("logs/")
   for row in df.itertuples():
       payload = json.loads(row.payload)
       if payload['event_type'] == 'llm_start':
           llm_type = payload['data'].get('llm_type', 'unknown')
   ```

4. **Path structure changes**:
   - Default paths now include version: `category/subcategory/v1_0_0/`
   - Version defaults to "unversioned" instead of "1.0.0"
   - Use `job_version_safe` in templates for sanitized versions

### Why This Change?
- Simplifies codebase by removing ~50+ lines of provider detection code
- More accurate - uses LangChain's own `_llm_type` classification
- Zero maintenance - no need to update provider mappings for new LLMs
- Future-proof - works automatically with any LangChain LLM