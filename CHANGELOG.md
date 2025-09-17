# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-01-17

### Added
- Complete data capture in `raw` payload section
- `_serialize_any()` helper for comprehensive object serialization
- All callback arguments now captured in raw (positional + kwargs)
- Tests for raw data capture verification

### Changed (BREAKING)
- **BREAKING**: Removed `provider` column from Parquet schema (8 columns → 7 columns)
- **BREAKING**: Provider detection has been completely removed in favor of using LangChain's native `_llm_type`
- The `llm_type` is now captured in the event payload data for `llm_start` events
- Schema now contains: timestamp, run_id, parent_run_id, logger_custom_id, event_type, logger_metadata, payload

### Removed
- Removed `DEFAULT_PROVIDER` constant from config
- Removed `_detect_provider()` method from ParquetLogger
- Removed `_detect_llm_provider()` function from batch processing
- Removed all provider detection logic and mappings

### Migration Guide
To migrate from v1.x to v2.0:

1. **Reading logs**: The `provider` column no longer exists. To get LLM type information:
   ```python
   import json
   df = pd.read_parquet("logs/")
   # Parse payload to get llm_type
   for row in df.itertuples():
       payload = json.loads(row.payload)
       if payload['event_type'] == 'llm_start':
           llm_type = payload['data'].get('llm_type', 'unknown')
   ```

2. **Schema changes**: Update any code that expects 8 columns to handle 7 columns

3. **Provider filtering**: Instead of filtering by provider column, filter by parsing llm_type from payload

### Why This Change?
- Simplifies codebase by removing ~50+ lines of provider detection code
- More accurate - uses LangChain's own `_llm_type` classification
- Zero maintenance - no need to update provider mappings for new LLMs
- Future-proof - works automatically with any LangChain LLM