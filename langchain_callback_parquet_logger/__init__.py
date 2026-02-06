"""LangChain Parquet Logger - High-performance callback handler for logging LangChain LLM interactions."""

from .logger import ParquetLogger
from .tagging import with_tags
from .config import (
    S3Config,
    JobConfig,
    ProcessingConfig,
    StorageConfig,
    ColumnConfig,
    LLMConfig,
    EventType,
)
from .batch import batch_run, batch_process

# Optional imports with better error messages
try:
    from .background_retrieval import retrieve_background_responses
except ImportError:
    # Background retrieval requires openai
    retrieve_background_responses = None


__version__ = "3.0.1"

__all__ = [
    # Core
    'ParquetLogger',
    'with_tags',

    # Batch processing
    'batch_run',
    'batch_process',

    # Configurations
    'S3Config',
    'JobConfig',
    'ProcessingConfig',
    'StorageConfig',
    'ColumnConfig',
    'LLMConfig',
    'EventType',

    # Version
    '__version__',
]

# Add background retrieval if available
if retrieve_background_responses is not None:
    __all__.append('retrieve_background_responses')
