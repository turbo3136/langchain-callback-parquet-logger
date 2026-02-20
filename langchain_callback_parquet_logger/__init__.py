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
    RetrievalConfig,
)
from .batch import batch_run, batch_process, retrieve_batch_responses

# Optional imports with better error messages
try:
    from .background_retrieval import retrieve_background_responses
except ImportError:
    # Background retrieval requires openai
    retrieve_background_responses = None


__version__ = "3.1.2"

__all__ = [
    # Core
    'ParquetLogger',
    'with_tags',

    # Batch processing
    'batch_run',
    'batch_process',
    'retrieve_batch_responses',

    # Configurations
    'S3Config',
    'JobConfig',
    'ProcessingConfig',
    'StorageConfig',
    'ColumnConfig',
    'LLMConfig',
    'EventType',
    'RetrievalConfig',

    # Version
    '__version__',
]

# Add low-level background retrieval if available (requires openai)
if retrieve_background_responses is not None:
    __all__.append('retrieve_background_responses')
