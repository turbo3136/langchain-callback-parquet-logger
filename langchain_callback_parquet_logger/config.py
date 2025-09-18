"""Configuration dataclasses for LangChain Parquet Logger."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal
from enum import Enum


class EventType(Enum):
    """Supported event types for logging."""
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_ERROR = "llm_error"
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    CHAIN_ERROR = "chain_error"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    AGENT_ACTION = "agent_action"
    AGENT_FINISH = "agent_finish"

    @classmethod
    def default_set(cls) -> List[str]:
        """Return default event types (LLM events only)."""
        return [cls.LLM_START.value, cls.LLM_END.value, cls.LLM_ERROR.value]

    @classmethod
    def all_events(cls) -> List[str]:
        """Return all available event types."""
        return [e.value for e in cls]


@dataclass
class S3Config:
    """Configuration for S3 storage backend."""
    bucket: str
    prefix: str = "langchain-logs/"
    on_failure: Literal["error", "continue"] = "error"
    retry_attempts: int = 3

    def __post_init__(self):
        """Ensure prefix ends with /."""
        if self.prefix and not self.prefix.endswith('/'):
            self.prefix += '/'


@dataclass
class JobConfig:
    """Job metadata configuration for batch processing."""
    category: str = "batch_processing"
    subcategory: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    environment: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingConfig:
    """Processing configuration for batch operations."""
    max_concurrency: int = 100
    buffer_size: int = 1000
    show_progress: bool = True
    return_exceptions: bool = True
    return_results: bool = False
    event_types: Optional[List[str]] = None
    partition_on: Optional[Literal["date"]] = "date"

    def __post_init__(self):
        """Set default event types if not specified."""
        if self.event_types is None:
            self.event_types = EventType.default_set()


@dataclass
class StorageConfig:
    """Storage configuration for batch processing."""
    output_dir: str = "./batch_logs"
    path_template: str = "{job_category}/{job_subcategory}"
    s3_config: Optional[S3Config] = None


@dataclass
class ColumnConfig:
    """DataFrame column configuration."""
    prompt: str = "prompt"
    config: str = "config"
    tools: Optional[str] = "tools"


# Constants
CUSTOM_ID_PREFIX = "logger_custom_id:"