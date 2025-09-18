"""Configuration dataclasses for LangChain Parquet Logger."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal, Type
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
    path_template: str = "{job_category}/{job_subcategory}/v{job_version_safe}"
    s3_config: Optional[S3Config] = None


@dataclass
class ColumnConfig:
    """DataFrame column configuration."""
    prompt: str = "prompt"
    config: str = "config"
    tools: Optional[str] = "tools"


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


# Constants
CUSTOM_ID_PREFIX = "logger_custom_id:"