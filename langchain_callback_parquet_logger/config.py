"""Configuration dataclasses for LangChain Parquet Logger."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal, Type, Union
from enum import Enum


class EventType(Enum):
    """Supported event types for logging."""
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    LLM_ERROR = "llm_error"
    CHAT_MODEL_START = "chat_model_start"
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
        """Return default event types (LLM and chat model events)."""
        return [cls.LLM_START.value, cls.LLM_END.value, cls.LLM_ERROR.value,
                cls.CHAT_MODEL_START.value]

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
    connect_timeout: int = 10  # S3 connection timeout in seconds
    read_timeout: int = 30  # S3 read timeout in seconds

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
    partition_on: Optional[Union[Literal["date", "event_type"], List[Literal["date", "event_type"]]]] = "date"
    row_timeout: Optional[float] = None  # Per-row timeout in seconds (None = no timeout)

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
    response_id: str = "response_id"
    custom_id: str = "custom_id"


@dataclass
class RetrievalConfig:
    """Configuration for retrieve_background_responses() — polling OpenAI background responses.

    Separates polling params (how long to wait for completion) from execution params
    (concurrency, timeouts, logging).

    The ``source`` field controls where the retriever looks for pending responses:

    - ``"memory"`` (default): only use response IDs captured in-memory during ``run()``;
      never touches the file system or S3.  Returns empty when no IDs are available.
    - ``"local"``: auto-discover pending responses from local Parquet files.
    - ``"s3"``: auto-discover pending responses from S3.

    It only affects the *read* side — new retrieval events are always written to the
    full configured storage regardless of this setting.

    Example:
        config = RetrievalConfig(
            source="s3",            # discover pending responses from S3
            poll_interval=30.0,     # check every 30 seconds
            max_poll_attempts=40,   # give up after ~20 minutes
            batch_size=50,          # 50 concurrent polls
        )
    """
    source: Literal["memory", "local", "s3"] = "memory"  # where to read pending responses from
    poll_interval: float = 30.0       # seconds between status checks when response is pending
    max_poll_attempts: int = 40       # max polls per response before giving up (40 × 30s ≈ 20 min)
    batch_size: int = 50              # number of concurrent requests
    timeout: float = 30.0            # per-request HTTP timeout in seconds
    max_retries: int = 3             # retries for transient errors (5xx, rate limits, timeouts)
    show_progress: bool = True
    return_results: bool = True
    checkpoint_file: Optional[str] = None


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

    def create_llm(self, callbacks: Optional[list] = None) -> Any:
        """Create the LLM instance from config.

        This combines llm_kwargs and model_kwargs appropriately:
        - llm_kwargs are passed directly to the LLM constructor
        - model_kwargs are passed as the 'model_kwargs' parameter

        Args:
            callbacks: Optional list of callbacks to attach to the LLM
        """
        kwargs = (self.llm_kwargs or {}).copy()
        if self.model_kwargs:
            # Most LangChain LLMs accept a model_kwargs parameter
            # for additional model-specific parameters
            kwargs['model_kwargs'] = self.model_kwargs

        # Add callbacks if provided
        if callbacks:
            kwargs['callbacks'] = callbacks

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