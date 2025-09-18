# langchain_callback_parquet_logger/logger.py
"""Core Parquet logger for LangChain callbacks."""

import json
import threading
import atexit
import warnings
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional, Literal, Set

import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.callbacks import BaseCallbackHandler

from .config import S3Config, EventType
from .storage import create_storage, StorageBackend
from .tagging import extract_custom_id


# Define explicit schema to avoid type inference issues
SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("run_id", pa.string()),
    ("parent_run_id", pa.string()),
    ("custom_id", pa.string()),
    ("event_type", pa.string()),
    ("logger_metadata", pa.string()),
    ("payload", pa.string()),
])


class ParquetLogger(BaseCallbackHandler):
    """Parquet logger for LangChain callbacks with flexible storage backends."""

    def __init__(
        self,
        log_dir: str = "./llm_logs",
        buffer_size: int = 100,
        logger_metadata: Optional[Dict[str, Any]] = None,
        partition_on: Optional[Literal["date"]] = "date",
        event_types: Optional[List[str]] = None,
        s3_config: Optional[S3Config] = None
    ):
        """
        Initialize the Parquet logger.

        Args:
            log_dir: Directory to save log files
            buffer_size: Number of entries to buffer before flushing to disk
            logger_metadata: Optional metadata to include with all log entries
            partition_on: Partitioning strategy - "date" or None
            event_types: List of event types to log (defaults to LLM events only)
            s3_config: Optional S3 configuration for uploading logs
        """
        # Validate inputs
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.logger_metadata = logger_metadata or {}
        self.partition_on = partition_on

        # Set event types to log
        if event_types is None:
            self.event_types: Set[str] = set(EventType.default_set())
        else:
            self.event_types: Set[str] = set(event_types)

        # Create storage backend
        self.storage = create_storage(log_dir, s3_config)

        # Safely serialize metadata
        try:
            self.logger_metadata_json = json.dumps(self.logger_metadata, default=str)
        except Exception:
            self.logger_metadata_json = "{}"

        self.buffer = []
        self.lock = threading.Lock()

        # Register flush to run on program exit
        atexit.register(self.flush)

        # Simple notebook warning
        if self._is_notebook() and buffer_size > 10:
            warnings.warn(
                f"Notebook detected: buffer_size={buffer_size}. "
                "Use context manager or call flush() for immediate writes.",
                stacklevel=2
            )

    def _is_notebook(self) -> bool:
        """Detect if running in a notebook environment."""
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

    def _serialize_any(self, obj: Any) -> Any:
        """Try all possible serialization methods for complete data capture."""
        try:
            # Try various serialization methods in order of preference
            if hasattr(obj, 'model_dump'):  # Pydantic v2
                return obj.model_dump()
            elif hasattr(obj, 'dict'):  # Pydantic v1 / LangChain objects
                return obj.dict()
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                # Get object attributes (skip private ones)
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
            else:
                # Return as-is, let _safe_json_dumps handle edge cases
                return obj
        except Exception:
            # If all else fails, return as-is
            return obj

    def _safe_json_dumps(self, obj: Any) -> str:
        """Convert object to JSON string safely."""
        def default(o):
            if hasattr(o, '__str__'):
                return str(o)
            return f"<{type(o).__name__}>"
        return json.dumps(obj, default=default)

    def _create_standard_payload(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """Create minimal standardized payload structure."""
        parent_run_id = kwargs.get('parent_run_id')
        parent_run_id = str(parent_run_id) if parent_run_id else ''

        tags = kwargs.get('tags', []) or []

        return {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution": {
                "run_id": str(kwargs.get('run_id', '')),
                "parent_run_id": parent_run_id,
                "custom_id": extract_custom_id(tags),
                "tags": tags,
                "metadata": kwargs.get('metadata', {}) or {}
            },
            "data": {},  # Populated by event handlers
            "raw": kwargs.copy()
        }

    def _add_error_info(self, payload: Dict[str, Any], error: Exception) -> None:
        """Add error information to payload."""
        payload["data"]["error"] = {
            "message": str(error),
            "type": type(error).__name__
        }

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        """Convert LangChain response to dict format."""
        try:
            if hasattr(response, 'dict'):
                return response.dict()
            elif hasattr(response, 'to_dict'):
                return response.to_dict()
            elif isinstance(response, dict):
                return response
            else:
                return {'content': str(response)}
        except Exception as e:
            return {'content': str(response), 'conversion_error': str(e)}

    def _log_event(self, payload: Dict[str, Any]) -> None:
        """Create entry and add to buffer."""
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': payload["execution"]["run_id"],
            'parent_run_id': payload["execution"]["parent_run_id"],
            'custom_id': payload["execution"]["custom_id"],
            'event_type': payload["event_type"],
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload)
        }
        self._add_entry(entry)

    def _handle_event(self, event_type: str, primary_data: Dict[str, Any], **kwargs):
        """Generic event handler to reduce duplication."""
        if event_type not in self.event_types:
            return

        payload = self._create_standard_payload(event_type, **kwargs)
        payload["data"] = primary_data
        # Raw already has kwargs from _create_standard_payload
        self._log_event(payload)

    # Event handlers
    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        """Log LLM start event."""
        # Keep structured data for easy access
        data = {
            "prompts": prompts,
            "llm_type": serialized.get('_type', 'unknown'),  # Extract LangChain's native _type
            "serialized": serialized,
            "model": serialized.get('kwargs', {}).get('model_name', ''),
            "invocation_params": serialized.get('kwargs', {}),
            "tools": kwargs.get('tools', []) or []
        }
        if 'messages' in kwargs:
            data["messages"] = kwargs['messages']

        # Capture everything in raw
        kwargs['serialized'] = serialized
        kwargs['prompts'] = prompts
        self._handle_event('llm_start', data, **kwargs)

    def on_llm_end(self, response, **kwargs):
        """Log LLM end event."""
        # Keep structured data for easy access
        response_data = self._convert_response(response)
        data = {"response": response_data}

        if hasattr(response, 'llm_output') and response.llm_output:
            data["usage"] = response.llm_output.get('token_usage', {})
            data["model"] = response.llm_output.get('model_name', '')

        if hasattr(response, 'response_metadata'):
            data["response_metadata"] = response.response_metadata

        # Capture complete response in raw
        kwargs['response'] = self._serialize_any(response)
        self._handle_event('llm_end', data, **kwargs)

    def on_llm_error(self, error, **kwargs):
        """Log LLM error event."""
        if 'llm_error' not in self.event_types:
            return

        payload = self._create_standard_payload('llm_error', **kwargs)
        self._add_error_info(payload, error)
        # Capture complete error in raw
        payload["raw"]["error"] = self._serialize_any(error)
        self._log_event(payload)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Log chain start event."""
        data = {
            "inputs": inputs,
            "serialized": serialized,
            "model": serialized.get('name', '')
        }
        # Capture everything in raw
        kwargs['serialized'] = serialized
        kwargs['inputs'] = inputs
        self._handle_event('chain_start', data, **kwargs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Log chain end event."""
        # Capture everything in raw
        kwargs['outputs'] = self._serialize_any(outputs)
        self._handle_event('chain_end', {"outputs": outputs}, **kwargs)

    def on_chain_error(self, error: Exception, **kwargs):
        """Log chain error event."""
        if 'chain_error' not in self.event_types:
            return

        payload = self._create_standard_payload('chain_error', **kwargs)
        self._add_error_info(payload, error)
        # Capture complete error in raw
        payload["raw"]["error"] = self._serialize_any(error)
        self._log_event(payload)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Log tool start event."""
        data = {
            "input_str": input_str,
            "serialized": serialized,
            "model": serialized.get('name', '')
        }
        if 'description' in serialized:
            data["description"] = serialized['description']
        # Capture everything in raw
        kwargs['serialized'] = serialized
        kwargs['input_str'] = input_str
        self._handle_event('tool_start', data, **kwargs)

    def on_tool_end(self, output: str, **kwargs):
        """Log tool end event."""
        # Capture everything in raw
        kwargs['output'] = self._serialize_any(output)
        self._handle_event('tool_end', {"output": output}, **kwargs)

    def on_tool_error(self, error: Exception, **kwargs):
        """Log tool error event."""
        if 'tool_error' not in self.event_types:
            return

        payload = self._create_standard_payload('tool_error', **kwargs)
        self._add_error_info(payload, error)
        # Capture complete error in raw
        payload["raw"]["error"] = self._serialize_any(error)
        self._log_event(payload)

    def on_agent_action(self, action, **kwargs):
        """Log agent action event."""
        # Keep structured data for easy access
        if hasattr(action, '__dict__'):
            action_data = {
                'tool': getattr(action, 'tool', ''),
                'tool_input': getattr(action, 'tool_input', ''),
                'log': getattr(action, 'log', ''),
            }
        else:
            action_data = {'action': str(action)}

        # Capture complete action in raw
        kwargs['action'] = self._serialize_any(action)
        self._handle_event('agent_action', {"action": action_data}, **kwargs)

    def on_agent_finish(self, finish, **kwargs):
        """Log agent finish event."""
        # Keep structured data for easy access
        if hasattr(finish, '__dict__'):
            finish_data = {
                'return_values': getattr(finish, 'return_values', {}),
                'log': getattr(finish, 'log', ''),
            }
        else:
            finish_data = {'finish': str(finish)}

        # Capture complete finish in raw
        kwargs['finish'] = self._serialize_any(finish)
        self._handle_event('agent_finish', {"finish": finish_data}, **kwargs)

    # Buffer management
    def _add_entry(self, entry):
        """Add entry to buffer and flush if needed."""
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                self._flush_locked()

    def flush(self):
        """Manual flush of the buffer."""
        with self.lock:
            self._flush_locked()

    def _flush_locked(self):
        """Internal flush that assumes lock is already held."""
        if not self.buffer:
            return

        # Copy buffer and clear it while holding lock
        buffer_to_write = self.buffer.copy()
        self.buffer = []

        # Release lock before doing I/O
        self._write_buffer(buffer_to_write)

    def _write_buffer(self, buffer):
        """Write buffer to Parquet file (called without lock held)."""
        try:
            # Build columns explicitly
            ts = pa.array([e["timestamp"] for e in buffer],
                          type=pa.timestamp("us", tz="UTC"))
            run_id = pa.array([e["run_id"] for e in buffer], type=pa.string())
            parent_run_id = pa.array([e.get("parent_run_id", "") for e in buffer],
                                    type=pa.string())
            custom_id = pa.array([e["custom_id"] for e in buffer],
                               type=pa.string())
            event_type = pa.array([e["event_type"] for e in buffer], type=pa.string())
            logger_metadata = pa.array([e["logger_metadata"] for e in buffer],
                                      type=pa.string())
            payload = pa.array([e["payload"] for e in buffer], type=pa.string())

            # Create table with explicit schema
            table = pa.Table.from_arrays(
                [ts, run_id, parent_run_id, custom_id, event_type,
                 logger_metadata, payload],
                schema=SCHEMA
            )

            # Determine relative path based on partitioning
            if self.partition_on == "date":
                today = date.today()
                relative_path = Path(f"date={today}") / f"logs_{datetime.now().strftime('%H%M%S_%f')}.parquet"
            else:
                relative_path = Path(f"logs_{datetime.now().strftime('%H%M%S_%f')}.parquet")

            # Write using storage backend
            self.storage.write(table, relative_path)

        except RuntimeError:
            # Re-raise storage errors
            raise
        except Exception as e:
            import traceback
            print(f"Failed to write logs: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")

    # Context manager support
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure buffer is flushed."""
        self.flush()
        return False