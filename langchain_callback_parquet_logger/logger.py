# langchain_callback_parquet_logger/logger.py
"""Core Parquet logger for LangChain callbacks."""

import json
import queue
import threading
import atexit
import warnings
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional, Literal, Set, Sequence, Union

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
        partition_on: Optional[Union[Literal["date", "event_type"], List[Literal["date", "event_type"]]]] = "date",
        event_types: Optional[List[str]] = None,
        s3_config: Optional[S3Config] = None
    ):
        """
        Initialize the Parquet logger.

        Args:
            log_dir: Directory to save log files
            buffer_size: Number of entries to buffer before flushing to disk
            logger_metadata: Optional metadata to include with all log entries
            partition_on: Partitioning strategy. A key or ordered list of keys.
                Valid keys: "date", "event_type". None for flat structure.
                Examples: "date" (default), ["date", "event_type"], ["event_type"]
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
        self._write_queue = queue.Queue()
        self._writer_errors = []
        self._writer_errors_lock = threading.Lock()

        # Start background writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="parquet-writer"
        )
        self._writer_thread.start()

        # Register flush to run on program exit
        atexit.register(self.flush)

        # Suppress Pydantic serialization warnings globally so it covers
        # LangChain/OpenAI SDK internals calling model_dump() during ainvoke()
        warnings.filterwarnings("ignore", category=UserWarning, module=r"^pydantic")

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

    @staticmethod
    def _serialize_obj(obj: Any) -> Any:
        """Try model_dump → to_dict → __dict__ in order. Returns obj unchanged if all fail."""
        try:
            if hasattr(obj, 'model_dump'):
                result = obj.model_dump(mode='json', by_alias=False)
                if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                    return result
            elif hasattr(obj, 'to_dict'):
                return obj.to_dict()
            elif hasattr(obj, '__dict__'):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        except Exception:
            pass
        return obj

    def _serialize_any(self, obj: Any) -> Any:
        """Try all possible serialization methods for complete data capture."""
        try:
            # Special handling for LLMResult to preserve nested AIMessage metadata
            if obj.__class__.__name__ == 'LLMResult':
                result = self._serialize_obj(obj)

                # Fix nested message serialization to preserve all fields
                if isinstance(result, dict) and 'generations' in result and hasattr(obj, 'generations'):
                    for i, gen_list in enumerate(obj.generations):
                        for j, gen in enumerate(gen_list):
                            if hasattr(gen, 'message'):
                                result['generations'][i][j]['message'] = self._serialize_obj(gen.message)
                return result

            return self._serialize_obj(obj)
        except Exception:
            return obj

    def _safe_json_dumps(self, obj: Any) -> str:
        """Convert object to JSON string safely."""
        def default(o):
            result = self._serialize_obj(o)
            if result is not o:
                return result
            return str(o)
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

    def _extract_message_metadata(self, response: Any) -> Dict[str, Any]:
        """Extract metadata from first AIMessage in generations if present."""
        metadata = {}
        if hasattr(response, 'generations') and response.generations:
            try:
                # Use try/except to handle both real lists and mock objects
                if response.generations[0] and len(response.generations[0]) > 0:
                    gen = response.generations[0][0]
                    if hasattr(gen, 'message'):
                        msg = gen.message
                        # Extract usage_metadata if present
                        if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
                            metadata['usage_metadata'] = msg.usage_metadata
                        # Extract response_metadata if present and not already captured
                        if hasattr(msg, 'response_metadata') and msg.response_metadata:
                            metadata['response_metadata'] = msg.response_metadata
            except (TypeError, IndexError, AttributeError):
                # Handle mock objects or malformed data gracefully
                pass
        return metadata

    def _convert_response(self, response: Any) -> Dict[str, Any]:
        """Convert LangChain response to dict format using _serialize_any."""
        try:
            result = self._serialize_any(response)
            if isinstance(result, dict):
                return result
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

    def _handle_error_event(self, event_type: str, error: Exception, **kwargs) -> None:
        """Generic error event handler shared by on_llm_error, on_chain_error, on_tool_error."""
        if event_type not in self.event_types:
            return
        payload = self._create_standard_payload(event_type, **kwargs)
        self._add_error_info(payload, error)
        payload["raw"]["error"] = self._serialize_any(error)
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

    def on_chat_model_start(self, serialized: Dict, messages: List[List[Any]], **kwargs):
        """Log chat model start event.

        Called when a chat model starts running. Chat models (e.g., ChatOpenAI)
        may fire this instead of on_llm_start.
        """
        data = {
            "messages": self._serialize_any(messages),
            "llm_type": serialized.get('_type', 'unknown'),
            "serialized": serialized,
            "model": serialized.get('kwargs', {}).get('model_name', ''),
            "invocation_params": serialized.get('kwargs', {}),
        }

        kwargs['serialized'] = serialized
        kwargs['messages'] = messages
        self._handle_event('chat_model_start', data, **kwargs)

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

        # Extract metadata from nested AIMessage (usage_metadata, response_metadata, etc.)
        data.update(self._extract_message_metadata(response))

        # Capture complete response in raw
        kwargs['response'] = self._serialize_any(response)
        self._handle_event('llm_end', data, **kwargs)

    def on_llm_error(self, error, **kwargs):
        """Log LLM error event."""
        self._handle_error_event('llm_error', error, **kwargs)

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
        # Serialize outputs first to handle Pydantic models
        serialized_outputs = self._serialize_any(outputs)
        # Capture serialized version in raw
        kwargs['outputs'] = serialized_outputs
        # Use serialized version in data section too
        self._handle_event('chain_end', {"outputs": serialized_outputs}, **kwargs)

    def on_chain_error(self, error: Exception, **kwargs):
        """Log chain error event."""
        self._handle_error_event('chain_error', error, **kwargs)

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
        # Serialize output first to handle Pydantic models from structured output
        serialized_output = self._serialize_any(output)
        # Capture serialized version in raw
        kwargs['output'] = serialized_output
        # Use serialized version in data section too
        self._handle_event('tool_end', {"output": serialized_output}, **kwargs)

    def on_tool_error(self, error: Exception, **kwargs):
        """Log tool error event."""
        self._handle_error_event('tool_error', error, **kwargs)

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
        """Add entry to buffer and enqueue writes when buffer is full.

        Never does I/O directly — all writes happen on the background writer thread.
        """
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                buffer_to_write = self.buffer.copy()
                self.buffer = []
                self._write_queue.put(("write", buffer_to_write))

    def flush(self, timeout: float = 120.0):
        """Flush the buffer and wait for all pending writes to complete.

        Args:
            timeout: Maximum seconds to wait for pending writes (default: 120).
        """
        # Drain buffer under lock
        with self.lock:
            if self.buffer:
                buffer_to_write = self.buffer.copy()
                self.buffer = []
                self._write_queue.put(("write", buffer_to_write))

        # Send a barrier and wait for the writer thread to process everything
        barrier = threading.Event()
        self._write_queue.put(("barrier", barrier))
        barrier.wait(timeout=timeout)

        # Raise any accumulated errors from the writer thread
        with self._writer_errors_lock:
            if self._writer_errors:
                errors = self._writer_errors.copy()
                self._writer_errors = []
                raise errors[0]

    def _writer_loop(self):
        """Background thread that processes write queue items."""
        while True:
            try:
                item = self._write_queue.get()
                if item is None:
                    break
                action, payload = item
                if action == "write":
                    try:
                        self._write_buffer(payload)
                    except Exception as e:
                        with self._writer_errors_lock:
                            self._writer_errors.append(e)
                elif action == "barrier":
                    payload.set()  # signal the Event
            except Exception:
                pass  # keep the writer alive

    def _normalize_partitions(self) -> List[str]:
        """Return partition_on as an ordered list of partition keys."""
        if self.partition_on is None:
            return []
        if isinstance(self.partition_on, str):
            return [self.partition_on]
        return list(self.partition_on)

    def _build_table(self, entries: list) -> pa.Table:
        """Build a PyArrow table from a list of entry dicts."""
        ts = pa.array([e["timestamp"] for e in entries], type=pa.timestamp("us", tz="UTC"))
        run_id = pa.array([e["run_id"] for e in entries], type=pa.string())
        parent_run_id = pa.array([e.get("parent_run_id", "") for e in entries], type=pa.string())
        custom_id = pa.array([e["custom_id"] for e in entries], type=pa.string())
        event_type_col = pa.array([e["event_type"] for e in entries], type=pa.string())
        logger_metadata = pa.array([e["logger_metadata"] for e in entries], type=pa.string())
        payload = pa.array([e["payload"] for e in entries], type=pa.string())
        return pa.Table.from_arrays(
            [ts, run_id, parent_run_id, custom_id, event_type_col, logger_metadata, payload],
            schema=SCHEMA,
        )

    def _write_buffer(self, buffer: list) -> None:
        """Write buffer to Parquet file(s) (called without lock held).

        Any exception raised here propagates to _writer_loop, which stores it in
        _writer_errors so it surfaces at the next flush() call.  This covers both
        S3 errors (RuntimeError from S3Storage with on_failure='error') and local
        write failures (OSError, PermissionError, disk-full, etc.).
        """
        if not buffer:
            return

        partitions = self._normalize_partitions()

        def get_partition_dir(entry: dict) -> Optional[Path]:
            parts = []
            for p in partitions:
                if p == "date":
                    parts.append(f"date={date.today()}")
                elif p == "event_type":
                    parts.append(f"event_type={entry.get('event_type', 'unknown')}")
            return Path(*parts) if parts else None

        # Group entries by their partition directory (preserves insertion order)
        groups: Dict[str, list] = {}
        for entry in buffer:
            partition_dir = get_partition_dir(entry)
            key = str(partition_dir) if partition_dir is not None else ""
            if key not in groups:
                groups[key] = []
            groups[key].append(entry)

        timestamp_str = datetime.now().strftime("%H%M%S_%f")
        for path_key, entries in groups.items():
            filename = f"logs_{timestamp_str}.parquet"
            relative_path = (Path(path_key) / filename) if path_key else Path(filename)
            table = self._build_table(entries)
            self.storage.write(table, relative_path)

    # Context manager support
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure buffer is flushed."""
        self.flush()
        return False
