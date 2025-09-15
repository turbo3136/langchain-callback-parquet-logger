# langchain_parquet_logger/logger.py
import json
import threading
import time  # Explicit import to ensure it's available for PyArrow
import atexit  # For automatic cleanup on exit
import warnings
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional, Literal, Set
from io import BytesIO
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.callbacks import BaseCallbackHandler

# Optional boto3 import for S3 support
try:
    import boto3
    HAS_BOTO3 = True
except ImportError:
    boto3 = None
    HAS_BOTO3 = False

# Define explicit schema to avoid type inference issues
SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("run_id", pa.string()),
    ("parent_run_id", pa.string()),
    ("logger_custom_id", pa.string()),
    ("event_type", pa.string()),
    ("provider", pa.string()),
    ("logger_metadata", pa.string()),
    ("payload", pa.string()),
])

class ParquetLogger(BaseCallbackHandler):
    """Simplified Parquet logger with flexible JSON payload schema."""
    
    def __init__(self,
                 log_dir: str = "./llm_logs",
                 buffer_size: int = 100,
                 provider: str = "openai",
                 logger_metadata: Optional[Dict[str, Any]] = None,
                 partition_on: Optional[Literal["date"]] = "date",
                 event_types: Optional[List[str]] = None,
                 s3_bucket: Optional[str] = None,
                 s3_prefix: str = "langchain-logs/",
                 s3_on_failure: Literal["error", "continue"] = "error",
                 s3_retry_attempts: int = 3):
        """
        Initialize the Parquet logger.
        
        Args:
            log_dir: Directory to save log files
            buffer_size: Number of entries to buffer before flushing to disk
            provider: LLM provider name (default: "openai")
            logger_metadata: Optional dictionary of metadata to include with all log entries
            partition_on: Partitioning strategy - "date" for daily partitions or None for no partitioning (default: "date")
            event_types: List of event types to log. If None, logs only LLM events.
                Available types: 'llm_start', 'llm_end', 'llm_error',
                                'chain_start', 'chain_end', 'chain_error',
                                'tool_start', 'tool_end', 'tool_error',
                                'agent_action', 'agent_finish'
            s3_bucket: Optional S3 bucket name for uploading logs
            s3_prefix: Prefix/folder path in S3 bucket (default: "langchain-logs/")
            s3_on_failure: How to handle S3 upload failures - "error" to raise exception, "continue" to log and continue
            s3_retry_attempts: Number of retry attempts for failed S3 uploads (default: 3)
        """
        # Validate inputs
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.provider = provider
        self.logger_metadata = logger_metadata or {}
        self.partition_on = partition_on
        
        # S3 configuration
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip('/') + '/' if s3_prefix else ''
        self.s3_on_failure = s3_on_failure
        self.s3_retry_attempts = s3_retry_attempts
        
        # Check if boto3 is available when S3 is configured
        if self.s3_bucket and not HAS_BOTO3:
            raise ImportError(
                "boto3 is required for S3 support. "
                "Install it with: pip install langchain-callback-parquet-logger[s3]"
            )
        
        # Set event types to log
        if event_types is None:
            # Default to LLM events only for backward compatibility
            self.event_types: Set[str] = {'llm_start', 'llm_end', 'llm_error'}
        else:
            self.event_types: Set[str] = set(event_types)
        # Safely serialize metadata with fallback
        try:
            self.logger_metadata_json = json.dumps(self.logger_metadata, default=str)
        except Exception:
            self.logger_metadata_json = "{}"  # Fallback to empty JSON if serialization fails
        
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
            if get_ipython() is not None:
                return True
        except ImportError:
            pass
        return False
    
    def _extract_custom_id_from_tags(self, kwargs: Dict[str, Any]) -> str:
        """Extract logger_custom_id from tags."""
        tags = kwargs.get('tags', []) or []
        for tag in tags:
            if isinstance(tag, str) and tag.startswith('logger_custom_id:'):
                return tag.split(':', 1)[1]  # Everything after first colon
        return ''
    
    def _safe_json_dumps(self, obj: Any) -> str:
        """Convert object to JSON string, handling UUIDs and other non-serializable types."""
        def default(o):
            # Handle common non-serializable types
            if hasattr(o, '__str__'):
                return str(o)
            return f"<{type(o).__name__}>"
        
        return json.dumps(obj, default=default)
    
    def _create_standard_payload(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """Create minimal standardized payload structure."""
        return {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution": {
                "run_id": str(kwargs.get('run_id', '')),
                "parent_run_id": str(kwargs.get('parent_run_id', '')) or '',
                "custom_id": self._extract_custom_id_from_tags(kwargs),
                "tags": kwargs.get('tags', []) or [],
                "metadata": kwargs.get('metadata', {}) or {}
            },
            "data": {},  # Populated by event handlers as needed
            "raw": kwargs.copy()
        }
    
    def _add_error_info(self, payload: Dict[str, Any], error: Exception) -> None:
        """Add simple error information to payload."""
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
            'logger_custom_id': payload["execution"]["custom_id"],
            'event_type': payload["event_type"],
            'provider': self.provider,
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
        payload["raw"]["primary_args"] = primary_data
        self._log_event(payload)

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        """Log LLM start event."""
        data = {
            "prompts": prompts,
            "serialized": serialized,
            "model": serialized.get('kwargs', {}).get('model_name', ''),
            "invocation_params": serialized.get('kwargs', {}),
            "tools": kwargs.get('tools', []) or []
        }
        if 'messages' in kwargs:
            data["messages"] = kwargs['messages']
        self._handle_event('llm_start', data, **kwargs)

    def on_llm_end(self, response, **kwargs):
        """Log LLM end event."""
        response_data = self._convert_response(response)
        data = {"response": response_data}

        if hasattr(response, 'llm_output') and response.llm_output:
            data["usage"] = response.llm_output.get('token_usage', {})
            data["model"] = response.llm_output.get('model_name', '')

        if hasattr(response, 'response_metadata'):
            data["response_metadata"] = response.response_metadata

        self._handle_event('llm_end', data, **kwargs)

    def on_llm_error(self, error, **kwargs):
        """Log LLM error event."""
        if 'llm_error' not in self.event_types:
            return

        payload = self._create_standard_payload('llm_error', **kwargs)
        self._add_error_info(payload, error)
        payload["raw"]["primary_args"] = {"error": str(error)}
        self._log_event(payload)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Log chain start event."""
        self._handle_event('chain_start', {
            "inputs": inputs,
            "serialized": serialized,
            "model": serialized.get('name', '')
        }, **kwargs)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Log chain end event."""
        self._handle_event('chain_end', {"outputs": outputs}, **kwargs)

    def on_chain_error(self, error: Exception, **kwargs):
        """Log chain error event."""
        if 'chain_error' not in self.event_types:
            return

        payload = self._create_standard_payload('chain_error', **kwargs)
        self._add_error_info(payload, error)
        payload["raw"]["primary_args"] = {"error": str(error)}
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
        self._handle_event('tool_start', data, **kwargs)

    def on_tool_end(self, output: str, **kwargs):
        """Log tool end event."""
        self._handle_event('tool_end', {"output": output}, **kwargs)

    def on_tool_error(self, error: Exception, **kwargs):
        """Log tool error event."""
        if 'tool_error' not in self.event_types:
            return

        payload = self._create_standard_payload('tool_error', **kwargs)
        self._add_error_info(payload, error)
        payload["raw"]["primary_args"] = {"error": str(error)}
        self._log_event(payload)

    def on_agent_action(self, action, **kwargs):
        """Log agent action event."""
        if hasattr(action, '__dict__'):
            action_data = {
                'tool': getattr(action, 'tool', ''),
                'tool_input': getattr(action, 'tool_input', ''),
                'log': getattr(action, 'log', ''),
            }
        else:
            action_data = {'action': str(action)}

        self._handle_event('agent_action', {"action": action_data}, **kwargs)

    def on_agent_finish(self, finish, **kwargs):
        """Log agent finish event."""
        if hasattr(finish, '__dict__'):
            finish_data = {
                'return_values': getattr(finish, 'return_values', {}),
                'log': getattr(finish, 'log', ''),
            }
        else:
            finish_data = {'finish': str(finish)}

        self._handle_event('agent_finish', {"finish": finish_data}, **kwargs)
    
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
    
    def _upload_to_s3(self, table: pa.Table, relative_path: Path):
        """Upload Parquet table to S3 with retry logic.
        
        Args:
            table: PyArrow table to upload
            relative_path: Path relative to log_dir for S3 key construction
        """
        if not self.s3_bucket:
            return
        
        # Construct S3 key
        s3_key = f"{self.s3_prefix}{relative_path}"
        
        # Attempt upload with retries
        for attempt in range(self.s3_retry_attempts):
            try:
                # Write table to BytesIO buffer
                buffer = BytesIO()
                pq.write_table(table, buffer, compression='snappy')
                buffer.seek(0)
                
                # Create S3 client (uses default credential chain)
                s3_client = boto3.client('s3')
                
                # Upload to S3
                s3_client.put_object(
                    Bucket=self.s3_bucket,
                    Key=s3_key,
                    Body=buffer.getvalue()
                )
                
                return  # Success
                
            except Exception as e:
                if attempt == self.s3_retry_attempts - 1:
                    # Final attempt failed
                    error_msg = f"Failed to upload to S3 after {self.s3_retry_attempts} attempts: {e}"
                    if self.s3_on_failure == "error":
                        raise RuntimeError(error_msg) from e
                    else:
                        print(f"S3 upload failed (continuing): {error_msg}")
                        return
                
                # Exponential backoff before retry
                time.sleep(2 ** attempt)
    
    def _write_buffer(self, buffer):
        """Write buffer to Parquet file (called without lock held)."""
        try:
            # Build columns explicitly to avoid type inference and NumPy issues
            ts = pa.array([e["timestamp"] for e in buffer],
                          type=pa.timestamp("us", tz="UTC"))
            run_id = pa.array([e["run_id"] for e in buffer], type=pa.string())
            # Handle backward compatibility - entries may not have parent_run_id
            parent_run_id = pa.array([e.get("parent_run_id", "") for e in buffer], type=pa.string())
            logger_custom_id = pa.array([e["logger_custom_id"] for e in buffer], type=pa.string())
            event_type = pa.array([e["event_type"] for e in buffer], type=pa.string())
            provider = pa.array([e["provider"] for e in buffer], type=pa.string())
            logger_metadata = pa.array([e["logger_metadata"] for e in buffer], type=pa.string())
            payload = pa.array([e["payload"] for e in buffer], type=pa.string())

            # Create table with explicit schema
            table = pa.Table.from_arrays(
                [ts, run_id, parent_run_id, logger_custom_id, event_type, provider, logger_metadata, payload],
                schema=SCHEMA
            )

            # Determine output directory based on partitioning strategy
            if self.partition_on == "date":
                # Daily partitioning
                today = date.today()
                output_dir = self.log_dir / f"date={today}"
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                # No partitioning - save directly to log_dir
                output_dir = self.log_dir

            # Unique filename
            timestamp = datetime.now().strftime('%H%M%S_%f')
            filepath = output_dir / f"logs_{timestamp}.parquet"

            # Write to Parquet
            pq.write_table(table, filepath, compression='snappy')

            # Upload to S3 if configured
            if self.s3_bucket:
                # Calculate relative path for S3 key
                relative_path = filepath.relative_to(self.log_dir)
                self._upload_to_s3(table, relative_path)

        except RuntimeError:
            # Re-raise S3 errors when in error mode
            raise
        except Exception as e:
            import traceback
            print(f"Failed to write logs: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure buffer is flushed."""
        _ = (exc_type, exc_val, exc_tb)  # Unused but required by protocol
        self.flush()
        return False
    
    def __del__(self):
        """Last resort flush on garbage collection."""
        try:
            if self.buffer:
                self.flush()
        except:
            pass  # Best effort, don't raise in destructor