# langchain_parquet_logger/logger.py
import json
import threading
import time  # Explicit import to ensure it's available for PyArrow
import sys  # Ensure sys is available
import atexit  # For automatic cleanup on exit
import warnings
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional, Literal, Set
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.callbacks import BaseCallbackHandler

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
                 event_types: Optional[List[str]] = None):
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
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.provider = provider
        self.logger_metadata = logger_metadata or {}
        self.partition_on = partition_on
        
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
        
        # Check if in notebook environment and warn if using default buffer_size
        if self._is_notebook() and buffer_size == 100:
            warnings.warn(
                "\n⚠️  Notebook environment detected with default buffer_size=100.\n"
                "   Logs will only write to disk after 100 LLM calls.\n"
                "   \n"
                "   For immediate writes in notebooks, use one of:\n"
                "   \n"
                "   Option 1 - Context manager (recommended):\n"
                "       with ParquetLogger('./logs') as logger:\n"
                "           llm.callbacks = [logger]\n"
                "           response = llm.invoke('your prompt')\n"
                "   \n"
                "   Option 2 - Small buffer:\n"
                "       logger = ParquetLogger('./logs', buffer_size=1)\n"
                "   \n"
                "   Option 3 - Manual flush:\n"
                "       logger.flush()  # Call after your LLM operations\n",
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
    
    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        """Log LLM start event with all request data in payload."""
        if 'llm_start' not in self.event_types:
            return
            
        # Extract logger_custom_id from tags (persists through all events)
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'model': serialized.get('kwargs', {}).get('model_name', 'unknown'),
            'prompts': prompts,
            'serialized': serialized,
            'metadata': kwargs,
            # Include specific fields that might be present
            'invocation_params': serialized.get('kwargs', {}),
            'tags': kwargs.get('tags', []),
            'metadata_dict': kwargs.get('metadata', {}),
            'tools': kwargs.get('tools', None),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'llm_start',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_llm_end(self, response, **kwargs):
        """Log LLM completion event with all response data in payload."""
        if 'llm_end' not in self.event_types:
            return
            
        # Extract logger_custom_id from tags (persists through all events)
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        # Convert response to dict (handles all LangChain response types)
        try:
            if hasattr(response, 'dict'):
                response_data = response.dict()
            elif hasattr(response, 'to_dict'):
                response_data = response.to_dict()
            else:
                response_data = {'content': str(response)}
        except Exception as e:
            response_data = {'content': str(response), 'conversion_error': str(e)}
        
        # Extract key fields if available
        payload_data = {
            'response': response_data,
            'metadata': kwargs,
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
            'tags': kwargs.get('tags', []),
        }
        
        # Add token usage if available
        if hasattr(response, 'llm_output') and response.llm_output:
            payload_data['usage'] = response.llm_output.get('token_usage', {})
            payload_data['model_name'] = response.llm_output.get('model_name', '')
        
        # Add response metadata if available
        if hasattr(response, 'response_metadata'):
            payload_data['response_metadata'] = response.response_metadata
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'llm_end',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_llm_error(self, error, **kwargs):
        """Log LLM error event with error details in payload."""
        if 'llm_error' not in self.event_types:
            return
            
        # Extract logger_custom_id from tags (persists through all events)
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'error': str(error),
            'error_type': type(error).__name__,
            'metadata': kwargs,
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
            'tags': kwargs.get('tags', []),
        }
        
        # Add error details if available
        if hasattr(error, '__dict__'):
            payload_data['error_details'] = {k: str(v) for k, v in error.__dict__.items() 
                                            if not k.startswith('_')}
        
        # Add traceback if available
        if hasattr(error, '__traceback__'):
            import traceback
            payload_data['traceback'] = traceback.format_tb(error.__traceback__)
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'llm_error',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Log chain start event."""
        if 'chain_start' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'name': serialized.get('name', 'unknown'),
            'inputs': inputs,
            'serialized': serialized,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'chain_start',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Log chain end event."""
        if 'chain_end' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'outputs': outputs,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'chain_end',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_chain_error(self, error: Exception, **kwargs):
        """Log chain error event."""
        if 'chain_error' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'error': str(error),
            'error_type': type(error).__name__,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        # Add error details if available
        if hasattr(error, '__dict__'):
            payload_data['error_details'] = {k: str(v) for k, v in error.__dict__.items() 
                                            if not k.startswith('_')}
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'chain_error',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Log tool start event."""
        if 'tool_start' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'name': serialized.get('name', 'unknown'),
            'description': serialized.get('description', ''),
            'input': input_str,
            'serialized': serialized,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'tool_start',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_tool_end(self, output: str, **kwargs):
        """Log tool end event."""
        if 'tool_end' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'output': output,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'tool_end',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Log tool error event."""
        if 'tool_error' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        payload_data = {
            'error': str(error),
            'error_type': type(error).__name__,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        # Add error details if available
        if hasattr(error, '__dict__'):
            payload_data['error_details'] = {k: str(v) for k, v in error.__dict__.items() 
                                            if not k.startswith('_')}
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'tool_error',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_agent_action(self, action, **kwargs):
        """Log agent action event."""
        if 'agent_action' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        # Handle AgentAction object
        if hasattr(action, '__dict__'):
            action_data = {
                'tool': getattr(action, 'tool', ''),
                'tool_input': getattr(action, 'tool_input', ''),
                'log': getattr(action, 'log', ''),
            }
        else:
            action_data = str(action)
        
        payload_data = {
            'action': action_data,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'agent_action',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def on_agent_finish(self, finish, **kwargs):
        """Log agent finish event."""
        if 'agent_finish' not in self.event_types:
            return
            
        logger_custom_id = self._extract_custom_id_from_tags(kwargs)
        
        # Handle AgentFinish object
        if hasattr(finish, '__dict__'):
            finish_data = {
                'return_values': getattr(finish, 'return_values', {}),
                'log': getattr(finish, 'log', ''),
            }
        else:
            finish_data = str(finish)
        
        payload_data = {
            'finish': finish_data,
            'metadata': kwargs,
            'tags': kwargs.get('tags', []),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else None,
        }
        
        entry = {
            'timestamp': datetime.now(timezone.utc),
            'run_id': str(kwargs.get('run_id', '')),
            'parent_run_id': str(kwargs.get('parent_run_id', '')) if kwargs.get('parent_run_id') else '',
            'logger_custom_id': logger_custom_id,
            'event_type': 'agent_finish',
            'provider': self.provider,
            'logger_metadata': self.logger_metadata_json,
            'payload': self._safe_json_dumps(payload_data)
        }
        self._add_entry(entry)
    
    def _add_entry(self, entry):
        """Add entry to buffer and flush if needed."""
        with self.lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                self._flush()
    
    def flush(self):
        """Manual flush of the buffer."""
        with self.lock:
            self._flush()
    
    def _flush(self):
        """Write buffer to Parquet file."""
        if not self.buffer:
            return
        
        try:
            # Build columns explicitly to avoid type inference and NumPy issues
            ts = pa.array([e["timestamp"] for e in self.buffer],
                          type=pa.timestamp("us", tz="UTC"))
            run_id = pa.array([e["run_id"] for e in self.buffer], type=pa.string())
            parent_run_id = pa.array([e["parent_run_id"] for e in self.buffer], type=pa.string())
            logger_custom_id = pa.array([e["logger_custom_id"] for e in self.buffer], type=pa.string())
            event_type = pa.array([e["event_type"] for e in self.buffer], type=pa.string())
            provider = pa.array([e["provider"] for e in self.buffer], type=pa.string())
            logger_metadata = pa.array([e["logger_metadata"] for e in self.buffer], type=pa.string())
            payload = pa.array([e["payload"] for e in self.buffer], type=pa.string())
            
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
                output_dir.mkdir(exist_ok=True)
            else:
                # No partitioning - save directly to log_dir
                output_dir = self.log_dir
            
            # Unique filename
            timestamp = datetime.now().strftime('%H%M%S_%f')
            filepath = output_dir / f"logs_{timestamp}.parquet"
            
            # Write to Parquet
            pq.write_table(table, filepath, compression='snappy')
            self.buffer = []
        except Exception as e:
            import traceback
            print(f"Failed to write logs: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure buffer is flushed."""
        self.flush()
        return False
    
    def __del__(self):
        """Last resort flush on garbage collection."""
        try:
            if self.buffer:
                self.flush()
        except:
            pass  # Best effort, don't raise in destructor