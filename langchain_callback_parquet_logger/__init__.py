# langchain_parquet_logger/__init__.py
from .logger import ParquetLogger
from typing import Optional, Dict, Any, List

__version__ = "0.5.0"


def with_tags(*additional_tags: str, custom_id: Optional[str] = None,
              tags: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None,
              replace_tags: bool = False) -> Dict[str, Any]:
    """
    Add custom ID and tags to config for LangChain tracking.
    
    Args:
        *additional_tags: Tags as positional arguments
        custom_id: Optional ID for request tracking (will be prefixed with 'logger_custom_id:')
        tags: Tags as a list (alternative to positional)
        config: Existing config to extend
        replace_tags: If True, replace existing tags instead of extending (default: False)
        
    Returns:
        Config dict with the custom ID and tags added
        
    Examples:
        >>> from langchain_callback_parquet_logger import with_tags
        >>> 
        >>> # Simple custom ID
        >>> config = with_tags(custom_id="session-123")
        >>> 
        >>> # Custom ID with additional tags (positional)
        >>> config = with_tags("production", "v2", custom_id="session-123")
        >>> 
        >>> # Just tags, no custom ID
        >>> config = with_tags("production", "experimental")
        >>> 
        >>> # Custom ID with tags as list
        >>> config = with_tags(custom_id="session-123", tags=["production", "v2"])
        >>> 
        >>> # Extend existing config
        >>> existing = {"tags": ["test"], "metadata": {"user": "john"}}
        >>> config = with_tags("urgent", custom_id="req-456", config=existing)
        >>> # Result: tags = ["test", "urgent", "logger_custom_id:req-456"]
        >>> 
        >>> # Replace existing tags
        >>> config = with_tags("new", custom_id="req-789", config=existing, replace_tags=True)
        >>> # Result: tags = ["new", "logger_custom_id:req-789"]
    """
    config = config or {}
    
    if replace_tags:
        tag_list = []
        config['tags'] = tag_list
    else:
        tag_list = config.setdefault('tags', [])
    
    # Add positional tags
    if additional_tags:
        tag_list.extend(additional_tags)
    
    # Add list tags
    if tags:
        tag_list.extend(tags)
    
    # Add custom ID if provided
    if custom_id:
        tag_list.append(f'logger_custom_id:{custom_id}')
    
    return config


# Import batch processing helper
try:
    from .batch_helpers import batch_run
    _batch_helpers_available = True
except ImportError:
    # Batch helper is optional
    _batch_helpers_available = False
    batch_run = None

# Import background retrieval helper
try:
    from .background_retrieval import retrieve_background_responses
    _background_retrieval_available = True
except ImportError:
    # Background retrieval is optional (requires openai)
    _background_retrieval_available = False
    retrieve_background_responses = None

# Define exports
__all__ = ['ParquetLogger', 'with_tags', '__version__']

# Add batch helper to exports if available
if _batch_helpers_available:
    __all__.append('batch_run')

# Add background retrieval to exports if available
if _background_retrieval_available:
    __all__.append('retrieve_background_responses')