# langchain_parquet_logger/__init__.py
from .logger import ParquetLogger
from typing import Optional, Dict, Any

__version__ = "0.1.6"


def with_custom_id(custom_id: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Helper function to add a custom ID to the config for tracking.
    
    Args:
        custom_id: The custom ID to track this LLM call
        config: Optional existing config dict to extend
        
    Returns:
        Config dict with the custom ID added as a tag
        
    Example:
        >>> from langchain_callback_parquet_logger import with_custom_id
        >>> response = llm.invoke("What's the weather?", config=with_custom_id("weather-miami"))
    """
    config = config or {}
    config.setdefault('tags', []).append(f'logger_custom_id:{custom_id}')
    return config


__all__ = ['ParquetLogger', 'with_custom_id', '__version__']