"""Tagging utilities for custom ID tracking in LangChain callbacks."""

from typing import Optional, Dict, Any, List
from .config import CUSTOM_ID_PREFIX


def with_tags(*additional_tags: str, custom_id: Optional[str] = None,
              custom_id_description: Optional[str] = None,
              tags: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None,
              replace_tags: bool = False) -> Dict[str, Any]:
    """
    Add custom ID and tags to config for LangChain tracking.

    This function ensures that custom IDs survive through the entire LangChain
    execution lifecycle by embedding them in the tags array with a special prefix.

    Args:
        *additional_tags: Tags as positional arguments
        custom_id: Optional ID for request tracking (will be prefixed with 'logger_custom_id:')
        custom_id_description: Human-readable description of what the custom_id represents
                               (e.g., "Customer support ticket ID", "Research question ID")
        tags: Tags as a list (alternative to positional)
        config: Existing config to extend
        replace_tags: If True, replace existing tags instead of extending (default: False)

    Returns:
        Config dict with the custom ID and tags added

    Examples:
        >>> from langchain_callback_parquet_logger import with_tags
        >>>
        >>> # Simple custom ID with description
        >>> config = with_tags(
        ...     custom_id="session-123",
        ...     custom_id_description="User session identifier"
        ... )
        >>>
        >>> # Custom ID with description and additional tags
        >>> config = with_tags(
        ...     "production", "v2",
        ...     custom_id="ticket-456",
        ...     custom_id_description="Customer support ticket ID"
        ... )
        >>>
        >>> # Just tags, no custom ID
        >>> config = with_tags("production", "experimental")
        >>>
        >>> # Extend existing config
        >>> existing = {"tags": ["test"], "metadata": {"user": "john"}}
        >>> config = with_tags(
        ...     "urgent",
        ...     custom_id="req-789",
        ...     custom_id_description="API request ID",
        ...     config=existing
        ... )
        >>> # Result: tags includes ["test", "urgent", "logger_custom_id:req-789", "custom_id_description:API request ID"]
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
        tag_list.append(f'{CUSTOM_ID_PREFIX}{custom_id}')

        # Add the description as a separate tag if provided
        if custom_id_description:
            tag_list.append(f'custom_id_description:{custom_id_description}')

    return config


def extract_custom_id(tags: List[str]) -> str:
    """
    Extract custom ID from tags list.

    Args:
        tags: List of tags that may contain a custom ID

    Returns:
        The custom ID if found, empty string otherwise
    """
    for tag in tags or []:
        if isinstance(tag, str) and tag.startswith(CUSTOM_ID_PREFIX):
            return tag[len(CUSTOM_ID_PREFIX):]  # Everything after prefix
    return ''