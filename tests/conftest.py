"""
Shared fixtures and utilities for tests.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, AsyncMock
import pandas as pd


@pytest.fixture
def temp_log_dir(tmp_path):
    """Create a temporary directory for test logs."""
    log_dir = tmp_path / "test_logs"
    log_dir.mkdir(exist_ok=True)
    return str(log_dir)


@pytest.fixture
def mock_llm():
    """Create a mock LangChain LLM."""
    llm = Mock()
    llm.callbacks = []
    
    # Mock invoke method
    def mock_invoke(input_text, **kwargs):
        response = Mock()
        response.content = f"Response to: {input_text[:50]}"
        return response
    
    llm.invoke = Mock(side_effect=mock_invoke)
    
    # Mock async invoke - accept input as keyword arg since batch_run passes it that way
    async def mock_ainvoke(**kwargs):
        input_text = kwargs.get('input', '')
        response = Mock()
        response.content = f"Response to: {input_text[:50]}"
        return response
    
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    
    return llm


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'text': ['Hello world', 'Test message', 'Sample data'],
        'category': ['greeting', 'test', 'sample']
    })


@pytest.fixture
def mock_callback_events():
    """Create mock callback event data."""
    return {
        'llm_start': {
            'serialized': {'kwargs': {'model_name': 'test-model'}},
            'prompts': ['Test prompt'],
            'run_id': 'test-run-123',
            'tags': ['test-tag'],
            'metadata': {'test': 'metadata'}
        },
        'llm_end': {
            'response': Mock(
                dict=lambda: {'generations': [['Test response']]},
                llm_output={'token_usage': {'total_tokens': 10}}
            ),
            'run_id': 'test-run-123'
        },
        'llm_error': {
            'error': Exception('Test error'),
            'run_id': 'test-run-123'
        }
    }


@pytest.fixture
def mock_structured_llm():
    """Create a mock LLM with structured output."""
    llm = Mock()
    llm.callbacks = []
    
    class MockStructuredOutput:
        def __init__(self, value: str):
            self.value = value
            self.content = value
    
    async def mock_ainvoke(input_text, **kwargs):
        return MockStructuredOutput(f"Structured: {input_text[:30]}")
    
    llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)
    return llm


@pytest.fixture
def clean_imports(monkeypatch):
    """Clean up imports to ensure fresh module loading."""
    # Remove any cached imports
    import sys
    modules_to_remove = [
        key for key in sys.modules.keys() 
        if 'langchain_callback_parquet_logger' in key
    ]
    for module in modules_to_remove:
        del sys.modules[module]