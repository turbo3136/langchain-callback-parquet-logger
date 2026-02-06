"""Test that raw section captures complete data."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from langchain_callback_parquet_logger import ParquetLogger


class TestRawDataCapture:
    """Test complete data capture in raw section."""

    def test_llm_raw_captures_everything(self):
        """Test that raw section captures all positional and keyword arguments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create logger with small buffer for immediate write
            logger = ParquetLogger(temp_dir, buffer_size=1)

            # Create test data
            serialized = {
                '_type': 'openai-chat',
                'kwargs': {'model_name': 'gpt-4', 'temperature': 0.7}
            }
            prompts = ["Test prompt 1", "Test prompt 2"]
            run_id = "test-run-123"
            parent_run_id = "parent-456"
            tags = ["test", "logger_custom_id:custom-789"]
            metadata = {"user": "test_user"}

            # Simulate LLM start
            logger.on_llm_start(
                serialized,
                prompts,
                run_id=run_id,
                parent_run_id=parent_run_id,
                tags=tags,
                metadata=metadata,
                extra_kwarg="extra_value"
            )

            # Force flush
            logger.flush()

            # Read the logged data
            df = pd.read_parquet(temp_dir)
            assert len(df) == 1

            # Parse the payload
            payload = json.loads(df.iloc[0]['payload'])

            # Check that data section has structured info
            assert payload['data']['prompts'] == prompts
            assert payload['data']['llm_type'] == 'openai-chat'

            # Check that raw section has EVERYTHING
            raw = payload['raw']
            assert raw['serialized'] == serialized  # Positional arg captured
            assert raw['prompts'] == prompts  # Positional arg captured
            assert raw['run_id'] == run_id  # Kwarg captured
            assert raw['parent_run_id'] == parent_run_id  # Kwarg captured
            assert raw['tags'] == tags  # Kwarg captured
            assert raw['metadata'] == metadata  # Kwarg captured
            assert raw['extra_kwarg'] == "extra_value"  # Extra kwarg captured

    def test_llm_end_raw_captures_response(self):
        """Test that raw section captures complete response object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create logger with small buffer
            logger = ParquetLogger(temp_dir, buffer_size=1)

            # Create mock response - use spec to prevent auto-creation of attributes
            response = Mock(spec=['generations', 'llm_output', 'model_dump'])
            response.generations = [["Generated text"]]
            response.llm_output = {
                'token_usage': {'total_tokens': 100},
                'model_name': 'gpt-4'
            }
            # The model_dump method will be called by _serialize_any
            response.model_dump.return_value = {
                'generations': [['Generated text']],
                'llm_output': {'token_usage': {'total_tokens': 100}}
            }

            # Simulate LLM end
            logger.on_llm_end(
                response,
                run_id="test-run-456",
                custom_end_param="custom_value"
            )

            # Force flush
            logger.flush()

            # Read and verify
            df = pd.read_parquet(temp_dir)
            payload = json.loads(df.iloc[0]['payload'])

            # Check raw contains serialized response
            assert 'response' in payload['raw']
            # The response is serialized via model_dump() method
            assert payload['raw']['response'] == {
                'generations': [['Generated text']],
                'llm_output': {'token_usage': {'total_tokens': 100}}
            }
            assert payload['raw']['custom_end_param'] == "custom_value"

    def test_error_raw_captures_exception(self):
        """Test that raw section captures exception details."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = ParquetLogger(temp_dir, buffer_size=1)

            # Create test exception
            error = ValueError("Test error message")
            error.custom_attr = "custom_error_data"

            # Simulate error
            logger.on_llm_error(
                error,
                run_id="error-run-123",
                error_context="test_context"
            )

            logger.flush()

            # Read and verify
            df = pd.read_parquet(temp_dir)
            payload = json.loads(df.iloc[0]['payload'])

            # Check raw contains error info
            assert 'error' in payload['raw']
            assert payload['raw']['error_context'] == "test_context"
