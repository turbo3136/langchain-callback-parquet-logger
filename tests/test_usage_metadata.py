"""Tests for usage_metadata and response_metadata extraction."""

import json
from unittest.mock import Mock, MagicMock
import pytest

from langchain_callback_parquet_logger import ParquetLogger


class TestUsageMetadataExtraction:
    """Test extraction of usage_metadata from nested AIMessage objects."""

    def test_extract_usage_metadata_from_response(self, tmp_path):
        """Test that usage_metadata is extracted from AIMessage in generations."""
        logger = ParquetLogger(str(tmp_path))

        # Create a mock response with nested AIMessage containing usage_metadata
        mock_message = Mock()
        mock_message.usage_metadata = {
            'input_tokens': 10,
            'output_tokens': 20,
            'total_tokens': 30,
            'input_token_details': {'cache_read': 0},
            'output_token_details': {'reasoning': 5}
        }
        mock_message.response_metadata = {
            'model': 'gpt-4o-mini',
            'finish_reason': 'stop'
        }

        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        mock_response.llm_output = None

        # Call on_llm_end
        logger.on_llm_end(
            mock_response,
            run_id='test-run-id',
            parent_run_id='parent-id'
        )

        # Flush and check the logged data
        logger.flush()

        # Read the parquet file and verify
        import pandas as pd
        df = pd.read_parquet(str(tmp_path))

        assert len(df) == 1
        event = json.loads(df.iloc[0]['payload'])

        # Check that usage_metadata was captured in data section
        assert 'usage_metadata' in event['data']
        assert event['data']['usage_metadata']['input_tokens'] == 10
        assert event['data']['usage_metadata']['output_tokens'] == 20
        assert event['data']['usage_metadata']['total_tokens'] == 30

        # Check that response_metadata was captured
        assert 'response_metadata' in event['data']
        assert event['data']['response_metadata']['model'] == 'gpt-4o-mini'

    def test_serialize_any_preserves_usage_metadata(self, tmp_path):
        """Test that _serialize_any properly handles LLMResult with nested AIMessage."""
        logger = ParquetLogger(str(tmp_path))

        # Create a mock AIMessage with model_dump method
        mock_message = MagicMock()
        mock_message.model_dump.return_value = {
            'content': 'test content',
            'usage_metadata': {
                'input_tokens': 15,
                'output_tokens': 25,
                'total_tokens': 40
            },
            'response_metadata': {'model': 'test-model'},
            'type': 'ai'
        }

        # Create mock generation and response
        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_response = Mock()
        mock_response.__class__.__name__ = 'LLMResult'
        mock_response.generations = [[mock_generation]]
        mock_response.model_dump = Mock(return_value={
            'generations': [[{'message': {'content': 'test', 'type': 'ai'}}]],
            'llm_output': None
        })

        # Test _serialize_any
        result = logger._serialize_any(mock_response)

        # Verify that the message was re-serialized with model_dump
        assert result['generations'][0][0]['message']['usage_metadata'] == {
            'input_tokens': 15,
            'output_tokens': 25,
            'total_tokens': 40
        }

        # Verify model_dump was called on the message
        mock_message.model_dump.assert_called_once()

    def test_handles_missing_usage_metadata_gracefully(self, tmp_path):
        """Test that missing usage_metadata doesn't break logging."""
        logger = ParquetLogger(str(tmp_path))

        # Create a mock response without usage_metadata
        mock_message = Mock()
        mock_message.usage_metadata = None  # No usage metadata
        mock_message.response_metadata = {'model': 'test'}

        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        mock_response.llm_output = None

        # This should not raise an error
        logger.on_llm_end(
            mock_response,
            run_id='test-run-id'
        )

        logger.flush()

        # Verify the event was logged without usage_metadata
        import pandas as pd
        df = pd.read_parquet(str(tmp_path))

        assert len(df) == 1
        event = json.loads(df.iloc[0]['payload'])

        # Should have response_metadata but not usage_metadata
        assert 'usage_metadata' not in event['data']
        assert 'response_metadata' in event['data']

    def test_handles_malformed_generations_gracefully(self, tmp_path):
        """Test that malformed generations structure doesn't break logging."""
        logger = ParquetLogger(str(tmp_path))

        # Test with empty generations
        mock_response = Mock()
        mock_response.generations = []
        mock_response.llm_output = None

        # Should not raise an error
        logger.on_llm_end(mock_response, run_id='test-1')

        # Test with None generations
        mock_response.generations = None
        logger.on_llm_end(mock_response, run_id='test-2')

        # Test with mock object that can't be subscripted (simulating test mocks)
        mock_response.generations = Mock()
        logger.on_llm_end(mock_response, run_id='test-3')

        logger.flush()

        # All three events should be logged
        import pandas as pd
        df = pd.read_parquet(str(tmp_path))
        assert len(df) == 3

    def test_both_response_and_message_metadata_captured(self, tmp_path):
        """Test that both top-level response_metadata and message-level metadata are captured."""
        logger = ParquetLogger(str(tmp_path))

        # Create response with metadata at both levels
        mock_message = Mock()
        mock_message.usage_metadata = {'input_tokens': 10}
        mock_message.response_metadata = {'message_level': 'data'}

        mock_generation = Mock()
        mock_generation.message = mock_message

        mock_response = Mock()
        mock_response.generations = [[mock_generation]]
        mock_response.response_metadata = {'response_level': 'data'}  # Top-level metadata
        mock_response.llm_output = {'token_usage': {'total': 50}}  # Legacy format

        logger.on_llm_end(mock_response, run_id='test')
        logger.flush()

        # Check the logged data
        import pandas as pd
        df = pd.read_parquet(str(tmp_path))
        event = json.loads(df.iloc[0]['payload'])

        # Should have all three sources of metadata
        assert event['data']['usage'] == {'total': 50}  # From llm_output
        assert event['data']['response_metadata']['message_level'] == 'data'  # From message
        assert event['data']['usage_metadata']['input_tokens'] == 10  # From message
