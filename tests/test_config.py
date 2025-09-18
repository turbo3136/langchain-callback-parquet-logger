"""
Tests for config module, especially LLMConfig functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_callback_parquet_logger.config import LLMConfig


class TestLLMConfig:
    """Test LLMConfig functionality."""

    def test_create_llm_with_callbacks(self):
        """Test that create_llm properly passes callbacks to LLM constructor."""

        # Create a mock LLM class
        class MockLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Create LLMConfig
        config = LLMConfig(
            llm_class=MockLLM,
            llm_kwargs={'model': 'test-model', 'temperature': 0.7}
        )

        # Create mock callbacks
        mock_callback = Mock()

        # Create LLM with callbacks
        llm = config.create_llm(callbacks=[mock_callback])

        # Verify callbacks were passed to constructor
        assert 'callbacks' in llm.kwargs
        assert llm.kwargs['callbacks'] == [mock_callback]
        assert llm.kwargs['model'] == 'test-model'
        assert llm.kwargs['temperature'] == 0.7

    def test_create_llm_without_callbacks(self):
        """Test that create_llm works without callbacks (backward compatibility)."""

        class MockLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        config = LLMConfig(
            llm_class=MockLLM,
            llm_kwargs={'model': 'test-model'}
        )

        # Create LLM without callbacks
        llm = config.create_llm()

        # Verify no callbacks key in kwargs
        assert 'callbacks' not in llm.kwargs
        assert llm.kwargs['model'] == 'test-model'

    def test_create_llm_with_structured_output_and_callbacks(self):
        """Test that callbacks are preserved when using structured output."""

        from pydantic import BaseModel

        class TestSchema(BaseModel):
            answer: str

        class MockLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def with_structured_output(self, schema):
                """Simulate RunnableSequence wrapping."""
                class MockRunnableSequence:
                    def __init__(self, base_llm, schema):
                        self.base_llm = base_llm
                        self.schema = schema
                        # RunnableSequence doesn't have callbacks attribute
                        # but uses the base LLM's callbacks

                return MockRunnableSequence(self, schema)

        config = LLMConfig(
            llm_class=MockLLM,
            llm_kwargs={'model': 'test-model'},
            structured_output=TestSchema
        )

        mock_callback = Mock()
        llm = config.create_llm(callbacks=[mock_callback])

        # Verify result is a RunnableSequence-like object
        assert hasattr(llm, 'base_llm')
        assert hasattr(llm, 'schema')
        assert llm.schema == TestSchema

        # Verify callbacks were passed to the base LLM
        assert 'callbacks' in llm.base_llm.kwargs
        assert llm.base_llm.kwargs['callbacks'] == [mock_callback]

    def test_create_llm_with_model_kwargs(self):
        """Test that model_kwargs are properly handled with callbacks."""

        class MockLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        config = LLMConfig(
            llm_class=MockLLM,
            llm_kwargs={'model': 'test-model'},
            model_kwargs={'top_p': 0.9, 'frequency_penalty': 0.5}
        )

        mock_callback = Mock()
        llm = config.create_llm(callbacks=[mock_callback])

        # Verify both model_kwargs and callbacks are present
        assert 'callbacks' in llm.kwargs
        assert llm.kwargs['callbacks'] == [mock_callback]
        assert 'model_kwargs' in llm.kwargs
        assert llm.kwargs['model_kwargs'] == {'top_p': 0.9, 'frequency_penalty': 0.5}