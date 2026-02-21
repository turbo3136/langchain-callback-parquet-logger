"""
Tests for Batch.run() and core batch processing functionality.
"""

import asyncio
import pytest
import pandas as pd
from unittest.mock import Mock, AsyncMock
from langchain_callback_parquet_logger import Batch, with_tags
from langchain_callback_parquet_logger.config import (
    LLMConfig, ProcessingConfig, StorageConfig, ColumnConfig
)


def _make_mock_llm_class(ainvoke_side_effect=None, ainvoke_return=None):
    """Create a MockLLM class with a shared AsyncMock for ainvoke.

    Returns (MockLLM, ainvoke_mock) so tests can inspect call_count and call_args.
    """
    if ainvoke_side_effect is not None:
        ainvoke_mock = AsyncMock(side_effect=ainvoke_side_effect)
    else:
        default_return = ainvoke_return or Mock(content="response")
        ainvoke_mock = AsyncMock(return_value=default_return)

    class MockLLM:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get('callbacks', [])
            self.ainvoke = ainvoke_mock

    MockLLM.__name__ = 'MockLLM'
    MockLLM.__module__ = 'test_module'
    return MockLLM, ainvoke_mock


class TestBatch:
    """Test Batch.run() functionality (core batch processing engine)."""

    @pytest.mark.asyncio
    async def test_batch_run_basic(self, sample_dataframe, tmp_path):
        """Test basic batch processing of DataFrame."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM, ainvoke_mock = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == len(df)
        for r in results:
            assert hasattr(r, 'content') or isinstance(r, Mock)
        assert ainvoke_mock.call_count == len(df)

    @pytest.mark.asyncio
    async def test_batch_run_returns_results(self, sample_dataframe, tmp_path):
        """Test that Batch.run() returns list when return_results=True."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        MockLLM, _ = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert isinstance(results, list)
        assert len(results) == len(df)

    @pytest.mark.asyncio
    async def test_memory_efficient_mode(self, sample_dataframe, tmp_path):
        """Test return_results=False returns None."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        MockLLM, ainvoke_mock = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=False, return_results=False),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert results is None
        assert ainvoke_mock.call_count == len(df)

    @pytest.mark.asyncio
    async def test_empty_dataframe(self, tmp_path):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['prompt', 'config'])

        MockLLM, ainvoke_mock = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert results == []
        assert ainvoke_mock.call_count == 0

    @pytest.mark.asyncio
    async def test_custom_columns(self, sample_dataframe, tmp_path):
        """Test using custom column names via ColumnConfig."""
        df = sample_dataframe.copy()
        df['my_prompt'] = df['text']
        df['my_config'] = df['id'].apply(lambda x: {'custom': x})
        df['my_tools'] = [[{'type': 'test'}]] * len(df)

        MockLLM, ainvoke_mock = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
            column_config=ColumnConfig(prompt='my_prompt', config='my_config', tools='my_tools'),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == len(df)

        # Check that correct columns were used
        call_args = ainvoke_mock.call_args_list[0]
        assert call_args.kwargs['input'] == 'Hello world'  # First prompt
        assert call_args.kwargs['config'] == {'custom': 1}
        assert call_args.kwargs['tools'] == [{'type': 'test'}]

    @pytest.mark.asyncio
    async def test_progress_bar(self, sample_dataframe, tmp_path, capsys):
        """Test that print-based progress output works."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        MockLLM, ainvoke_mock = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=True, return_results=True),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == len(df)
        assert ainvoke_mock.call_count == len(df)

        captured = capsys.readouterr()
        assert "Processed" in captured.out

    @pytest.mark.asyncio
    async def test_exception_handling(self, sample_dataframe, tmp_path):
        """Test return_exceptions=True handles errors gracefully."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        call_count = 0

        async def mock_invoke_with_error(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Test error")
            response = Mock()
            response.content = f"Response {call_count}"
            return response

        MockLLM, ainvoke_mock = _make_mock_llm_class(ainvoke_side_effect=mock_invoke_with_error)
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(
                show_progress=False, return_results=True, return_exceptions=True
            ),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == len(df)
        assert isinstance(results[1], ValueError)
        assert results[0].content == "Response 1"
        assert results[2].content == "Response 3"

    @pytest.mark.asyncio
    async def test_max_concurrency(self, tmp_path):
        """Test max_concurrency limits parallel requests."""
        df = pd.DataFrame({
            'prompt': [f'Prompt {i}' for i in range(10)]
        })

        max_concurrent = 0
        current_concurrent = 0

        async def track_concurrency(**kwargs):
            nonlocal current_concurrent, max_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            response = Mock()
            response.content = "Response"
            return response

        MockLLM, _ = _make_mock_llm_class(ainvoke_side_effect=track_concurrency)
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(
                show_progress=False, return_results=True, max_concurrency=3
            ),
        )
        await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_missing_columns(self, sample_dataframe, tmp_path):
        """Test handling of missing optional columns."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        # No config or tools columns

        MockLLM, ainvoke_mock = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
            column_config=ColumnConfig(
                prompt='prompt', config='missing_config', tools='missing_tools'
            ),
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == len(df)

        # Check that config and tools were not passed (columns don't exist)
        call_args = ainvoke_mock.call_args_list[0]
        assert 'config' not in call_args.kwargs or call_args.kwargs.get('config') is None
        assert 'tools' not in call_args.kwargs or call_args.kwargs.get('tools') is None

    @pytest.mark.asyncio
    async def test_tools_column_none(self, sample_dataframe, tmp_path):
        """Test setting tools=None in ColumnConfig to skip tools."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['tools'] = [[{'type': 'test'}]] * len(df)  # Should be ignored

        MockLLM, ainvoke_mock = _make_mock_llm_class()
        batch = Batch(
            storage_config=StorageConfig(output_dir=str(tmp_path)),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
            column_config=ColumnConfig(tools=None),  # Explicitly skip tools
        )
        results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == len(df)

        # Verify tools were not passed
        call_args = ainvoke_mock.call_args_list[0]
        assert 'tools' not in call_args.kwargs
