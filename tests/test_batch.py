"""
Tests for batch_run helper functionality.
"""

import pytest
import pandas as pd
from unittest.mock import patch, Mock, AsyncMock
from langchain_callback_parquet_logger import batch_run, with_tags


class TestBatchRun:
    """Test batch_run functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_run_basic(self, sample_dataframe, mock_llm):
        """Test basic batch processing of DataFrame."""
        # Prepare DataFrame
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))
        
        # Run batch
        results = await batch_run(
            df, 
            mock_llm, 
            show_progress=False
        )
        
        # Verify results
        assert len(results) == len(df)
        # Check that we got mock responses
        for r in results:
            assert hasattr(r, 'content') or isinstance(r, Mock)
        assert mock_llm.ainvoke.call_count == len(df)
    
    @pytest.mark.asyncio
    async def test_batch_run_returns_results(self, sample_dataframe, mock_llm):
        """Test that batch_run returns list by default."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        
        results = await batch_run(
            df, 
            mock_llm,
            show_progress=False,
            return_results=True  # Explicit default
        )
        
        assert isinstance(results, list)
        assert len(results) == len(df)
    
    @pytest.mark.asyncio
    async def test_memory_efficient_mode(self, sample_dataframe, mock_llm):
        """Test return_results=False returns None."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        
        results = await batch_run(
            df,
            mock_llm,
            show_progress=False,
            return_results=False  # Memory-efficient mode
        )
        
        assert results is None
        # But LLM was still called
        assert mock_llm.ainvoke.call_count == len(df)
    
    @pytest.mark.asyncio
    async def test_empty_dataframe(self, mock_llm):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame(columns=['prompt', 'config'])
        
        results = await batch_run(
            df,
            mock_llm,
            show_progress=False
        )
        
        assert results == []
        assert mock_llm.ainvoke.call_count == 0
    
    @pytest.mark.asyncio
    async def test_custom_columns(self, sample_dataframe, mock_llm):
        """Test using custom column names."""
        df = sample_dataframe.copy()
        df['my_prompt'] = df['text']
        df['my_config'] = df['id'].apply(lambda x: {'custom': x})
        df['my_tools'] = [[{'type': 'test'}]] * len(df)
        
        results = await batch_run(
            df,
            mock_llm,
            prompt_col='my_prompt',
            config_col='my_config',
            tools_col='my_tools',
            show_progress=False
        )
        
        assert len(results) == len(df)
        
        # Check that correct columns were used
        call_args = mock_llm.ainvoke.call_args_list[0]
        # Check the keyword arguments
        assert call_args.kwargs['input'] == 'Hello world'  # First prompt
        assert call_args.kwargs['config'] == {'custom': 1}
        assert call_args.kwargs['tools'] == [{'type': 'test'}]
    
    @pytest.mark.asyncio
    async def test_progress_bar(self, sample_dataframe, mock_llm):
        """Test that progress bar doesn't crash."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        
        # Test with progress enabled (will try to import tqdm)
        # If tqdm is not available, it should fall back gracefully
        results = await batch_run(
            df,
            mock_llm,
            show_progress=True
        )
        
        assert len(results) == len(df)
        assert mock_llm.ainvoke.call_count == len(df)
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, sample_dataframe):
        """Test return_exceptions=True handles errors gracefully."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        
        # Create LLM that raises error on second call
        llm = Mock()
        call_count = 0
        
        async def mock_invoke_with_error(**kwargs):  # Accept keyword args
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Test error")
            response = Mock()
            response.content = f"Response {call_count}"
            return response
        
        llm.ainvoke = AsyncMock(side_effect=mock_invoke_with_error)
        
        # Run with return_exceptions=True
        results = await batch_run(
            df,
            llm,
            show_progress=False,
            return_exceptions=True
        )
        
        assert len(results) == len(df)
        assert isinstance(results[1], ValueError)
        assert results[0].content == "Response 1"
        assert results[2].content == "Response 3"
    
    @pytest.mark.asyncio
    async def test_max_concurrency(self, mock_llm):
        """Test max_concurrency limits parallel requests."""
        # Create larger DataFrame
        df = pd.DataFrame({
            'prompt': [f'Prompt {i}' for i in range(10)]
        })
        
        # Track concurrent calls
        concurrent_calls = []
        max_concurrent = 0
        current_concurrent = 0
        
        async def track_concurrency(input_text, **kwargs):
            nonlocal current_concurrent, max_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            # Simulate some work
            import asyncio
            await asyncio.sleep(0.01)
            current_concurrent -= 1
            response = Mock()
            response.content = "Response"
            return response
        
        llm = Mock()
        llm.ainvoke = AsyncMock(side_effect=track_concurrency)
        
        await batch_run(
            df,
            llm,
            max_concurrency=3,
            show_progress=False
        )
        
        # Max concurrency should not exceed limit
        assert max_concurrent <= 3
    
    @pytest.mark.asyncio
    async def test_missing_columns(self, sample_dataframe, mock_llm):
        """Test handling of missing optional columns."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        # No config or tools columns
        
        results = await batch_run(
            df,
            mock_llm,
            config_col='missing_config',
            tools_col='missing_tools',
            show_progress=False
        )
        
        # Should still work
        assert len(results) == len(df)
        
        # Check that None was passed for missing columns
        call_args = mock_llm.ainvoke.call_args_list[0]
        assert 'config' not in call_args[1] or call_args[1].get('config') is None
        assert 'tools' not in call_args[1] or call_args[1].get('tools') is None
    
    @pytest.mark.asyncio
    async def test_tools_column_none(self, sample_dataframe, mock_llm):
        """Test setting tools_col=None to skip tools."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['tools'] = [[{'type': 'test'}]] * len(df)  # This should be ignored
        
        results = await batch_run(
            df,
            mock_llm,
            tools_col=None,  # Explicitly skip tools
            show_progress=False
        )
        
        assert len(results) == len(df)
        
        # Verify tools were not passed
        call_args = mock_llm.ainvoke.call_args_list[0]
        assert 'tools' not in call_args[1]