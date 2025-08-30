"""
Tests for background response retrieval functionality.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import pytest
import pandas as pd

from langchain_callback_parquet_logger import ParquetLogger
from langchain_callback_parquet_logger.background_retrieval import (
    retrieve_background_responses,
    save_checkpoint
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with response IDs."""
    return pd.DataFrame({
        'response_id': ['resp_001', 'resp_002', 'resp_003'],
        'logger_custom_id': ['user-001', 'user-002', 'user-003'],
        'other_data': ['data1', 'data2', 'data3']
    })


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        'id': 'resp_001',
        'object': 'response',
        'created': 1234567890,
        'model': 'gpt-4',
        'choices': [{'message': {'content': 'Test response'}}],
        'usage': {'total_tokens': 100}
    }
    
    client.responses.retrieve = AsyncMock(return_value=mock_response)
    return client


@pytest.mark.asyncio
async def test_basic_retrieval(sample_df, mock_openai_client):
    """Test basic retrieval functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ParquetLogger(log_dir=tmpdir, buffer_size=10)
        
        results = await retrieve_background_responses(
            sample_df,
            mock_openai_client,
            logger=logger,
            show_progress=False
        )
        
        # Check results DataFrame
        assert results is not None
        assert len(results) == 3
        assert 'response_id' in results.columns
        assert 'status' in results.columns
        assert 'openai_response' in results.columns
        assert all(results['status'] == 'completed')
        
        # Verify API was called for each response
        assert mock_openai_client.responses.retrieve.call_count == 3
        
        # Check logs were created
        logger.flush()
        log_files = list(Path(tmpdir).rglob('*.parquet'))
        assert len(log_files) > 0


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting with 429 errors."""
    df = pd.DataFrame({
        'response_id': ['resp_001'],
        'logger_custom_id': ['user-001']
    })
    
    # Create a mock RateLimitError if openai is available
    try:
        import openai
        rate_limit_error = openai.RateLimitError("Rate limit exceeded")
    except (ImportError, AttributeError):
        # Fallback if openai is not installed or doesn't have RateLimitError
        rate_limit_error = Exception("429 Rate limit exceeded")
    
    client = AsyncMock()
    # Simulate rate limit error then success
    client.responses.retrieve = AsyncMock(
        side_effect=[
            rate_limit_error,
            MagicMock(model_dump=lambda: {'id': 'resp_001', 'status': 'completed'})
        ]
    )
    
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        results = await retrieve_background_responses(
            df,
            client,
            max_retries=2,
            show_progress=False
        )
        
        # Should have slept due to rate limit
        mock_sleep.assert_called()
        assert results.iloc[0]['status'] == 'completed'


@pytest.mark.asyncio
async def test_checkpoint_resume(sample_df):
    """Test checkpoint save and resume functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_file = f"{tmpdir}/checkpoint.parquet"
        
        # Create a checkpoint with one processed ID
        save_checkpoint(checkpoint_file, [
            {'response_id': 'resp_001', 'processed': True, 'error': None}
        ])
        
        client = AsyncMock()
        mock_response = MagicMock(model_dump=lambda: {'id': 'resp', 'status': 'completed'})
        client.responses.retrieve = AsyncMock(return_value=mock_response)
        
        # Run retrieval with checkpoint
        results = await retrieve_background_responses(
            sample_df,
            client,
            checkpoint_file=checkpoint_file,
            show_progress=False
        )
        
        # Should only retrieve 2 responses (resp_002 and resp_003)
        assert client.responses.retrieve.call_count == 2
        
        # First response should be marked as already processed
        resp_001_result = results[results['response_id'] == 'resp_001'].iloc[0]
        assert resp_001_result['status'] == 'already_processed'


@pytest.mark.asyncio
async def test_memory_efficient_mode(sample_df, mock_openai_client):
    """Test return_results=False for memory efficiency."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ParquetLogger(log_dir=tmpdir, buffer_size=10)
        
        results = await retrieve_background_responses(
            sample_df,
            mock_openai_client,
            logger=logger,
            return_results=False,  # Don't keep results in memory
            show_progress=False
        )
        
        # Should return None
        assert results is None
        
        # But logs should still be written
        logger.flush()
        log_files = list(Path(tmpdir).rglob('*.parquet'))
        assert len(log_files) > 0
        
        # Verify all responses were retrieved
        assert mock_openai_client.responses.retrieve.call_count == 3


@pytest.mark.asyncio
async def test_partial_failures(mock_openai_client):
    """Test handling of partial failures."""
    df = pd.DataFrame({
        'response_id': ['resp_001', 'resp_002', 'resp_003'],
        'logger_custom_id': ['user-001', 'user-002', 'user-003']
    })
    
    # Mock mixed success/failure responses
    mock_openai_client.responses.retrieve = AsyncMock(
        side_effect=[
            MagicMock(model_dump=lambda: {'id': 'resp_001', 'status': 'completed'}),
            Exception("404 Not found"),
            MagicMock(model_dump=lambda: {'id': 'resp_003', 'status': 'completed'})
        ]
    )
    
    results = await retrieve_background_responses(
        df,
        mock_openai_client,
        max_retries=1,
        show_progress=False
    )
    
    # Check mixed statuses
    assert len(results) == 3
    assert results.iloc[0]['status'] == 'completed'
    assert results.iloc[1]['status'] == 'failed'
    assert results.iloc[1]['error'] == "404 Not found"
    assert results.iloc[2]['status'] == 'completed'


@pytest.mark.asyncio
async def test_missing_columns():
    """Test handling of missing columns."""
    # DataFrame missing logger_custom_id column
    df = pd.DataFrame({
        'response_id': ['resp_001']
    })
    
    client = AsyncMock()
    client.responses.retrieve = AsyncMock(
        return_value=MagicMock(model_dump=lambda: {'id': 'resp_001'})
    )
    
    # Should work with warning
    with pytest.warns(UserWarning, match="logger_custom_id"):
        results = await retrieve_background_responses(
            df,
            client,
            show_progress=False
        )
    
    assert results is not None
    assert len(results) == 1
    
    # Missing response_id column should raise error
    df_bad = pd.DataFrame({'other_column': ['data']})
    
    with pytest.raises(ValueError, match="response_id"):
        await retrieve_background_responses(
            df_bad,
            client,
            show_progress=False
        )


@pytest.mark.asyncio
async def test_timeout_handling():
    """Test timeout handling."""
    df = pd.DataFrame({
        'response_id': ['resp_001'],
        'logger_custom_id': ['user-001']
    })
    
    client = AsyncMock()
    
    # Create a coroutine that never completes (needs to accept response_id parameter)
    async def never_complete(response_id):
        await asyncio.sleep(100)
    
    client.responses.retrieve = AsyncMock(side_effect=never_complete)
    
    results = await retrieve_background_responses(
        df,
        client,
        timeout=0.1,  # Very short timeout
        max_retries=1,
        show_progress=False
    )
    
    # Should fail with timeout
    assert results.iloc[0]['status'] == 'failed'
    assert 'Timeout' in results.iloc[0]['error']


@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing with batch_size."""
    # Create larger DataFrame
    df = pd.DataFrame({
        'response_id': [f'resp_{i:03d}' for i in range(10)],
        'logger_custom_id': [f'user_{i:03d}' for i in range(10)]
    })
    
    client = AsyncMock()
    call_times = []
    
    async def track_calls(response_id):
        call_times.append(asyncio.get_event_loop().time())
        return MagicMock(model_dump=lambda: {'id': response_id})
    
    client.responses.retrieve = AsyncMock(side_effect=track_calls)
    
    await retrieve_background_responses(
        df,
        client,
        batch_size=3,  # Process 3 at a time
        show_progress=False
    )
    
    # All should be retrieved
    assert client.responses.retrieve.call_count == 10


@pytest.mark.asyncio
async def test_logging_event_types():
    """Test that correct event types are logged."""
    df = pd.DataFrame({
        'response_id': ['resp_001'],
        'logger_custom_id': ['user-001']
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ParquetLogger(log_dir=tmpdir, buffer_size=1)
        
        client = AsyncMock()
        client.responses.retrieve = AsyncMock(
            return_value=MagicMock(model_dump=lambda: {'id': 'resp_001'})
        )
        
        await retrieve_background_responses(
            df,
            client,
            logger=logger,
            show_progress=False
        )
        
        # Read logs and check event types
        log_df = pd.read_parquet(tmpdir)
        event_types = log_df['event_type'].unique()
        
        assert 'background_retrieval_attempt' in event_types
        assert 'background_retrieval_complete' in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])