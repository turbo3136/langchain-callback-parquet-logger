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

import pyarrow as pa
import pyarrow.parquet as pq

from langchain_callback_parquet_logger import ParquetLogger
from langchain_callback_parquet_logger.background_retrieval import (
    retrieve_background_responses,
    save_checkpoint,
    _query_pending_responses,
    _BACKGROUND_EVENT_TYPES,
    _TERMINAL_RETRIEVAL_EVENT_TYPES,
)
from langchain_callback_parquet_logger.config import StorageConfig, JobConfig
from langchain_callback_parquet_logger.storage import LocalStorage
from langchain_callback_parquet_logger.logger import SCHEMA


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with response IDs."""
    return pd.DataFrame({
        'response_id': ['resp_001', 'resp_002', 'resp_003'],
        'custom_id': ['user-001', 'user-002', 'user-003'],
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
    import sys
    from unittest.mock import patch
    
    df = pd.DataFrame({
        'response_id': ['resp_001'],
        'custom_id': ['user-001']
    })
    
    client = AsyncMock()
    
    # Test with proper openai.RateLimitError if available
    try:
        import openai
        # Mock the openai module in background_retrieval
        with patch.object(sys.modules['langchain_callback_parquet_logger.background_retrieval'], 'openai', openai):
            # Create a mock response object for RateLimitError
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {}
            rate_limit_error = openai.RateLimitError(
                message="Rate limit exceeded",
                response=mock_response,
                body={"error": {"message": "Rate limit exceeded"}}
            )
            
            # Simulate rate limit error on first call, then success
            client.responses.retrieve = AsyncMock(
                side_effect=[
                    rate_limit_error,
                    MagicMock(model_dump=lambda **kwargs: {'id': 'resp_001', 'status': 'completed'})
                ]
            )
            
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                results = await retrieve_background_responses(
                    df,
                    client,
                    max_retries=3,
                    show_progress=False
                )
                
                # Sleep should be called for RateLimitError
                mock_sleep.assert_called()
                assert results.iloc[0]['status'] == 'completed'
                
    except (ImportError, AttributeError, TypeError):
        # If openai is not available, test with generic exception
        rate_limit_error = Exception("Server error")
        
        # Simulate error on first call, then success
        client.responses.retrieve = AsyncMock(
            side_effect=[
                rate_limit_error,
                MagicMock(model_dump=lambda **kwargs: {'id': 'resp_001', 'status': 'completed'})
            ]
        )
        
        results = await retrieve_background_responses(
            df,
            client,
            max_retries=3,
            show_progress=False
        )
        
        # The generic exception won't be retried, so it should fail
        assert results.iloc[0]['status'] == 'failed'


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
        mock_response = MagicMock(model_dump=lambda **kwargs: {'id': 'resp', 'status': 'completed'})
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
        'custom_id': ['user-001', 'user-002', 'user-003']
    })
    
    # Mock mixed success/failure responses
    mock_openai_client.responses.retrieve = AsyncMock(
        side_effect=[
            MagicMock(model_dump=lambda **kwargs: {'id': 'resp_001', 'status': 'completed'}),
            Exception("404 Not found"),
            MagicMock(model_dump=lambda **kwargs: {'id': 'resp_003', 'status': 'completed'})
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
    # DataFrame missing custom_id column
    df = pd.DataFrame({
        'response_id': ['resp_001']
    })
    
    client = AsyncMock()
    client.responses.retrieve = AsyncMock(
        return_value=MagicMock(model_dump=lambda **kwargs: {'id': 'resp_001'})
    )
    
    # Should work with warning
    with pytest.warns(UserWarning, match="custom_id"):
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
        'custom_id': ['user-001']
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
        'custom_id': [f'user_{i:03d}' for i in range(10)]
    })
    
    client = AsyncMock()
    call_times = []
    
    async def track_calls(response_id):
        call_times.append(asyncio.get_event_loop().time())
        return MagicMock(model_dump=lambda **kwargs: {'id': response_id})
    
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
        'custom_id': ['user-001']
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ParquetLogger(log_dir=tmpdir, buffer_size=1)
        
        client = AsyncMock()
        client.responses.retrieve = AsyncMock(
            return_value=MagicMock(model_dump=lambda **kwargs: {'id': 'resp_001'})
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


@pytest.mark.asyncio
async def test_schema_consistency():
    """run_id should equal response_id and payload should have execution wrapper."""
    df = pd.DataFrame({
        'response_id': ['resp_schema_001'],
        'custom_id': ['schema-test-001']
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ParquetLogger(log_dir=tmpdir, buffer_size=1)

        client = AsyncMock()
        client.responses.retrieve = AsyncMock(
            return_value=MagicMock(model_dump=lambda **kwargs: {'id': 'resp_schema_001', 'status': 'completed'})
        )

        await retrieve_background_responses(
            df,
            client,
            logger=logger,
            show_progress=False
        )

        log_df = pd.read_parquet(tmpdir)

        # Verify standard SCHEMA columns are present
        for col in ['timestamp', 'run_id', 'parent_run_id', 'custom_id', 'event_type', 'logger_metadata', 'payload']:
            assert col in log_df.columns, f"Missing column: {col}"

        # run_id must equal response_id (not empty string)
        bg_rows = log_df[log_df['event_type'].isin(_BACKGROUND_EVENT_TYPES)]
        assert all(bg_rows['run_id'] == 'resp_schema_001'), \
            f"Expected run_id='resp_schema_001', got: {bg_rows['run_id'].unique()}"

        # custom_id must be set
        assert all(bg_rows['custom_id'] == 'schema-test-001')

        # payload must have execution wrapper with run_id, custom_id, and data section
        for _, row in bg_rows.iterrows():
            payload = json.loads(row['payload'])
            assert 'execution' in payload, "payload missing 'execution' section"
            assert 'data' in payload, "payload missing 'data' section"
            assert 'raw' in payload, "payload missing 'raw' section"
            assert payload['execution']['run_id'] == 'resp_schema_001'
            assert payload['execution']['custom_id'] == 'schema-test-001'
            assert payload['execution']['parent_run_id'] == ''


@pytest.mark.asyncio
async def test_auto_discovery_from_storage():
    """Auto-discovery should find pending responses and skip completed ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write some background retrieval events to simulate prior state:
        #   resp_pending  → latest event is background_retrieval_attempt (non-terminal)
        #   resp_done     → latest event is background_retrieval_complete (terminal)
        logger = ParquetLogger(log_dir=tmpdir, buffer_size=10)
        from datetime import timezone

        # Pending response: only an attempt event logged
        logger._add_entry({
            'timestamp': datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            'run_id': 'resp_pending',
            'parent_run_id': '',
            'custom_id': 'cid-pending',
            'event_type': 'background_retrieval_attempt',
            'logger_metadata': '{}',
            'payload': json.dumps({
                'event_type': 'background_retrieval_attempt',
                'timestamp': '2026-01-01T00:00:00+00:00',
                'execution': {'run_id': 'resp_pending', 'parent_run_id': '', 'custom_id': 'cid-pending', 'tags': [], 'metadata': {}},
                'data': {'response_id': 'resp_pending', 'attempt_time': '2026-01-01T00:00:00+00:00'},
                'raw': {}
            })
        })

        # Completed response: attempt then complete events logged
        logger._add_entry({
            'timestamp': datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            'run_id': 'resp_done',
            'parent_run_id': '',
            'custom_id': 'cid-done',
            'event_type': 'background_retrieval_attempt',
            'logger_metadata': '{}',
            'payload': json.dumps({
                'event_type': 'background_retrieval_attempt',
                'timestamp': '2026-01-01T00:00:00+00:00',
                'execution': {'run_id': 'resp_done', 'parent_run_id': '', 'custom_id': 'cid-done', 'tags': [], 'metadata': {}},
                'data': {'response_id': 'resp_done', 'attempt_time': '2026-01-01T00:00:00+00:00'},
                'raw': {}
            })
        })
        logger._add_entry({
            'timestamp': datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc),
            'run_id': 'resp_done',
            'parent_run_id': '',
            'custom_id': 'cid-done',
            'event_type': 'background_retrieval_complete',
            'logger_metadata': '{}',
            'payload': json.dumps({
                'event_type': 'background_retrieval_complete',
                'timestamp': '2026-01-01T00:01:00+00:00',
                'execution': {'run_id': 'resp_done', 'parent_run_id': '', 'custom_id': 'cid-done', 'tags': [], 'metadata': {}},
                'data': {'response_id': 'resp_done', 'status': 'completed', 'poll_attempts': 1},
                'raw': {}
            })
        })
        logger.flush()

        # Mock client returns completed for the pending response
        client = AsyncMock()
        client.responses.retrieve = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda **kwargs: {'id': 'resp_pending', 'status': 'completed'}
            )
        )

        # Use path_template="" so the resolved path is exactly tmpdir
        # (mirroring how batch_process + retrieve would share the same resolved path)
        storage_config = StorageConfig(output_dir=tmpdir, path_template="")
        results = await retrieve_background_responses(
            storage_config=storage_config,
            openai_client=client,
            show_progress=False,
        )

        # Only the pending response should have been retrieved
        assert client.responses.retrieve.call_count == 1
        call_args = client.responses.retrieve.call_args[0]
        assert call_args[0] == 'resp_pending'

        assert results is not None
        assert len(results) == 1
        assert results.iloc[0]['response_id'] == 'resp_pending'
        assert results.iloc[0]['status'] == 'completed'


@pytest.mark.asyncio
async def test_auto_discovery_no_pending():
    """When all responses are terminal, auto-discovery returns empty without API calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ParquetLogger(log_dir=tmpdir, buffer_size=10)
        from datetime import timezone

        # Write a completed response
        logger._add_entry({
            'timestamp': datetime(2026, 1, 1, 0, 1, 0, tzinfo=timezone.utc),
            'run_id': 'resp_already_done',
            'parent_run_id': '',
            'custom_id': 'cid-done',
            'event_type': 'background_retrieval_complete',
            'logger_metadata': '{}',
            'payload': json.dumps({
                'event_type': 'background_retrieval_complete',
                'timestamp': '2026-01-01T00:01:00+00:00',
                'execution': {'run_id': 'resp_already_done', 'parent_run_id': '', 'custom_id': 'cid-done', 'tags': [], 'metadata': {}},
                'data': {'response_id': 'resp_already_done', 'status': 'completed', 'poll_attempts': 1},
                'raw': {}
            })
        })
        logger.flush()

        client = AsyncMock()
        client.responses.retrieve = AsyncMock()

        storage_config = StorageConfig(output_dir=tmpdir, path_template="")
        results = await retrieve_background_responses(
            storage_config=storage_config,
            openai_client=client,
            show_progress=False,
        )

        # No API calls should be made
        assert client.responses.retrieve.call_count == 0
        assert results is not None
        assert len(results) == 0


@pytest.mark.asyncio
async def test_auto_discovery_empty_storage():
    """When storage is empty, auto-discovery returns empty without API calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = AsyncMock()
        client.responses.retrieve = AsyncMock()

        storage_config = StorageConfig(output_dir=tmpdir, path_template="")
        results = await retrieve_background_responses(
            storage_config=storage_config,
            openai_client=client,
            show_progress=False,
        )

        assert client.responses.retrieve.call_count == 0
        assert results is not None
        assert len(results) == 0


@pytest.mark.asyncio
async def test_auto_discovery_requires_storage_config():
    """Calling with neither df nor storage_config should raise ValueError."""
    client = AsyncMock()

    with pytest.raises(ValueError, match="storage_config"):
        await retrieve_background_responses(
            openai_client=client,
            show_progress=False,
        )


@pytest.mark.asyncio
async def test_source_local_uses_only_local_storage():
    """source='local' should read only from LocalStorage, not S3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        from langchain_callback_parquet_logger.config import RetrievalConfig, S3Config

        logger = ParquetLogger(log_dir=tmpdir, buffer_size=10)
        from datetime import timezone

        logger._add_entry({
            'timestamp': datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            'run_id': 'resp_local_only',
            'parent_run_id': '',
            'custom_id': 'cid-local',
            'event_type': 'background_retrieval_attempt',
            'logger_metadata': '{}',
            'payload': json.dumps({
                'event_type': 'background_retrieval_attempt',
                'timestamp': '2026-01-01T00:00:00+00:00',
                'execution': {'run_id': 'resp_local_only', 'parent_run_id': '', 'custom_id': 'cid-local', 'tags': [], 'metadata': {}},
                'data': {'response_id': 'resp_local_only', 'attempt_time': '2026-01-01T00:00:00+00:00'},
                'raw': {}
            })
        })
        logger.flush()

        client = AsyncMock()
        client.responses.retrieve = AsyncMock(
            return_value=MagicMock(
                model_dump=lambda **kwargs: {'id': 'resp_local_only', 'status': 'completed'}
            )
        )

        # source="local" (default) — should find the entry written locally
        rc = RetrievalConfig(source="local")
        storage_config = StorageConfig(output_dir=tmpdir, path_template="")
        results = await retrieve_background_responses(
            storage_config=storage_config,
            retrieval_config=rc,
            openai_client=client,
            show_progress=False,
        )

        assert client.responses.retrieve.call_count == 1
        assert results.iloc[0]['response_id'] == 'resp_local_only'
        assert results.iloc[0]['status'] == 'completed'


@pytest.mark.asyncio
async def test_source_s3_requires_s3_config():
    """source='s3' without an s3_config should raise ValueError."""
    from langchain_callback_parquet_logger.config import RetrievalConfig

    client = AsyncMock()
    # StorageConfig with no s3_config
    storage_config = StorageConfig(output_dir="/tmp/test", path_template="")

    with pytest.raises(ValueError, match="s3_config"):
        await retrieve_background_responses(
            storage_config=storage_config,
            retrieval_config=RetrievalConfig(source="s3"),
            openai_client=client,
            show_progress=False,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])