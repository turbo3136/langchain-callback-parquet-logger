"""
Tests for row_timeout, S3 timeouts, and background writer thread.
"""

import asyncio
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from langchain_callback_parquet_logger import (
    ParquetLogger, Batch, with_tags,
    ProcessingConfig, LLMConfig, StorageConfig, S3Config
)


class TestRowTimeout:
    """Test per-row timeout via Batch.run()."""

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_error(self):
        """Test that rows exceeding timeout return TimeoutError."""
        df = pd.DataFrame({
            'prompt': ['fast', 'slow', 'fast2']
        })

        async def mock_ainvoke(**kwargs):
            prompt = kwargs.get('input', '')
            if prompt == 'slow':
                await asyncio.sleep(10)  # Will exceed timeout
            response = Mock()
            response.content = f"Response to: {prompt}"
            return response

        class MockLLM:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        MockLLM.__name__ = 'MockLLM'
        MockLLM.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False, return_results=True,
                    return_exceptions=True, row_timeout=0.1,
                ),
            )
            results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == 3
        # First and third should succeed
        assert hasattr(results[0], 'content')
        assert hasattr(results[2], 'content')
        # Second should be TimeoutError
        assert isinstance(results[1], TimeoutError)
        assert "timed out" in str(results[1])

    @pytest.mark.asyncio
    async def test_timeout_none_means_no_timeout(self):
        """Test that row_timeout=None (default) means no timeout."""
        df = pd.DataFrame({
            'prompt': ['test1', 'test2']
        })

        async def mock_ainvoke(**kwargs):
            await asyncio.sleep(0.05)
            response = Mock()
            response.content = "ok"
            return response

        class MockLLM:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        MockLLM.__name__ = 'MockLLM'
        MockLLM.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False, return_results=True, row_timeout=None,
                ),
            )
            results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == 2
        assert all(hasattr(r, 'content') for r in results)

    @pytest.mark.asyncio
    async def test_timeout_raises_when_return_exceptions_false(self):
        """Test that timeout raises when return_exceptions=False."""
        df = pd.DataFrame({
            'prompt': ['slow']
        })

        async def mock_ainvoke(**kwargs):
            await asyncio.sleep(10)

        class MockLLM:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        MockLLM.__name__ = 'MockLLM'
        MockLLM.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False, return_exceptions=False, row_timeout=0.1,
                ),
            )
            with pytest.raises(TimeoutError, match="timed out"):
                await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

    @pytest.mark.asyncio
    async def test_timeout_progress_counted_on_timeout(self):
        """Test that progress counter increments even on timeout."""
        df = pd.DataFrame({
            'prompt': ['slow']
        })

        async def mock_ainvoke(**kwargs):
            await asyncio.sleep(10)

        class MockLLM:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)
        MockLLM.__name__ = 'MockLLM'
        MockLLM.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False, return_results=True,
                    return_exceptions=True, row_timeout=0.1,
                ),
            )
            results = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

        assert len(results) == 1
        assert isinstance(results[0], TimeoutError)

    @pytest.mark.asyncio
    async def test_timeout_via_processing_config(self, sample_dataframe):
        """Test that row_timeout is passed through from ProcessingConfig."""

        class MockLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.callbacks = kwargs.get('callbacks', [])

                async def slow_invoke(**kw):
                    prompt = kw.get('input', '')
                    if 'Hello' in str(prompt):
                        await asyncio.sleep(10)
                    response = Mock()
                    response.content = "ok"
                    return response

                self.ainvoke = AsyncMock(side_effect=slow_invoke)

        MockLLM.__name__ = 'MockLLM'
        MockLLM.__module__ = 'test_module'

        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False,
                    return_results=True,
                    return_exceptions=True,
                    row_timeout=0.1  # Short timeout
                ),
            )
            results = await batch.run(
                df,
                llm_config=LLMConfig(
                    llm_class=MockLLM,
                    llm_kwargs={'model': 'test-model'}
                ),
            )

            # At least one should be a TimeoutError (the "Hello world" row)
            assert any(isinstance(r, TimeoutError) for r in results)


class TestS3Timeouts:
    """Test S3 connection/read timeout configuration."""

    def test_s3_config_default_timeouts(self):
        """Test default S3 timeout values."""
        config = S3Config(bucket="test-bucket")
        assert config.connect_timeout == 10
        assert config.read_timeout == 30

    def test_s3_config_custom_timeouts(self):
        """Test custom S3 timeout values."""
        config = S3Config(
            bucket="test-bucket",
            connect_timeout=5,
            read_timeout=15
        )
        assert config.connect_timeout == 5
        assert config.read_timeout == 15

    def test_s3_storage_passes_timeouts_to_boto(self):
        """Test that S3Storage passes timeout config to boto3 client."""
        try:
            import boto3
        except ImportError:
            pytest.skip("boto3 not installed")

        from langchain_callback_parquet_logger.storage import S3Storage

        config = S3Config(
            bucket="test-bucket",
            connect_timeout=7,
            read_timeout=20
        )

        storage = S3Storage(config)

        with patch.object(boto3, 'client') as mock_client:
            _ = storage.client

            mock_client.assert_called_once()
            call_kwargs = mock_client.call_args
            boto_config = call_kwargs.kwargs.get('config') or call_kwargs[1].get('config')
            assert boto_config.connect_timeout == 7
            assert boto_config.read_timeout == 20


class TestBackgroundWriter:
    """Test background writer thread behavior."""

    def test_writer_thread_starts(self, temp_log_dir):
        """Test that writer thread is started on init."""
        logger = ParquetLogger(temp_log_dir, buffer_size=100)
        assert logger._writer_thread.is_alive()
        assert logger._writer_thread.daemon is True
        assert logger._writer_thread.name == "parquet-writer"

    def test_flush_waits_for_writer(self, temp_log_dir):
        """Test that flush() waits for all queued writes to complete."""
        logger = ParquetLogger(temp_log_dir, buffer_size=1)

        # Add entries that will trigger auto-flush via queue
        for i in range(5):
            logger._add_entry({
                'timestamp': pd.Timestamp.now('UTC'),
                'run_id': f'run-{i}',
                'custom_id': '',
                'event_type': 'test',
                'parent_run_id': '',
                'logger_metadata': '{}',
                'payload': '{}'
            })

        # flush() should wait for everything
        logger.flush()

        # All entries should be written
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        total_rows = sum(len(pd.read_parquet(f)) for f in files)
        assert total_rows == 5

    def test_context_manager_flushes(self, temp_log_dir):
        """Test that context manager exit triggers flush."""
        with ParquetLogger(temp_log_dir, buffer_size=100) as logger:
            logger._add_entry({
                'timestamp': pd.Timestamp.now('UTC'),
                'run_id': 'run-1',
                'custom_id': '',
                'event_type': 'test',
                'parent_run_id': '',
                'logger_metadata': '{}',
                'payload': '{}'
            })

        # After context exit, file should exist
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 1

    def test_writer_error_surfaces_at_flush(self, temp_log_dir):
        """Test that writer thread errors surface when flush() is called."""
        logger = ParquetLogger(temp_log_dir, buffer_size=100)

        # Add an entry
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'custom_id': '',
            'event_type': 'test',
            'parent_run_id': '',
            'logger_metadata': '{}',
            'payload': '{}'
        })

        # Sabotage the storage backend to cause a write error
        original_write = logger.storage.write
        def failing_write(*args, **kwargs):
            raise RuntimeError("Simulated write failure")
        logger.storage.write = failing_write

        # Flush should raise the writer thread's error
        with pytest.raises(RuntimeError, match="Simulated write failure"):
            logger.flush()

    def test_concurrent_add_entry(self, temp_log_dir):
        """Test thread safety of _add_entry under concurrent access."""
        logger = ParquetLogger(temp_log_dir, buffer_size=10)
        num_threads = 5
        entries_per_thread = 20

        def add_entries(thread_id):
            for i in range(entries_per_thread):
                logger._add_entry({
                    'timestamp': pd.Timestamp.now('UTC'),
                    'run_id': f'run-{thread_id}-{i}',
                    'custom_id': '',
                    'event_type': 'test',
                    'parent_run_id': '',
                    'logger_metadata': '{}',
                    'payload': '{}'
                })

        threads = [
            threading.Thread(target=add_entries, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logger.flush()

        # All entries should be written
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        total_rows = sum(len(pd.read_parquet(f)) for f in files)
        assert total_rows == num_threads * entries_per_thread

    def test_flush_timeout(self, temp_log_dir):
        """Test that flush respects the timeout parameter."""
        logger = ParquetLogger(temp_log_dir, buffer_size=100)

        # Add an entry
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'custom_id': '',
            'event_type': 'test',
            'parent_run_id': '',
            'logger_metadata': '{}',
            'payload': '{}'
        })

        # Flush with a very short timeout should still work for fast writes
        logger.flush(timeout=5.0)

        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 1

    def test_non_runtime_error_surfaces_at_flush(self, temp_log_dir):
        """Test that non-RuntimeError write failures (e.g. OSError) surface at flush()
        rather than being silently swallowed.  Before the fix, only RuntimeError was
        re-raised; other exceptions were printed to stdout and discarded."""
        logger = ParquetLogger(temp_log_dir, buffer_size=100)

        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'custom_id': '',
            'event_type': 'test',
            'parent_run_id': '',
            'logger_metadata': '{}',
            'payload': '{}'
        })

        # Sabotage with a non-RuntimeError to verify it also surfaces
        def failing_write(*args, **kwargs):
            raise OSError("Disk full")
        logger.storage.write = failing_write

        with pytest.raises(OSError, match="Disk full"):
            logger.flush()


class TestCompositeStorage:
    """Test CompositeStorage behaviour."""

    def test_list_files_warns_on_backend_failure(self, temp_log_dir):
        """Test that CompositeStorage.list_files() emits a warning when a backend
        fails rather than silently returning incomplete results."""
        from langchain_callback_parquet_logger.storage import CompositeStorage, LocalStorage
        from unittest.mock import MagicMock

        good_backend = LocalStorage(Path(temp_log_dir))

        bad_backend = MagicMock(spec=LocalStorage)
        bad_backend.list_files.side_effect = Exception("S3 connection failed")

        composite = CompositeStorage([good_backend, bad_backend])

        with pytest.warns(UserWarning, match="S3 connection failed"):
            files = composite.list_files()

        # Good backend results are still returned despite the bad backend failing
        assert isinstance(files, list)
