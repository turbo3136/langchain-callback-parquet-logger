"""
Tests for Batch class functionality.
"""

import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock, MagicMock, PropertyMock
from langchain_callback_parquet_logger import (
    Batch, BatchOutcome, with_tags,
    LLMConfig, JobConfig, StorageConfig, ProcessingConfig,
    ColumnConfig, S3Config
)


def create_mock_llm_class():
    """Create a mock LLM class for testing."""
    class MockLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.callbacks = kwargs.get('callbacks', [])
            self.ainvoke = AsyncMock(return_value=Mock(content="response"))

        def with_structured_output(self, schema):
            """Simulate RunnableSequence that doesn't have callbacks attribute."""
            class MockRunnableSequence:
                def __init__(self, base_llm, schema):
                    self.base_llm = base_llm
                    self.schema = schema
                    # Importantly, no callbacks attribute
                    self.ainvoke = AsyncMock(return_value=Mock(content="structured_response"))

            return MockRunnableSequence(self, schema)

    MockLLM.__name__ = 'MockLLM'
    MockLLM.__module__ = 'test_module'
    return MockLLM


class TestBatch:
    """Test Batch class functionality."""

    @pytest.mark.asyncio
    async def test_batch_local_only(self, sample_dataframe):
        """Test Batch.run() with local storage only."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                job_config=JobConfig(category="test_job", subcategory="test_sub"),
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            results = await batch.run(
                df,
                llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
            )

            # Check that results were returned
            assert results is not None
            assert len(results) == len(df)

            # Check that directory structure was created
            expected_path = Path(tmpdir) / "test_job" / "test_sub"
            assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_batch_with_s3(self, sample_dataframe):
        """Test Batch.run() with S3 configuration."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock S3Storage.client property to avoid boto3 import
            with patch('langchain_callback_parquet_logger.storage.S3Storage.client', new_callable=PropertyMock) as mock_client:
                # Mock S3 client
                mock_s3_client = MagicMock()
                mock_client.return_value = mock_s3_client

                batch = Batch(
                    job_config=JobConfig(category="test_job"),
                    storage_config=StorageConfig(
                        output_dir=tmpdir,
                        path_template="{job_category}/{job_subcategory}",
                        s3_config=S3Config(
                            bucket="test-bucket",
                            prefix="jobs/{job_category}/{date}/"
                        )
                    ),
                    processing_config=ProcessingConfig(
                        show_progress=False,
                        buffer_size=1  # Force immediate flush to test S3 upload
                    ),
                )
                results = await batch.run(
                    df,
                    llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
                )

                # Check local directory was created
                expected_path = Path(tmpdir) / "test_job" / "default"
                assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_batch_structured_output(self, sample_dataframe):
        """Test Batch.run() with structured output."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            value: str

        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        # Create mock LLM class that returns structured output
        class MockStructuredLLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.callbacks = []
                self.structured_output = None
                self.ainvoke = AsyncMock(return_value=TestModel(value="test"))

            def with_structured_output(self, schema):
                self.structured_output = schema
                return self

        MockStructuredLLM.__name__ = 'MockStructuredLLM'
        MockStructuredLLM.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            results = await batch.run(
                df,
                llm_config=LLMConfig(
                    llm_class=MockStructuredLLM,
                    llm_kwargs={'model': 'test-model'},
                    structured_output=TestModel
                ),
            )

            # Check results
            assert len(results) == len(df)

    @pytest.mark.asyncio
    async def test_batch_llm_types(self, sample_dataframe):
        """Test that different LLM classes work with batch processing."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        # Test different LLM class names
        for llm_class_name in ['ChatOpenAI', 'ChatAnthropic', 'ChatCohere', 'UnknownLLM']:
            # Create mock LLM class with specific name
            class MockTypedLLM:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs
                    self.callbacks = []
                    self.ainvoke = AsyncMock(return_value=Mock(content="response"))

                def with_structured_output(self, schema):
                    return self

            MockTypedLLM.__name__ = llm_class_name
            MockTypedLLM.__module__ = 'test_module'

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch('langchain_callback_parquet_logger.batch.ParquetLogger') as MockLogger:
                    mock_logger_instance = MagicMock()
                    MockLogger.return_value = mock_logger_instance
                    mock_logger_instance.__enter__ = Mock(return_value=mock_logger_instance)
                    mock_logger_instance.__exit__ = Mock(return_value=None)

                    batch = Batch(
                        storage_config=StorageConfig(output_dir=tmpdir),
                        processing_config=ProcessingConfig(show_progress=False, buffer_size=1000),
                    )
                    await batch.run(
                        df,
                        llm_config=LLMConfig(llm_class=MockTypedLLM, llm_kwargs={'model': 'test-model'}),
                    )

                    # Check that ParquetLogger was called
                    assert MockLogger.called

    @pytest.mark.asyncio
    async def test_batch_path_templates(self, sample_dataframe):
        """Test path template formatting."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                job_config=JobConfig(
                    category="emails",
                    subcategory="validation",
                    version="2.0.1",
                    environment="staging"
                ),
                storage_config=StorageConfig(
                    output_dir=tmpdir,
                    path_template="{environment}/{job_category}/v{job_version_safe}/{job_subcategory}"
                ),
                processing_config=ProcessingConfig(show_progress=False),
            )
            await batch.run(
                df,
                llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
            )

            # Check that path was formatted correctly (version dots replaced with underscores)
            expected_path = Path(tmpdir) / "staging" / "emails" / "v2_0_1" / "validation"
            assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_batch_override_params(self, sample_dataframe):
        """Test logger and batch override parameters."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('langchain_callback_parquet_logger.batch.ParquetLogger') as MockLogger:
                mock_logger_instance = MagicMock()
                MockLogger.return_value = mock_logger_instance
                mock_logger_instance.__enter__ = Mock(return_value=mock_logger_instance)
                mock_logger_instance.__exit__ = Mock(return_value=None)

                batch = Batch(
                    storage_config=StorageConfig(
                        output_dir=tmpdir,
                        s3_config=S3Config(bucket="test-bucket", retry_attempts=5)
                    ),
                    processing_config=ProcessingConfig(
                        buffer_size=500,
                        event_types=['llm_start', 'llm_end', 'chain_start'],
                        show_progress=False
                    ),
                )
                await batch.run(
                    df,
                    llm_config=LLMConfig(
                        llm_class=MockLLM,
                        llm_kwargs={'model': 'test-model'},
                        model_kwargs={'temperature': 0.5}
                    ),
                )

                # Check that logger overrides were applied
                assert MockLogger.called
                if MockLogger.call_args:
                    call_kwargs = MockLogger.call_args.kwargs
                    assert call_kwargs['buffer_size'] == 500
                    assert call_kwargs['event_types'] == ['llm_start', 'llm_end', 'chain_start']

    @pytest.mark.asyncio
    async def test_batch_missing_columns(self, sample_dataframe):
        """Test error when required columns are missing."""
        df = sample_dataframe.copy()
        # Don't add the required 'prompt' column

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(storage_config=StorageConfig(output_dir=tmpdir))
            with pytest.raises(ValueError, match="DataFrame missing required column"):
                await batch.run(
                    df,
                    llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
                )

    @pytest.mark.asyncio
    async def test_batch_environment_s3_bucket(self, sample_dataframe):
        """Test that S3 bucket is read from environment variable."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'LANGCHAIN_S3_BUCKET': 'env-bucket'}):
                with patch('langchain_callback_parquet_logger.batch.ParquetLogger') as MockLogger:
                    mock_logger_instance = MagicMock()
                    MockLogger.return_value = mock_logger_instance
                    mock_logger_instance.__enter__ = Mock(return_value=mock_logger_instance)
                    mock_logger_instance.__exit__ = Mock(return_value=None)

                    batch = Batch(
                        storage_config=StorageConfig(
                            output_dir=tmpdir,
                            s3_config=S3Config(bucket='env-bucket')
                        ),
                        processing_config=ProcessingConfig(show_progress=False),
                    )
                    await batch.run(
                        df,
                        llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
                    )

                    # Check that S3 bucket from env was used
                    assert MockLogger.called
                    if MockLogger.call_args:
                        call_kwargs = MockLogger.call_args.kwargs
                        # S3 config should be passed as s3_config object
                        assert call_kwargs.get('s3_config') is not None
                        if call_kwargs.get('s3_config'):
                            assert call_kwargs['s3_config'].bucket == 'env-bucket'

    @pytest.mark.asyncio
    async def test_batch_version_paths(self, sample_dataframe):
        """Test that versions are correctly sanitized in both local and S3 paths."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        # Test 1: With version specified (dots should become underscores)
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('langchain_callback_parquet_logger.storage.S3Storage.client', new_callable=PropertyMock) as mock_client:
                mock_s3_client = MagicMock()
                mock_client.return_value = mock_s3_client

                batch = Batch(
                    job_config=JobConfig(
                        category="ml_models",
                        subcategory="classification",
                        version="3.2.1"  # Version with dots
                    ),
                    storage_config=StorageConfig(
                        output_dir=tmpdir,
                        # Using default template which includes version
                        s3_config=S3Config(bucket="test-bucket", prefix="models/")
                    ),
                    processing_config=ProcessingConfig(
                        show_progress=False,
                        buffer_size=1  # Force immediate flush
                    ),
                )
                await batch.run(
                    df,
                    llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
                )

                # Check local path has sanitized version
                expected_local = Path(tmpdir) / "ml_models" / "classification" / "v3_2_1"
                assert expected_local.exists(), f"Expected path {expected_local} does not exist"

        # Test 2: Without version specified (should use 'unversioned')
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                job_config=JobConfig(category="experiments", subcategory="baseline"),
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False),
            )
            await batch.run(
                df,
                llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
            )

            # Check local path has 'unversioned' as default
            expected_local = Path(tmpdir) / "experiments" / "baseline" / "vunversioned"
            assert expected_local.exists(), f"Expected path {expected_local} does not exist"

    @pytest.mark.asyncio
    async def test_structured_output_with_callbacks(self, sample_dataframe):
        """Test that structured output works correctly with callbacks.

        This test ensures that when using structured output (which wraps the LLM in a
        RunnableSequence), callbacks are still properly attached and functional.
        This addresses the bug where RunnableSequence doesn't have a callbacks attribute.
        """
        from pydantic import BaseModel

        class TestSchema(BaseModel):
            answer: str
            confidence: float

        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            # This should not raise "RunnableSequence has no field callbacks"
            batch = Batch(
                job_config=JobConfig(
                    category="structured_test",
                    description="Testing structured output with callbacks"
                ),
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            results = await batch.run(
                df,
                llm_config=LLMConfig(
                    llm_class=MockLLM,
                    llm_kwargs={'model': 'test-model'},
                    structured_output=TestSchema  # This causes LLM to be wrapped in RunnableSequence
                ),
            )

            # Verify the process completed without errors
            expected_path = Path(tmpdir) / "structured_test" / "default" / "vunversioned"
            assert expected_path.exists(), f"Expected path {expected_path} does not exist"

            # Check that callbacks were properly passed through LLM constructor
            # The MockLLM should have received callbacks in its kwargs
            # This is verified implicitly by the test not raising an error

    @pytest.mark.asyncio
    async def test_batch_pending_response_ids(self, sample_dataframe):
        """Test that response_id_extractor captures background response IDs."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        # Mock LLM that returns objects with a response_id attribute
        class MockBgLLM:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self._call_count = 0
                self.ainvoke = AsyncMock(side_effect=self._invoke)

            async def _invoke(self, **kwargs):
                self._call_count += 1
                result = Mock()
                result.response_id = f'resp_{self._call_count:03d}'
                result.content = 'queued response'
                return result

        MockBgLLM.__name__ = 'MockBgLLM'
        MockBgLLM.__module__ = 'test_module'

        def extract_id(result, row):
            return getattr(result, 'response_id', None)

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False,
                    return_results=True,
                    return_exceptions=True,
                ),
            )
            results = await batch.run(
                df,
                llm_config=LLMConfig(llm_class=MockBgLLM),
                response_id_extractor=extract_id,
            )

            # Should have captured response IDs
            assert len(batch.pending_response_ids) == len(df)
            for rid in batch.pending_response_ids:
                assert rid.startswith('resp_')

    @pytest.mark.asyncio
    async def test_batch_run_and_retrieve_structure(self, sample_dataframe):
        """Test run_and_retrieve() with a synchronous LLM: returns BatchOutcome with
        batch_results populated and retrieval_results=None (no background IDs captured,
        so retrieve() is skipped)."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            batch.retrieve = AsyncMock(return_value=pd.DataFrame())

            outcome = await batch.run_and_retrieve(
                df,
                llm_config=LLMConfig(llm_class=MockLLM, llm_kwargs={'model': 'test-model'}),
            )

            assert isinstance(outcome, BatchOutcome)
            assert outcome.batch_results is not None
            # Synchronous LLM — no background IDs captured, retrieve() must be skipped
            assert outcome.retrieval_results is None
            batch.retrieve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retrieval_config_in_constructor(self, sample_dataframe):
        """Test that RetrievalConfig stored in constructor is accessible and used as default."""
        from langchain_callback_parquet_logger import RetrievalConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            rc = RetrievalConfig(poll_interval=99.0, max_poll_attempts=7, show_progress=False)
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
                retrieval_config=rc,
            )

            # Verify it's stored
            assert batch.retrieval_config is rc
            assert batch.retrieval_config.poll_interval == 99.0
            assert batch.retrieval_config.max_poll_attempts == 7

            # Calling retrieve() without args uses self.retrieval_config (source='memory' default)
            # — returns empty DataFrame immediately without touching storage or needing a client
            result = await batch.retrieve()
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

    @pytest.mark.asyncio
    async def test_openai_extractor_auto_applied(self, sample_dataframe):
        """Test that response_id_extractor is auto-detected for ChatOpenAI-named classes
        when response_metadata contains a pending status."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        call_count = 0

        async def mock_ainvoke(**kwargs):
            nonlocal call_count
            call_count += 1
            return Mock(
                response_metadata={'id': f'resp_{call_count:03d}', 'status': 'in_progress'},
                content='',
            )

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            # No response_id_extractor passed — should be auto-detected
            await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))

            assert len(batch.pending_response_ids) == len(df)
            for rid in batch.pending_response_ids:
                assert rid.startswith('resp_')

    @pytest.mark.asyncio
    async def test_openai_extractor_skips_synchronous(self, sample_dataframe):
        """Test that auto-detected extractor returns None for synchronous completions
        (status 'completed'), so they are not queued for background retrieval."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(return_value=Mock(
                    response_metadata={'id': 'chatcmpl_sync456', 'status': 'completed'},
                    content='hello',
                ))

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))

            # Synchronous results should NOT be captured for retrieval
            assert len(batch.pending_response_ids) == 0

    @pytest.mark.asyncio
    async def test_run_and_retrieve_calls_retrieve_when_ids_captured(self, sample_dataframe):
        """Test that run_and_retrieve() calls retrieve() when background IDs are captured."""
        from langchain_callback_parquet_logger import RetrievalConfig

        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        call_count = 0

        async def mock_ainvoke(**kwargs):
            nonlocal call_count
            call_count += 1
            return Mock(
                response_metadata={'id': f'resp_{call_count:03d}', 'status': 'in_progress'},
                content='',
            )

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            # Patch retrieve() so we don't need a real OpenAI client
            batch.retrieve = AsyncMock(return_value=pd.DataFrame({'response_id': ['resp_001']}))

            outcome = await batch.run_and_retrieve(
                df,
                llm_config=LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}),
            )

            # Background IDs were captured — retrieve() should have been called
            assert isinstance(outcome, BatchOutcome)
            assert outcome.batch_results is not None
            assert outcome.retrieval_results is not None
            batch.retrieve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_source_memory_returns_empty(self):
        """Test that retrieve() with source='memory' returns empty DataFrame
        immediately without touching storage when no in-memory IDs are available."""
        from langchain_callback_parquet_logger import RetrievalConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False),
            )
            # No run() called — _background_response_ids is empty
            # source='memory' (default) should return empty without reading any files
            result = await batch.retrieve(
                retrieval_config=RetrievalConfig(source='memory', return_results=True)
            )
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0

            # Verify no Parquet files were created (no storage access)
            parquet_files = list(Path(tmpdir).glob('**/*.parquet'))
            assert len(parquet_files) == 0

    @pytest.mark.asyncio
    async def test_run_captures_ids_when_return_results_false(self, sample_dataframe):
        """Test that background response IDs are captured even when return_results=False.
        run() should hold results in memory internally for ID extraction, then discard
        them before returning to the caller."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        call_count = 0

        async def mock_ainvoke(**kwargs):
            nonlocal call_count
            call_count += 1
            return Mock(
                response_metadata={'id': f'resp_{call_count:03d}', 'status': 'in_progress'},
                content='',
            )

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                # return_results=False (the default) — would silently break ID capture before fix
                processing_config=ProcessingConfig(show_progress=False, return_results=False),
            )
            result = await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))

            # Caller gets None (respects return_results=False)
            assert result is None
            # But IDs were captured internally
            assert len(batch.pending_response_ids) == len(df)
            for rid in batch.pending_response_ids:
                assert rid.startswith('resp_')

    @pytest.mark.asyncio
    async def test_retrieve_show_progress_false_suppresses_no_responses_print(self, capsys):
        """Test that show_progress=False suppresses the 'No pending responses' message
        when storage discovery finds nothing."""
        from langchain_callback_parquet_logger import RetrievalConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False),
            )
            # source='local' triggers storage discovery (which finds nothing)
            await batch.retrieve(
                retrieval_config=RetrievalConfig(source='local', return_results=True, show_progress=False)
            )

            captured = capsys.readouterr()
            assert "No pending responses" not in captured.out

    @pytest.mark.asyncio
    async def test_extractor_exception_emits_warning(self, sample_dataframe):
        """Test that a crashing response_id_extractor emits a warning rather than
        silently swallowing the exception."""
        import warnings as warnings_module

        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        MockLLM = create_mock_llm_class()

        def crashing_extractor(result, row):
            raise ValueError("Extractor exploded!")

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            with pytest.warns(UserWarning, match="Extractor exploded"):
                await batch.run(
                    df,
                    LLMConfig(llm_class=MockLLM, llm_kwargs={}),
                    response_id_extractor=crashing_extractor,
                )

    @pytest.mark.asyncio
    async def test_run_returns_results_by_default(self, sample_dataframe):
        """Test that run() returns a list by default (return_results now defaults to True).
        Before this fix the default was False, causing silent None returns."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                # No return_results set — should default to True
                processing_config=ProcessingConfig(show_progress=False),
            )
            result = await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))

            assert result is not None
            assert isinstance(result, list)
            assert len(result) == len(df)

    @pytest.mark.asyncio
    async def test_run_resets_ids_between_calls(self, sample_dataframe):
        """Test that each run() call resets _background_response_ids so the second
        call's IDs don't accumulate on top of the first call's IDs."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        call_count = 0

        async def mock_ainvoke(**kwargs):
            nonlocal call_count
            call_count += 1
            return Mock(
                response_metadata={'id': f'resp_{call_count:03d}', 'status': 'in_progress'},
                content='',
            )

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke)

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            # First run
            await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))
            first_ids = set(batch.pending_response_ids)

            # Second run — IDs should be replaced entirely, not accumulated
            await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))
            second_ids = set(batch.pending_response_ids)

            assert len(second_ids) == len(df)
            assert first_ids.isdisjoint(second_ids), (
                "Second run should have replaced first run's IDs, not accumulated them"
            )

    @pytest.mark.asyncio
    async def test_run_clears_ids_when_no_background_llm(self, sample_dataframe):
        """Test that run() with a non-background LLM clears IDs from a prior run(),
        preventing stale IDs from leaking into retrieve()."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']

        call_count = 0

        async def mock_ainvoke_bg(**kwargs):
            nonlocal call_count
            call_count += 1
            return Mock(
                response_metadata={'id': f'resp_{call_count:03d}', 'status': 'in_progress'},
                content='',
            )

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])
                self.ainvoke = AsyncMock(side_effect=mock_ainvoke_bg)

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False, return_results=True),
            )
            # First run with background LLM — captures IDs
            await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))
            assert len(batch.pending_response_ids) == len(df)

            # Second run with a plain LLM — IDs should be cleared, not preserved
            await batch.run(df, LLMConfig(llm_class=MockLLM, llm_kwargs={}))
            assert len(batch.pending_response_ids) == 0, (
                "run() with a non-background LLM should clear IDs from the previous run"
            )

    @pytest.mark.asyncio
    async def test_retrieve_logger_has_job_metadata(self):
        """Test that retrieve() creates its ParquetLogger with job metadata so that
        retrieval events have the same batch context as run() events."""
        from langchain_callback_parquet_logger import JobConfig, RetrievalConfig, ParquetLogger
        from unittest.mock import patch as _patch, AsyncMock as _AsyncMock

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                job_config=JobConfig(category='mycat', version='2.0'),
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False),
            )
            # Inject a fake response ID to trigger the retrieval path
            batch._background_response_ids = {'resp_001': {'custom_id': 'cid_001', 'run_id': '', 'parent_run_id': '', 'tags': []}}

            captured_metadata = []

            # Subclass to capture what logger_metadata is passed
            original_init = ParquetLogger.__init__

            def capturing_init(self, *args, **kwargs):
                captured_metadata.append(kwargs.get('logger_metadata'))
                original_init(self, *args, **kwargs)

            with _patch.object(ParquetLogger, '__init__', capturing_init):
                with _patch(
                    'langchain_callback_parquet_logger.background_retrieval.retrieve_background_responses',
                    new=_AsyncMock(return_value=pd.DataFrame()),
                ):
                    await batch.retrieve(
                        retrieval_config=RetrievalConfig(show_progress=False, return_results=True)
                    )

            assert len(captured_metadata) == 1
            meta = captured_metadata[0]
            assert meta is not None, "retrieve() must pass logger_metadata to ParquetLogger"
            assert 'batch_config' in meta
            assert meta['batch_config']['job']['category'] == 'mycat'
            assert meta['batch_config']['job']['version'] == '2.0'
            assert 'retrieval_started_at' in meta

    @pytest.mark.asyncio
    async def test_run_captures_ids_from_structured_output_valueerror(self):
        """Test that response IDs are captured even when with_structured_output() raises
        ValueError for queued background responses (the ID is in the exception message)."""
        from langchain_callback_parquet_logger.batch import _try_extract_id_from_exception

        # Verify the helper itself works on the expected ValueError format
        fake_exc = ValueError(
            "Structured Output response does not have a 'parsed' field nor a 'refusal' field. "
            "Received message:\n\n"
            "content=[] additional_kwargs={} response_metadata={'id': 'resp_abc123', "
            "'status': 'queued', 'model': 'gpt-5'} id='resp_abc123'"
        )
        assert _try_extract_id_from_exception(fake_exc) == 'resp_abc123'

        # Returns None for completed/failed statuses
        completed_exc = ValueError(
            "content=[] response_metadata={'id': 'resp_xyz', 'status': 'completed'} id='resp_xyz'"
        )
        assert _try_extract_id_from_exception(completed_exc) is None

        # Returns None for unrelated exceptions
        assert _try_extract_id_from_exception(RuntimeError("boom")) is None

        # Now verify that run() captures IDs when all results are ValueErrors
        df = pd.DataFrame({'prompt': ['a', 'b']})

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])

            def with_structured_output(self, schema):
                call_count = 0

                class StructuredWrapper:
                    async def ainvoke(self_, input, config=None, **kwargs):
                        nonlocal call_count
                        call_count += 1
                        rid = f'resp_bg{call_count:03d}'
                        raise ValueError(
                            f"Structured Output response does not have a 'parsed' field. "
                            f"Received message:\n\ncontent=[] response_metadata={{'id': '{rid}', "
                            f"'status': 'queued'}} id='{rid}'"
                        )

                return StructuredWrapper()

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        from pydantic import BaseModel

        class MySchema(BaseModel):
            value: str

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False, return_results=True, return_exceptions=True
                ),
            )
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore", UserWarning)
                await batch.run(
                    df,
                    LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}, structured_output=MySchema),
                )

        assert len(batch.pending_response_ids) == 2, (
            "IDs should be extracted from ValueError exceptions thrown by with_structured_output()"
        )
        assert all(rid.startswith('resp_bg') for rid in batch.pending_response_ids)

    @pytest.mark.asyncio
    async def test_run_warns_when_structured_output_with_background_llm(self):
        """Test that combining structured_output with a background-capable LLM emits a warning."""
        import warnings as _w
        from pydantic import BaseModel

        class MySchema(BaseModel):
            value: str

        df = pd.DataFrame({'prompt': ['hello']})

        class MockChatOpenAI:
            def __init__(self, **kwargs):
                self.callbacks = kwargs.get('callbacks', [])

            def with_structured_output(self, schema):
                class W:
                    async def ainvoke(self_, input, config=None, **kwargs):
                        raise ValueError(
                            "Structured Output response does not have a 'parsed' field. "
                            "Received message:\n\ncontent=[] response_metadata={'id': 'resp_warn1', "
                            "'status': 'queued'} id='resp_warn1'"
                        )
                return W()

        MockChatOpenAI.__name__ = 'ChatOpenAI'
        MockChatOpenAI.__module__ = 'test_module'

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(
                    show_progress=False, return_results=True, return_exceptions=True
                ),
            )
            with pytest.warns(UserWarning, match="structured_output"):
                await batch.run(
                    df,
                    LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}, structured_output=MySchema),
                )

    @pytest.mark.asyncio
    async def test_run_and_retrieve_prints_when_no_ids_captured(self, capsys):
        """Test that run_and_retrieve() prints an explanatory message when no background
        IDs are captured (e.g. synchronous LLM), instead of silently returning None."""
        df = pd.DataFrame({'prompt': ['hello']})
        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=True, return_results=True),
            )
            outcome = await batch.run_and_retrieve(
                df, LLMConfig(llm_class=MockLLM, llm_kwargs={})
            )

        assert outcome.retrieval_results is None
        captured = capsys.readouterr()
        assert "No background response IDs captured" in captured.out
        assert "retrieval skipped" in captured.out

    @pytest.mark.asyncio
    async def test_retrieve_adds_parsed_output_column(self):
        """Test that retrieve() adds a parsed_output column when _last_structured_output
        is set, using _parse_from_openai_response() to validate the JSON text."""
        from pydantic import BaseModel
        from langchain_callback_parquet_logger.batch import _parse_from_openai_response
        from langchain_callback_parquet_logger import RetrievalConfig

        class Location(BaseModel):
            city: str
            country: str

        # Verify the parser helper directly
        response_dict = {
            'output': [
                {
                    'content': [
                        {'type': 'output_text', 'text': '{"city": "Paris", "country": "France"}'}
                    ]
                }
            ]
        }
        result = _parse_from_openai_response(response_dict, Location)
        assert isinstance(result, Location)
        assert result.city == 'Paris'
        assert result.country == 'France'

        # Also verify via output_text convenience key
        response_dict2 = {'output_text': '{"city": "Berlin", "country": "Germany"}'}
        result2 = _parse_from_openai_response(response_dict2, Location)
        assert isinstance(result2, Location)
        assert result2.city == 'Berlin'

        # Returns None for bad input
        assert _parse_from_openai_response({}, Location) is None
        assert _parse_from_openai_response({'output': []}, Location) is None

        # End-to-end: retrieve() applies the schema when _last_structured_output is set
        fake_retrieval_df = pd.DataFrame([
            {
                'response_id': 'resp_001',
                'status': 'completed',
                'openai_response': {
                    'output': [
                        {'content': [{'type': 'output_text', 'text': '{"city": "Tokyo", "country": "Japan"}'}]}
                    ]
                },
                # retrieve_background_responses() now returns parsed_output as a dict;
                # Batch.retrieve() converts it to a Pydantic instance.
                'parsed_output': {'city': 'Tokyo', 'country': 'Japan'},
                'error': None,
            }
        ])

        from unittest.mock import patch as _patch, AsyncMock as _AsyncMock

        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Batch(
                storage_config=StorageConfig(output_dir=tmpdir),
                processing_config=ProcessingConfig(show_progress=False),
            )
            batch._background_response_ids = {'resp_001': {'custom_id': 'cid_001', 'run_id': '', 'parent_run_id': '', 'tags': []}}
            batch._last_structured_output = Location

            with _patch(
                'langchain_callback_parquet_logger.background_retrieval.retrieve_background_responses',
                new=_AsyncMock(return_value=fake_retrieval_df),
            ):
                results = await batch.retrieve(
                    retrieval_config=RetrievalConfig(show_progress=False, return_results=True)
                )

        assert results is not None
        assert 'parsed_output' in results.columns
        parsed = results['parsed_output'].iloc[0]
        assert isinstance(parsed, Location)
        assert parsed.city == 'Tokyo'
        assert parsed.country == 'Japan'


def test_parse_from_openai_response_handles_null_content():
    """Test that _parse_from_openai_response does not raise TypeError when reasoning
    output items have content=None (explicitly null in the JSON, not a missing key)."""
    from pydantic import BaseModel
    from langchain_callback_parquet_logger.batch import _parse_from_openai_response

    class City(BaseModel):
        name: str

    # Reasoning item has content=None; message item has the actual output_text.
    # item.get('content', []) returns None when the key exists with null value —
    # (item.get('content') or []) returns [] instead, avoiding the TypeError.
    response_dict = {
        'output': [
            {'type': 'reasoning', 'content': None},
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': '{"name": "Rome"}'}],
            },
        ]
    }
    result = _parse_from_openai_response(response_dict, City)
    assert isinstance(result, City)
    assert result.name == 'Rome'

    # A response with only a reasoning item (no output_text) should return None gracefully.
    response_only_reasoning = {'output': [{'type': 'reasoning', 'content': None}]}
    assert _parse_from_openai_response(response_only_reasoning, City) is None


def test_extract_custom_id_from_row_uses_config_tags():
    """Test that extract_custom_id correctly extracts the ID from the tags list
    produced by with_tags(custom_id='x') — the same format that Batch.run() now uses."""
    from langchain_callback_parquet_logger.tagging import extract_custom_id
    from langchain_callback_parquet_logger import with_tags

    config = with_tags(custom_id='abc')
    tags = config.get('tags', [])
    assert extract_custom_id(tags) == 'abc'

    # Multiple tags — extract_custom_id should pick the right one
    config2 = with_tags(custom_id='xyz')
    tags2 = config2.get('tags', []) + ['some_other:tag']
    assert extract_custom_id(tags2) == 'xyz'

    # No custom_id tag → empty string
    assert extract_custom_id(['other:tag', 'unrelated']) == ''
    assert extract_custom_id([]) == ''


@pytest.mark.asyncio
async def test_run_captures_custom_id_from_config_tags():
    """Test that Batch.run() populates custom_id from the config column tags when
    there is no top-level custom_id column — mirrors what logger.py does for llm_end."""
    call_count = 0

    async def mock_ainvoke(**kwargs):
        nonlocal call_count
        call_count += 1
        return Mock(
            response_metadata={'id': f'resp_cid{call_count:03d}', 'status': 'in_progress'},
            content='',
        )

    class MockChatOpenAI:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get('callbacks', [])
            self.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    MockChatOpenAI.__name__ = 'ChatOpenAI'
    MockChatOpenAI.__module__ = 'test_module'

    df = pd.DataFrame({
        'prompt': ['hello', 'world'],
        'config': [
            with_tags(custom_id='user-001'),
            with_tags(custom_id='user-002'),
        ],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        batch = Batch(
            storage_config=StorageConfig(output_dir=tmpdir),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
        )
        await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))

    # Both IDs should have been captured with the correct custom_id values
    assert len(batch.pending_response_ids) == 2
    custom_ids = {e['custom_id'] for e in batch.pending_response_ids.values()}
    assert custom_ids == {'user-001', 'user-002'}


@pytest.mark.asyncio
async def test_retrieve_background_responses_includes_parsed_output():
    """Test that retrieve_background_responses() calls response_parser for completed
    responses and includes parsed_output in the returned DataFrame."""
    from langchain_callback_parquet_logger.background_retrieval import retrieve_background_responses
    from langchain_callback_parquet_logger import RetrievalConfig, ColumnConfig

    # A fake completed response object
    class FakeResponse:
        status = 'completed'
        output_text = '{"city": "Paris"}'

        def model_dump(self, **kwargs):
            return {'status': 'completed', 'output_text': self.output_text}

    mock_client = AsyncMock()
    mock_client.responses.retrieve = AsyncMock(return_value=FakeResponse())

    df = pd.DataFrame({
        'response_id': ['resp_test001'],
        'custom_id': ['cid_001'],
    })

    # response_parser returns a dict (as Batch.retrieve() would build it)
    def response_parser(response_dict: dict):
        return {'city': 'Paris'}

    with tempfile.TemporaryDirectory() as tmpdir:
        results = await retrieve_background_responses(
            df=df,
            openai_client=mock_client,
            retrieval_config=RetrievalConfig(
                show_progress=False,
                return_results=True,
                poll_interval=0,
            ),
            response_parser=response_parser,
        )

    assert results is not None
    assert not results.empty
    assert 'parsed_output' in results.columns
    assert results['parsed_output'].iloc[0] == {'city': 'Paris'}
    assert results['status'].iloc[0] == 'completed'


# ---------------------------------------------------------------------------
# New tests: run_id/parent_run_id/tags/raw flow through background retrieval
# ---------------------------------------------------------------------------

def test_background_run_id_capture_on_llm_end():
    """Test that _BackgroundRunIdCapture.on_llm_end captures run_id and parent_run_id
    when the LLM response contains a background (queued/in_progress) status."""
    from langchain_callback_parquet_logger.batch import _BackgroundRunIdCapture
    from unittest.mock import MagicMock
    import uuid

    capture = _BackgroundRunIdCapture()

    run_id = uuid.uuid4()
    parent_run_id = uuid.uuid4()

    # Build a fake LLMResult with a background-status ChatGeneration
    gen = MagicMock()
    gen.message.response_metadata = {'id': 'resp_run_test001', 'status': 'queued'}
    response = MagicMock()
    response.generations = [[gen]]

    capture.on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id)

    assert 'resp_run_test001' in capture.response_to_run
    info = capture.response_to_run['resp_run_test001']
    assert info['run_id'] == str(run_id)
    assert info['parent_run_id'] == str(parent_run_id)

    # Synchronous (completed) responses must NOT be captured
    gen2 = MagicMock()
    gen2.message.response_metadata = {'id': 'chatcmpl_sync', 'status': 'completed'}
    response2 = MagicMock()
    response2.generations = [[gen2]]

    capture.on_llm_end(response2, run_id=uuid.uuid4())
    assert 'chatcmpl_sync' not in capture.response_to_run


@pytest.mark.asyncio
async def test_run_stores_run_id_and_tags_in_background_ids():
    """Test that Batch.run() stores run_id, parent_run_id, and tags in the
    _background_response_ids dict alongside custom_id."""
    call_count = 0

    async def mock_ainvoke(**kwargs):
        nonlocal call_count
        call_count += 1
        return Mock(
            response_metadata={'id': f'resp_rtag{call_count:03d}', 'status': 'in_progress'},
            content='',
        )

    class MockChatOpenAI:
        def __init__(self, **kwargs):
            self.callbacks = kwargs.get('callbacks', [])
            self.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    MockChatOpenAI.__name__ = 'ChatOpenAI'
    MockChatOpenAI.__module__ = 'test_module'

    df = pd.DataFrame({
        'prompt': ['hello'],
        'config': [with_tags(custom_id='run-tag-user')],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        batch = Batch(
            storage_config=StorageConfig(output_dir=tmpdir),
            processing_config=ProcessingConfig(show_progress=False, return_results=True),
        )
        await batch.run(df, LLMConfig(llm_class=MockChatOpenAI, llm_kwargs={}))

    assert len(batch._background_response_ids) == 1
    entry = next(iter(batch._background_response_ids.values()))
    assert entry['custom_id'] == 'run-tag-user'
    # Tags should include the logger_custom_id tag from with_tags()
    assert any('logger_custom_id' in t for t in entry['tags'])
    # run_id is empty string here because MockChatOpenAI doesn't go through
    # the real LangChain callback chain — just verify the key exists
    assert 'run_id' in entry
    assert 'parent_run_id' in entry


def test_pending_response_ids_returns_full_dict():
    """Test that pending_response_ids returns the full dict (response_id → entry dict)
    rather than a simple string mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        batch = Batch(
            storage_config=StorageConfig(output_dir=tmpdir),
            processing_config=ProcessingConfig(show_progress=False),
        )
        batch._background_response_ids = {
            'resp_A': {'custom_id': 'cid-a', 'run_id': 'uuid-1', 'parent_run_id': '', 'tags': ['x']},
            'resp_B': {'custom_id': 'cid-b', 'run_id': 'uuid-2', 'parent_run_id': '', 'tags': []},
        }

        ids = batch.pending_response_ids
        assert set(ids.keys()) == {'resp_A', 'resp_B'}
        assert ids['resp_A']['custom_id'] == 'cid-a'
        assert ids['resp_A']['run_id'] == 'uuid-1'
        assert ids['resp_A']['tags'] == ['x']
        assert ids['resp_B']['custom_id'] == 'cid-b'


@pytest.mark.asyncio
async def test_retrieval_events_use_original_run_id():
    """Test that retrieve_background_responses() uses the run_id column from the df
    (the original LangChain UUID) as the execution.run_id in logged events."""
    from langchain_callback_parquet_logger.background_retrieval import retrieve_background_responses
    from langchain_callback_parquet_logger import RetrievalConfig

    class FakeResponse:
        status = 'completed'
        output_text = None

        def model_dump(self, **kwargs):
            return {'status': 'completed'}

    mock_client = AsyncMock()
    mock_client.responses.retrieve = AsyncMock(return_value=FakeResponse())

    df = pd.DataFrame({
        'response_id': ['resp_run_link001'],
        'custom_id': ['cid_run'],
        'run_id': ['langchain-uuid-xyz'],
        'parent_run_id': ['parent-uuid-abc'],
        'tags': [['logger_custom_id:cid_run', 'env:test']],
    })

    logged_payloads = []

    class CapturingLogger:
        def _log_event(self, payload):
            logged_payloads.append(payload)

        def flush(self):
            pass

    await retrieve_background_responses(
        df=df,
        openai_client=mock_client,
        logger=CapturingLogger(),
        retrieval_config=RetrievalConfig(show_progress=False, return_results=True, poll_interval=0),
    )

    assert len(logged_payloads) >= 2  # attempt + complete
    for p in logged_payloads:
        assert p['execution']['run_id'] == 'langchain-uuid-xyz', (
            f"Expected run_id='langchain-uuid-xyz', got {p['execution']['run_id']!r}"
        )
        assert p['execution']['parent_run_id'] == 'parent-uuid-abc'
        assert 'logger_custom_id:cid_run' in p['execution']['tags']

    # The complete event should have raw populated with response_data
    complete_events = [p for p in logged_payloads if p['event_type'] == 'background_retrieval_complete']
    assert len(complete_events) == 1
    assert isinstance(complete_events[0]['raw'], dict)
    assert complete_events[0]['raw']  # non-empty


@pytest.mark.asyncio
async def test_retrieval_complete_has_raw_response_data():
    """Test that background_retrieval_complete events have raw populated with
    the full serialized OpenAI response, matching the llm_end raw convention."""
    from langchain_callback_parquet_logger.background_retrieval import retrieve_background_responses
    from langchain_callback_parquet_logger import RetrievalConfig

    class FakeResponse:
        status = 'completed'
        output_text = 'hello'

        def model_dump(self, **kwargs):
            return {'status': 'completed', 'output': [], 'output_text': 'hello'}

    mock_client = AsyncMock()
    mock_client.responses.retrieve = AsyncMock(return_value=FakeResponse())

    df = pd.DataFrame({
        'response_id': ['resp_raw_test'],
        'custom_id': ['raw_cid'],
    })

    logged_payloads = []

    class CapturingLogger:
        def _log_event(self, payload):
            logged_payloads.append(payload)

        def flush(self):
            pass

    await retrieve_background_responses(
        df=df,
        openai_client=mock_client,
        logger=CapturingLogger(),
        retrieval_config=RetrievalConfig(show_progress=False, return_results=True, poll_interval=0),
    )

    complete_events = [p for p in logged_payloads if p['event_type'] == 'background_retrieval_complete']
    assert len(complete_events) == 1
    raw = complete_events[0]['raw']
    assert isinstance(raw, dict)
    assert raw.get('status') == 'completed'
    assert raw.get('output_text') == 'hello'


def test_query_pending_responses_reads_response_id_from_payload():
    """Test that _query_pending_responses() extracts response_id from payload.data
    rather than the run_id column (which now holds the LangChain UUID)."""
    import json
    from langchain_callback_parquet_logger.background_retrieval import _query_pending_responses
    from langchain_callback_parquet_logger.storage import LocalStorage
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a fake Parquet file with run_id = LangChain UUID and
        # payload.data.response_id = OpenAI response_id
        langchain_uuid = 'lc-uuid-abc123'
        openai_resp_id = 'resp_openai_xyz'

        payload = json.dumps({
            'event_type': 'background_retrieval_attempt',
            'data': {'response_id': openai_resp_id},
            'execution': {'run_id': langchain_uuid, 'custom_id': 'cid-test'},
        })

        table = pa.table({
            'timestamp': pa.array([1], type=pa.int64()),
            'run_id': [langchain_uuid],
            'parent_run_id': [''],
            'custom_id': ['cid-test'],
            'event_type': ['background_retrieval_attempt'],
            'logger_metadata': ['{}'],
            'payload': [payload],
        })
        pq.write_table(table, f'{tmpdir}/test.parquet')

        storage = LocalStorage(tmpdir)
        result = _query_pending_responses(storage)

    assert not result.empty
    assert result['response_id'].iloc[0] == openai_resp_id, (
        "Should use response_id from payload, not the LangChain run_id"
    )
    assert result['custom_id'].iloc[0] == 'cid-test'
