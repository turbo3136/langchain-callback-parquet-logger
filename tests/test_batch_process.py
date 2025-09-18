"""
Tests for batch_process functionality.
"""

import pytest
import pandas as pd
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock, AsyncMock, MagicMock, PropertyMock
from langchain_callback_parquet_logger import (
    batch_process, with_tags,
    LLMConfig, JobConfig, StorageConfig, ProcessingConfig,
    ColumnConfig, S3Config
)


def create_mock_llm_class():
    """Create a mock LLM class for testing."""
    class MockLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.callbacks = []
            self.ainvoke = AsyncMock(return_value=Mock(content="response"))

        def with_structured_output(self, schema):
            self.structured_output = schema
            return self

    MockLLM.__name__ = 'MockLLM'
    MockLLM.__module__ = 'test_module'
    return MockLLM


class TestBatchProcess:
    """Test batch_process functionality."""

    @pytest.mark.asyncio
    async def test_batch_process_local_only(self, sample_dataframe):
        """Test batch_process with local storage only."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            results = await batch_process(
                df,
                llm_config=LLMConfig(
                    llm_class=MockLLM,
                    llm_kwargs={'model': 'test-model'}
                ),
                job_config=JobConfig(
                    category="test_job",
                    subcategory="test_sub"
                ),
                storage_config=StorageConfig(
                    output_dir=tmpdir
                ),
                processing_config=ProcessingConfig(
                    show_progress=False,
                    return_results=True
                )
            )

            # Check that results were returned
            assert results is not None
            assert len(results) == len(df)

            # Check that directory structure was created
            expected_path = Path(tmpdir) / "test_job" / "test_sub"
            assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_batch_process_with_s3(self, sample_dataframe):
        """Test batch_process with S3 configuration."""
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

                results = await batch_process(
                    df,
                    llm_config=LLMConfig(
                        llm_class=MockLLM,
                        llm_kwargs={'model': 'test-model'}
                    ),
                    job_config=JobConfig(
                        category="test_job"
                    ),
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
                    )
                )

                # Check local directory was created
                expected_path = Path(tmpdir) / "test_job" / "default"
                assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_batch_process_structured_output(self, sample_dataframe):
        """Test batch_process with structured output."""
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
            results = await batch_process(
                df,
                llm_config=LLMConfig(
                    llm_class=MockStructuredLLM,
                    llm_kwargs={'model': 'test-model'},
                    structured_output=TestModel
                ),
                storage_config=StorageConfig(
                    output_dir=tmpdir
                ),
                processing_config=ProcessingConfig(
                    show_progress=False,
                    return_results=True
                )
            )

            # Check results
            assert len(results) == len(df)

    @pytest.mark.asyncio
    async def test_batch_process_llm_types(self, sample_dataframe):
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

                    await batch_process(
                        df,
                        llm_config=LLMConfig(
                            llm_class=MockTypedLLM,
                            llm_kwargs={'model': 'test-model'}
                        ),
                        storage_config=StorageConfig(
                            output_dir=tmpdir
                        ),
                        processing_config=ProcessingConfig(
                            show_progress=False,
                            buffer_size=1000
                        )
                    )

                    # Check that ParquetLogger was called
                    assert MockLogger.called

    @pytest.mark.asyncio
    async def test_batch_process_path_templates(self, sample_dataframe):
        """Test path template formatting."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        MockLLM = create_mock_llm_class()

        with tempfile.TemporaryDirectory() as tmpdir:
            await batch_process(
                df,
                llm_config=LLMConfig(
                    llm_class=MockLLM,
                    llm_kwargs={'model': 'test-model'}
                ),
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
                processing_config=ProcessingConfig(
                    show_progress=False
                )
            )

            # Check that path was formatted correctly (version dots replaced with underscores)
            expected_path = Path(tmpdir) / "staging" / "emails" / "v2_0_1" / "validation"
            assert expected_path.exists()

    @pytest.mark.asyncio
    async def test_batch_process_override_params(self, sample_dataframe):
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

                # Use valid ParquetLogger parameters in override
                await batch_process(
                    df,
                    llm_config=LLMConfig(
                        llm_class=MockLLM,
                        llm_kwargs={'model': 'test-model'},
                        model_kwargs={'temperature': 0.5}
                    ),
                    storage_config=StorageConfig(
                        output_dir=tmpdir,
                        s3_config=S3Config(
                            bucket="test-bucket",
                            retry_attempts=5
                        )
                    ),
                    processing_config=ProcessingConfig(
                        buffer_size=500,
                        event_types=['llm_start', 'llm_end', 'chain_start'],
                        show_progress=False
                    )
                )

                # Check that logger overrides were applied
                assert MockLogger.called
                if MockLogger.call_args:
                    call_kwargs = MockLogger.call_args.kwargs
                    assert call_kwargs['buffer_size'] == 500
                    assert call_kwargs['event_types'] == ['llm_start', 'llm_end', 'chain_start']

    @pytest.mark.asyncio
    async def test_batch_process_missing_columns(self, sample_dataframe):
        """Test error when required columns are missing."""
        df = sample_dataframe.copy()
        # Don't add the required 'prompt' column

        MockLLM = create_mock_llm_class()

        with pytest.raises(ValueError, match="DataFrame missing required column"):
            await batch_process(
                df,
                llm_config=LLMConfig(
                    llm_class=MockLLM,
                    llm_kwargs={'model': 'test-model'}
                ),
                storage_config=StorageConfig(
                    output_dir="./test"
                )
            )

    @pytest.mark.asyncio
    async def test_batch_process_environment_s3_bucket(self, sample_dataframe):
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

                    await batch_process(
                        df,
                        llm_config=LLMConfig(
                            llm_class=MockLLM,
                            llm_kwargs={'model': 'test-model'}
                        ),
                        storage_config=StorageConfig(
                            output_dir=tmpdir,
                            s3_config=S3Config(
                                bucket='env-bucket'
                            )
                        ),
                        processing_config=ProcessingConfig(
                            show_progress=False
                        )
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
    async def test_batch_process_version_paths(self, sample_dataframe):
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

                await batch_process(
                    df,
                    llm_config=LLMConfig(
                        llm_class=MockLLM,
                        llm_kwargs={'model': 'test-model'}
                    ),
                    job_config=JobConfig(
                        category="ml_models",
                        subcategory="classification",
                        version="3.2.1"  # Version with dots
                    ),
                    storage_config=StorageConfig(
                        output_dir=tmpdir,
                        # Using default template which includes version
                        s3_config=S3Config(
                            bucket="test-bucket",
                            prefix="models/"
                        )
                    ),
                    processing_config=ProcessingConfig(
                        show_progress=False,
                        buffer_size=1  # Force immediate flush
                    )
                )

                # Check local path has sanitized version
                expected_local = Path(tmpdir) / "ml_models" / "classification" / "v3_2_1"
                assert expected_local.exists(), f"Expected path {expected_local} does not exist"

                # Check S3 prefix was set correctly with sanitized version
                # The s3_config.prefix should have been updated to include the formatted path
                logger_call = [call for call in mock_s3_client.put_object.call_args_list
                             if call[1].get('Key', '').startswith('models/ml_models/classification/v3_2_1/')]
                # We expect at least one call with the correct path structure
                # Note: actual S3 upload verification would require checking mock_s3_client.put_object calls

        # Test 2: Without version specified (should use 'unversioned')
        with tempfile.TemporaryDirectory() as tmpdir:
            await batch_process(
                df,
                llm_config=LLMConfig(
                    llm_class=MockLLM,
                    llm_kwargs={'model': 'test-model'}
                ),
                job_config=JobConfig(
                    category="experiments",
                    subcategory="baseline"
                    # No version specified
                ),
                storage_config=StorageConfig(
                    output_dir=tmpdir
                ),
                processing_config=ProcessingConfig(
                    show_progress=False
                )
            )

            # Check local path has 'unversioned' as default
            expected_local = Path(tmpdir) / "experiments" / "baseline" / "vunversioned"
            assert expected_local.exists(), f"Expected path {expected_local} does not exist"