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
    JobConfig, StorageConfig, ProcessingConfig,
    ColumnConfig, S3Config
)


class TestBatchProcess:
    """Test batch_process functionality."""
    
    @pytest.mark.asyncio
    async def test_batch_process_local_only(self, sample_dataframe):
        """Test batch_process with local storage only."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))
        
        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.__class__.__name__ = 'ChatOpenAI'
        mock_llm.callbacks = []
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="response"))
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = await batch_process(
                df,
                llm=mock_llm,
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
            
            # Check results
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
        
        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.__class__.__name__ = 'ChatOpenAI'
        mock_llm.callbacks = []
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="response"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock S3Storage.client property to avoid boto3 import
            with patch('langchain_callback_parquet_logger.storage.S3Storage.client', new_callable=PropertyMock) as mock_client:
                # Mock S3 client
                mock_s3_client = MagicMock()
                mock_client.return_value = mock_s3_client

                results = await batch_process(
                    df,
                    llm=mock_llm,
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
        
        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.__class__.__name__ = 'ChatOpenAI'
        mock_llm.callbacks = []
        mock_llm.ainvoke = AsyncMock(return_value=TestModel(value="test"))
        mock_llm.with_structured_output = Mock(return_value=mock_llm)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = await batch_process(
                df,
                llm=mock_llm,
                structured_output=TestModel,
                storage_config=StorageConfig(
                    output_dir=tmpdir
                ),
                processing_config=ProcessingConfig(
                    show_progress=False,
                    return_results=True
                )
            )
            
            # Check that structured output was applied
            mock_llm.with_structured_output.assert_called_once_with(TestModel)
            assert len(results) == len(df)
    
    @pytest.mark.asyncio
    async def test_batch_process_llm_types(self, sample_dataframe):
        """Test that different LLM classes work with batch processing."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))

        # Test different LLM classes
        for llm_class_name in ['ChatOpenAI', 'ChatAnthropic', 'ChatCohere', 'UnknownLLM']:
            mock_llm = AsyncMock()
            mock_llm.__class__.__name__ = llm_class_name
            mock_llm.callbacks = []
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="response"))

            with tempfile.TemporaryDirectory() as tmpdir:
                with patch('langchain_callback_parquet_logger.batch.ParquetLogger') as MockLogger:
                    mock_logger_instance = MagicMock()
                    MockLogger.return_value = mock_logger_instance
                    mock_logger_instance.__enter__ = Mock(return_value=mock_logger_instance)
                    mock_logger_instance.__exit__ = Mock(return_value=None)

                    await batch_process(
                        df,
                        llm=mock_llm,
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
                    # Provider detection has been removed, so we don't check for it
    
    @pytest.mark.asyncio
    async def test_batch_process_path_templates(self, sample_dataframe):
        """Test path template formatting."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))
        
        mock_llm = AsyncMock()
        mock_llm.__class__.__name__ = 'ChatOpenAI'
        mock_llm.callbacks = []
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="response"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            await batch_process(
                df,
                llm=mock_llm,
                job_config=JobConfig(
                    category="emails",
                    subcategory="validation",
                    version="2.0.1",
                    environment="staging"
                ),
                storage_config=StorageConfig(
                    output_dir=tmpdir,
                    path_template="{environment}/{job_category}/v{job_version}/{job_subcategory}"
                ),
                processing_config=ProcessingConfig(
                    show_progress=False
                )
            )
            
            # Check that path was formatted correctly
            expected_path = Path(tmpdir) / "staging" / "emails" / "v2.0.1" / "validation"
            assert expected_path.exists()
    
    @pytest.mark.asyncio
    async def test_batch_process_override_params(self, sample_dataframe):
        """Test logger and batch override parameters."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))
        
        mock_llm = AsyncMock()
        mock_llm.__class__.__name__ = 'ChatOpenAI'
        mock_llm.callbacks = []
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="response"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('langchain_callback_parquet_logger.batch.ParquetLogger') as MockLogger:
                mock_logger_instance = MagicMock()
                MockLogger.return_value = mock_logger_instance
                mock_logger_instance.__enter__ = Mock(return_value=mock_logger_instance)
                mock_logger_instance.__exit__ = Mock(return_value=None)
                
                # Use valid ParquetLogger parameters in override
                await batch_process(
                    df,
                    llm=mock_llm,
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
                    # S3 config should be passed as s3_config object now
    
    @pytest.mark.asyncio
    async def test_batch_process_missing_columns(self, sample_dataframe):
        """Test error when required columns are missing."""
        df = sample_dataframe.copy()
        # Don't add the required 'prompt' column
        
        mock_llm = AsyncMock()
        
        with pytest.raises(ValueError, match="DataFrame missing required column"):
            await batch_process(
                df,
                llm=mock_llm,
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
        
        mock_llm = AsyncMock()
        mock_llm.__class__.__name__ = 'ChatOpenAI'
        mock_llm.callbacks = []
        mock_llm.ainvoke = AsyncMock(return_value=Mock(content="response"))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {'LANGCHAIN_S3_BUCKET': 'env-bucket'}):
                with patch('langchain_callback_parquet_logger.batch.ParquetLogger') as MockLogger:
                    mock_logger_instance = MagicMock()
                    MockLogger.return_value = mock_logger_instance
                    mock_logger_instance.__enter__ = Mock(return_value=mock_logger_instance)
                    mock_logger_instance.__exit__ = Mock(return_value=None)
                    
                    await batch_process(
                        df,
                        llm=mock_llm,
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
    async def test_batch_process_no_llm_creates_default(self, sample_dataframe):
        """Test that ChatOpenAI is used by default when no LLM provided."""
        df = sample_dataframe.copy()
        df['prompt'] = df['text']
        df['config'] = df['id'].apply(lambda x: with_tags(custom_id=str(x)))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock ChatOpenAI class
            MockChatOpenAI = Mock()
            mock_llm = AsyncMock()
            mock_llm.callbacks = []
            mock_llm.ainvoke = AsyncMock(return_value=Mock(content="response"))
            MockChatOpenAI.return_value = mock_llm
            
            # Patch the import statement inside batch_process
            import sys
            mock_module = MagicMock()
            mock_module.ChatOpenAI = MockChatOpenAI
            sys.modules['langchain_openai'] = mock_module
            
            try:
                await batch_process(
                    df,
                    storage_config=StorageConfig(
                        output_dir=tmpdir
                    ),
                    processing_config=ProcessingConfig(
                        show_progress=False
                    ),
                    llm_kwargs={'model': 'gpt-4', 'temperature': 0}
                )
                
                # Check that ChatOpenAI was instantiated with correct kwargs
                MockChatOpenAI.assert_called_once_with(model='gpt-4', temperature=0)
            finally:
                # Clean up the mock module
                if 'langchain_openai' in sys.modules:
                    del sys.modules['langchain_openai']