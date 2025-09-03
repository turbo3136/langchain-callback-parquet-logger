"""Tests for S3 integration in ParquetLogger."""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
import tempfile
import pyarrow.parquet as pq
from io import BytesIO

from langchain_callback_parquet_logger import ParquetLogger


class TestS3Integration:
    """Test S3 upload functionality."""
    
    @patch('langchain_callback_parquet_logger.logger.HAS_BOTO3', True)
    @patch('langchain_callback_parquet_logger.logger.boto3')
    def test_s3_upload_success(self, mock_boto3, mock_llm):
        """Test successful S3 upload."""
        # Setup mock S3 client
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create logger with S3 configuration
            logger = ParquetLogger(
                log_dir=tmpdir,
                buffer_size=1,
                s3_bucket="test-bucket",
                s3_prefix="test-prefix/",
                s3_on_failure="error"
            )
            
            # Trigger a log entry
            logger.on_llm_start(
                serialized={"name": "test"},
                prompts=["test prompt"],
                run_id="test-run-id"
            )
            
            # Verify S3 upload was called
            assert mock_s3_client.put_object.called
            call_args = mock_s3_client.put_object.call_args[1]
            assert call_args['Bucket'] == "test-bucket"
            assert call_args['Key'].startswith("test-prefix/")
            assert isinstance(call_args['Body'], bytes)
    
    @patch('langchain_callback_parquet_logger.logger.HAS_BOTO3', True)
    @patch('langchain_callback_parquet_logger.logger.boto3')
    def test_s3_upload_with_retry(self, mock_boto3, mock_llm):
        """Test S3 upload with retry on failure."""
        # Setup mock S3 client that fails twice then succeeds
        mock_s3_client = MagicMock()
        mock_s3_client.put_object.side_effect = [
            Exception("Network error"),
            Exception("Timeout"),
            None  # Success on third attempt
        ]
        mock_boto3.client.return_value = mock_s3_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                log_dir=tmpdir,
                buffer_size=1,
                s3_bucket="test-bucket",
                s3_retry_attempts=3,
                s3_on_failure="error"
            )
            
            # Trigger a log entry
            logger.on_llm_start(
                serialized={"name": "test"},
                prompts=["test prompt"],
                run_id="test-run-id"
            )
            
            # Verify S3 upload was retried 3 times
            assert mock_s3_client.put_object.call_count == 3
    
    @patch('langchain_callback_parquet_logger.logger.HAS_BOTO3', True)
    @patch('langchain_callback_parquet_logger.logger.boto3')
    def test_s3_upload_failure_error_mode(self, mock_boto3, mock_llm):
        """Test S3 upload failure in error mode."""
        # Setup mock S3 client that always fails
        mock_s3_client = MagicMock()
        mock_s3_client.put_object.side_effect = Exception("Persistent error")
        mock_boto3.client.return_value = mock_s3_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                log_dir=tmpdir,
                buffer_size=1,
                s3_bucket="test-bucket",
                s3_retry_attempts=2,
                s3_on_failure="error"
            )
            
            # Should raise RuntimeError when S3 upload fails
            with pytest.raises(RuntimeError, match="Failed to upload to S3"):
                logger.on_llm_start(
                    serialized={"name": "test"},
                    prompts=["test prompt"],
                    run_id="test-run-id"
                )
    
    @patch('langchain_callback_parquet_logger.logger.HAS_BOTO3', True)
    @patch('langchain_callback_parquet_logger.logger.boto3')
    def test_s3_upload_failure_continue_mode(self, mock_boto3, mock_llm, capsys):
        """Test S3 upload failure in continue mode."""
        # Setup mock S3 client that always fails
        mock_s3_client = MagicMock()
        mock_s3_client.put_object.side_effect = Exception("Persistent error")
        mock_boto3.client.return_value = mock_s3_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                log_dir=tmpdir,
                buffer_size=1,
                s3_bucket="test-bucket",
                s3_retry_attempts=2,
                s3_on_failure="continue"
            )
            
            # Should not raise error, just print warning
            logger.on_llm_start(
                serialized={"name": "test"},
                prompts=["test prompt"],
                run_id="test-run-id"
            )
            
            # Check that error was printed
            captured = capsys.readouterr()
            assert "S3 upload failed (continuing)" in captured.out
            
            # Verify local file still exists
            files = list(Path(tmpdir).rglob("*.parquet"))
            assert len(files) == 1
    
    def test_s3_disabled_by_default(self, mock_llm):
        """Test that S3 is not used when not configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                log_dir=tmpdir,
                buffer_size=1
            )
            
            # No S3 bucket configured
            assert logger.s3_bucket is None
            
            # Should work normally without S3
            logger.on_llm_start(
                serialized={"name": "test"},
                prompts=["test prompt"],
                run_id="test-run-id"
            )
            
            # Verify local file exists
            files = list(Path(tmpdir).rglob("*.parquet"))
            assert len(files) == 1
    
    def test_s3_import_error_when_boto3_missing(self):
        """Test that ImportError is raised when boto3 is not available."""
        with patch('langchain_callback_parquet_logger.logger.HAS_BOTO3', False):
            with tempfile.TemporaryDirectory() as tmpdir:
                with pytest.raises(ImportError, match="boto3 is required for S3 support"):
                    ParquetLogger(
                        log_dir=tmpdir,
                        s3_bucket="test-bucket"
                    )
    
    @patch('langchain_callback_parquet_logger.logger.HAS_BOTO3', True)
    @patch('langchain_callback_parquet_logger.logger.boto3')
    def test_s3_key_structure_with_partitioning(self, mock_boto3, mock_llm):
        """Test S3 key structure with date partitioning."""
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                log_dir=tmpdir,
                buffer_size=1,
                s3_bucket="test-bucket",
                s3_prefix="logs/",
                partition_on="date"
            )
            
            logger.on_llm_start(
                serialized={"name": "test"},
                prompts=["test prompt"],
                run_id="test-run-id"
            )
            
            # Verify S3 key includes date partition
            call_args = mock_s3_client.put_object.call_args[1]
            key = call_args['Key']
            assert key.startswith("logs/date=")
            assert ".parquet" in key
    
    @patch('langchain_callback_parquet_logger.logger.HAS_BOTO3', True)
    @patch('langchain_callback_parquet_logger.logger.boto3')
    def test_s3_key_structure_without_partitioning(self, mock_boto3, mock_llm):
        """Test S3 key structure without partitioning."""
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                log_dir=tmpdir,
                buffer_size=1,
                s3_bucket="test-bucket",
                s3_prefix="flat-logs/",
                partition_on=None
            )
            
            logger.on_llm_start(
                serialized={"name": "test"},
                prompts=["test prompt"],
                run_id="test-run-id"
            )
            
            # Verify S3 key doesn't include date partition
            call_args = mock_s3_client.put_object.call_args[1]
            key = call_args['Key']
            assert key.startswith("flat-logs/logs_")
            assert "date=" not in key
            assert ".parquet" in key