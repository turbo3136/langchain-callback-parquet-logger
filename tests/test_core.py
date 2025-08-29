"""
Tests for core ParquetLogger and with_tags functionality.
"""

import pytest
import json
from pathlib import Path
from datetime import date
import pandas as pd
from unittest.mock import patch, Mock

from langchain_callback_parquet_logger import ParquetLogger, with_tags


class TestParquetLogger:
    """Test ParquetLogger functionality."""
    
    def test_basic_logging(self, temp_log_dir, mock_callback_events):
        """Test that basic logging creates parquet files."""
        logger = ParquetLogger(temp_log_dir, buffer_size=1)
        
        # Simulate LLM start event
        logger.on_llm_start(
            mock_callback_events['llm_start']['serialized'],
            mock_callback_events['llm_start']['prompts'],
            run_id=mock_callback_events['llm_start']['run_id'],
            tags=mock_callback_events['llm_start']['tags'],
            metadata=mock_callback_events['llm_start']['metadata']
        )
        
        # Check that a parquet file was created
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 1
        
        # Read and verify the content
        df = pd.read_parquet(files[0])
        assert len(df) == 1
        assert df.iloc[0]['event_type'] == 'llm_start'
        assert df.iloc[0]['run_id'] == 'test-run-123'
    
    def test_buffer_flush_at_threshold(self, temp_log_dir):
        """Test that buffer flushes when reaching threshold."""
        logger = ParquetLogger(temp_log_dir, buffer_size=2)
        
        # Add one entry - should not flush
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'logger_custom_id': '',
            'event_type': 'test1',
            'provider': 'test',
            'logger_metadata': '{}',
            'payload': '{}'
        })
        
        # No files yet
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 0
        
        # Add second entry - should trigger flush
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-2',
            'logger_custom_id': '',
            'event_type': 'test2',
            'provider': 'test',
            'logger_metadata': '{}',
            'payload': '{}'
        })
        
        # Now file should exist
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 1
    
    def test_manual_flush(self, temp_log_dir):
        """Test manual flush works correctly."""
        logger = ParquetLogger(temp_log_dir, buffer_size=100)
        
        # Add entry
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'logger_custom_id': '',
            'event_type': 'test',
            'provider': 'test',
            'logger_metadata': '{}',
            'payload': '{}'
        })
        
        # Manual flush
        logger.flush()
        
        # File should exist
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 1
    
    def test_context_manager(self, temp_log_dir):
        """Test context manager auto-flushes on exit."""
        with ParquetLogger(temp_log_dir, buffer_size=100) as logger:
            logger._add_entry({
                'timestamp': pd.Timestamp.now('UTC'),
                'run_id': 'run-1',
                'logger_custom_id': '',
                'event_type': 'test',
                'provider': 'test',
                'logger_metadata': '{}',
                'payload': '{}'
            })
            # No file yet (buffer not full)
            files = list(Path(temp_log_dir).glob("**/*.parquet"))
            assert len(files) == 0
        
        # After context exit, file should exist
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 1
    
    def test_partitioning_date(self, temp_log_dir):
        """Test date partitioning creates correct directories."""
        logger = ParquetLogger(temp_log_dir, buffer_size=1, partition_on="date")
        
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'logger_custom_id': '',
            'event_type': 'test',
            'provider': 'test',
            'logger_metadata': '{}',
            'payload': '{}'
        })
        
        # Check date partition exists
        today = date.today()
        partition_dir = Path(temp_log_dir) / f"date={today}"
        assert partition_dir.exists()
        
        files = list(partition_dir.glob("*.parquet"))
        assert len(files) == 1
    
    def test_partitioning_none(self, temp_log_dir):
        """Test no partitioning saves directly to log_dir."""
        logger = ParquetLogger(temp_log_dir, buffer_size=1, partition_on=None)
        
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'logger_custom_id': '',
            'event_type': 'test',
            'provider': 'test',
            'logger_metadata': '{}',
            'payload': '{}'
        })
        
        # File should be directly in log_dir
        files = list(Path(temp_log_dir).glob("*.parquet"))
        assert len(files) == 1
        
        # No subdirectories
        subdirs = [d for d in Path(temp_log_dir).iterdir() if d.is_dir()]
        assert len(subdirs) == 0
    
    def test_logger_metadata(self, temp_log_dir):
        """Test logger metadata persists in logs."""
        metadata = {"env": "test", "version": "1.0"}
        logger = ParquetLogger(
            temp_log_dir, 
            buffer_size=1,
            logger_metadata=metadata
        )
        
        logger._add_entry({
            'timestamp': pd.Timestamp.now('UTC'),
            'run_id': 'run-1',
            'logger_custom_id': '',
            'event_type': 'test',
            'provider': 'test',
            'logger_metadata': json.dumps(metadata),
            'payload': '{}'
        })
        
        # Read and verify metadata
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        
        saved_metadata = json.loads(df.iloc[0]['logger_metadata'])
        assert saved_metadata == metadata
    
    def test_all_event_types(self, temp_log_dir, mock_callback_events):
        """Test all event types are logged correctly."""
        logger = ParquetLogger(temp_log_dir, buffer_size=10)
        
        # Test llm_start
        logger.on_llm_start(
            mock_callback_events['llm_start']['serialized'],
            mock_callback_events['llm_start']['prompts'],
            run_id=mock_callback_events['llm_start']['run_id'],
            tags=mock_callback_events['llm_start']['tags'],
            metadata=mock_callback_events['llm_start']['metadata']
        )
        
        # Test llm_end
        logger.on_llm_end(
            mock_callback_events['llm_end']['response'],
            run_id=mock_callback_events['llm_end']['run_id']
        )
        
        # Test llm_error
        logger.on_llm_error(
            mock_callback_events['llm_error']['error'],
            run_id=mock_callback_events['llm_error']['run_id']
        )
        
        # Flush and read
        logger.flush()
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        
        # Check all three events
        assert len(df) == 3
        event_types = df['event_type'].tolist()
        assert 'llm_start' in event_types
        assert 'llm_end' in event_types
        assert 'llm_error' in event_types
    
    def test_custom_id_extraction(self, temp_log_dir):
        """Test custom ID extraction from tags."""
        logger = ParquetLogger(temp_log_dir, buffer_size=1)
        
        # Create event with custom ID in tags
        logger.on_llm_start(
            {'kwargs': {'model_name': 'test'}},
            ['Test prompt'],
            run_id='test-run',
            tags=['tag1', 'logger_custom_id:my-custom-id', 'tag2']
        )
        
        # Read and verify
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        
        assert df.iloc[0]['logger_custom_id'] == 'my-custom-id'


class TestWithTags:
    """Test with_tags helper function."""
    
    def test_with_tags_custom_id(self):
        """Test with_tags adds custom_id as tag."""
        config = with_tags(custom_id="test-123")
        
        assert 'tags' in config
        assert 'logger_custom_id:test-123' in config['tags']
    
    def test_with_tags_multiple(self):
        """Test with_tags handles multiple tags."""
        config = with_tags("tag1", "tag2", custom_id="test-123")
        
        assert 'tags' in config
        assert "tag1" in config['tags']
        assert "tag2" in config['tags']
        assert 'logger_custom_id:test-123' in config['tags']
    
    def test_with_tags_extend_config(self):
        """Test with_tags extends existing config."""
        existing = {'tags': ['existing'], 'metadata': {'key': 'value'}}
        config = with_tags("new-tag", custom_id="test-123", config=existing)
        
        assert 'existing' in config['tags']
        assert 'new-tag' in config['tags']
        assert 'logger_custom_id:test-123' in config['tags']
        assert config['metadata'] == {'key': 'value'}
    
    def test_with_tags_replace(self):
        """Test replace_tags=True replaces existing tags."""
        existing = {'tags': ['old-tag']}
        config = with_tags("new-tag", config=existing, replace_tags=True)
        
        assert 'old-tag' not in config['tags']
        assert 'new-tag' in config['tags']
    
    def test_with_tags_no_custom_id(self):
        """Test with_tags works without custom_id."""
        config = with_tags("tag1", "tag2")
        
        assert 'tags' in config
        assert "tag1" in config['tags']
        assert "tag2" in config['tags']
        assert not any('logger_custom_id:' in tag for tag in config['tags'])
    
    def test_with_tags_list_parameter(self):
        """Test with_tags accepts tags as list."""
        config = with_tags(tags=["tag1", "tag2"], custom_id="test")
        
        assert "tag1" in config['tags']
        assert "tag2" in config['tags']
        assert 'logger_custom_id:test' in config['tags']