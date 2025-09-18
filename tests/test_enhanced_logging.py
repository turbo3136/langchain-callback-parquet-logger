"""
Tests for enhanced event logging features (chains, tools, agents, hierarchy).
"""

import pytest
import json
from pathlib import Path
from datetime import date
from uuid import UUID
import pandas as pd
from unittest.mock import Mock

from langchain_callback_parquet_logger import ParquetLogger


class TestEnhancedEventLogging:
    """Test enhanced event logging features."""
    
    def test_configurable_event_types(self, temp_log_dir):
        """Test that event types can be configured."""
        # Create logger with only chain events enabled
        logger = ParquetLogger(
            temp_log_dir,
            buffer_size=1,
            event_types=['chain_start', 'chain_end']
        )
        
        # Should log chain events
        logger.on_chain_start(
            {'name': 'test_chain'},
            {'input': 'test'},
            run_id='chain-123',
            parent_run_id='parent-456'
        )
        
        # Should NOT log LLM events
        logger.on_llm_start(
            {'kwargs': {'model_name': 'gpt-4'}},
            ['test prompt'],
            run_id='llm-789'
        )
        
        # Check that only chain event was logged
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        assert len(files) == 1
        
        df = pd.read_parquet(files[0])
        assert len(df) == 1
        assert df.iloc[0]['event_type'] == 'chain_start'
    
    def test_chain_events(self, temp_log_dir):
        """Test logging of chain events."""
        logger = ParquetLogger(
            temp_log_dir,
            buffer_size=10,
            event_types=['chain_start', 'chain_end', 'chain_error']
        )
        
        # Test chain start
        logger.on_chain_start(
            {'name': 'test_chain', 'type': 'sequential'},
            {'question': 'What is 2+2?'},
            run_id='chain-start-123',
            parent_run_id='parent-123',
            tags=['test-tag']
        )
        
        # Test chain end
        logger.on_chain_end(
            {'answer': '4', 'confidence': 0.99},
            run_id='chain-end-123',
            parent_run_id='parent-123'
        )
        
        # Test chain error
        test_error = ValueError("Chain processing failed")
        logger.on_chain_error(
            test_error,
            run_id='chain-error-123',
            parent_run_id='parent-123'
        )
        
        logger.flush()
        
        # Verify events were logged correctly
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        assert len(df) == 3
        
        # Check chain_start
        start_event = df[df['event_type'] == 'chain_start'].iloc[0]
        assert start_event['run_id'] == 'chain-start-123'
        assert start_event['parent_run_id'] == 'parent-123'
        payload = json.loads(start_event['payload'])
        assert payload['data']['model'] == 'test_chain'
        assert payload['data']['inputs']['question'] == 'What is 2+2?'
        
        # Check chain_end
        end_event = df[df['event_type'] == 'chain_end'].iloc[0]
        assert end_event['run_id'] == 'chain-end-123'
        payload = json.loads(end_event['payload'])
        assert payload['data']['outputs']['answer'] == '4'
        
        # Check chain_error
        error_event = df[df['event_type'] == 'chain_error'].iloc[0]
        assert error_event['run_id'] == 'chain-error-123'
        payload = json.loads(error_event['payload'])
        assert 'Chain processing failed' in payload['data']['error']['message']
    
    def test_tool_events(self, temp_log_dir):
        """Test logging of tool events."""
        logger = ParquetLogger(
            temp_log_dir,
            buffer_size=10,
            event_types=['tool_start', 'tool_end', 'tool_error']
        )
        
        # Test tool start
        logger.on_tool_start(
            {'name': 'calculator', 'description': 'Performs math operations'},
            '2 + 2',
            run_id='tool-start-123',
            parent_run_id='llm-123'
        )
        
        # Test tool end
        logger.on_tool_end(
            '4',
            run_id='tool-end-123',
            parent_run_id='llm-123'
        )
        
        # Test tool error
        test_error = RuntimeError("Calculator malfunction")
        logger.on_tool_error(
            test_error,
            run_id='tool-error-123',
            parent_run_id='llm-123'
        )
        
        logger.flush()
        
        # Verify events
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        assert len(df) == 3
        
        # Check tool_start
        start_event = df[df['event_type'] == 'tool_start'].iloc[0]
        payload = json.loads(start_event['payload'])
        assert payload['data']['model'] == 'calculator'
        assert payload['data']['input_str'] == '2 + 2'
        
        # Check tool_end
        end_event = df[df['event_type'] == 'tool_end'].iloc[0]
        payload = json.loads(end_event['payload'])
        assert payload['data']['output'] == '4'
        
        # Check tool_error
        error_event = df[df['event_type'] == 'tool_error'].iloc[0]
        payload = json.loads(error_event['payload'])
        assert 'Calculator malfunction' in payload['data']['error']['message']
    
    def test_agent_events(self, temp_log_dir):
        """Test logging of agent events."""
        logger = ParquetLogger(
            temp_log_dir,
            buffer_size=10,
            event_types=['agent_action', 'agent_finish']
        )
        
        # Test agent action with mock AgentAction
        action = Mock()
        action.tool = 'search'
        action.tool_input = 'weather today'
        action.log = 'Searching for weather information'
        
        logger.on_agent_action(
            action,
            run_id='agent-action-123',
            parent_run_id='agent-123'
        )
        
        # Test agent finish with mock AgentFinish
        finish = Mock()
        finish.return_values = {'answer': 'Sunny, 72°F'}
        finish.log = 'Found weather information'
        
        logger.on_agent_finish(
            finish,
            run_id='agent-finish-123',
            parent_run_id='agent-123'
        )
        
        logger.flush()
        
        # Verify events
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        assert len(df) == 2
        
        # Check agent_action
        action_event = df[df['event_type'] == 'agent_action'].iloc[0]
        payload = json.loads(action_event['payload'])
        assert payload['data']['action']['tool'] == 'search'
        assert payload['data']['action']['tool_input'] == 'weather today'
        
        # Check agent_finish
        finish_event = df[df['event_type'] == 'agent_finish'].iloc[0]
        payload = json.loads(finish_event['payload'])
        assert payload['data']['finish']['return_values']['answer'] == 'Sunny, 72°F'
    
    def test_parent_run_id_hierarchy(self, temp_log_dir):
        """Test that parent_run_id properly tracks hierarchy."""
        logger = ParquetLogger(
            temp_log_dir,
            buffer_size=10,
            event_types=[
                'chain_start', 'llm_start', 'tool_start',
                'tool_end', 'llm_end', 'chain_end'
            ]
        )
        
        # Simulate hierarchical execution: Chain -> LLM -> Tool
        logger.on_chain_start(
            {'name': 'main_chain'},
            {'question': 'Calculate 5 * 7'},
            run_id='chain-1',
            parent_run_id=None  # Top-level
        )
        
        logger.on_llm_start(
            {'kwargs': {'model_name': 'gpt-4'}},
            ['I need to calculate 5 * 7'],
            run_id='llm-1',
            parent_run_id='chain-1'  # Child of chain
        )
        
        logger.on_tool_start(
            {'name': 'calculator'},
            '5 * 7',
            run_id='tool-1',
            parent_run_id='llm-1'  # Child of LLM
        )
        
        logger.on_tool_end(
            '35',
            run_id='tool-1',
            parent_run_id='llm-1'
        )
        
        logger.on_llm_end(
            Mock(llm_output={'token_usage': {'total': 20}}),
            run_id='llm-1',
            parent_run_id='chain-1'
        )
        
        logger.on_chain_end(
            {'result': '35'},
            run_id='chain-1',
            parent_run_id=None
        )
        
        logger.flush()
        
        # Verify hierarchy
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        assert len(df) == 6
        
        # Check hierarchy relationships
        chain_events = df[df['run_id'] == 'chain-1']
        assert all(chain_events['parent_run_id'] == '')
        
        llm_events = df[df['run_id'] == 'llm-1']
        assert all(llm_events['parent_run_id'] == 'chain-1')
        
        tool_events = df[df['run_id'] == 'tool-1']
        assert all(tool_events['parent_run_id'] == 'llm-1')
    
    def test_backward_compatibility(self, temp_log_dir):
        """Test that default behavior maintains backward compatibility."""
        # Logger without event_types should default to LLM events only
        logger = ParquetLogger(temp_log_dir, buffer_size=10)
        
        # Should log LLM events
        logger.on_llm_start(
            {'kwargs': {'model_name': 'gpt-4'}},
            ['test'],
            run_id='llm-1'
        )
        
        # Should NOT log chain events (not in default)
        logger.on_chain_start(
            {'name': 'chain'},
            {'input': 'test'},
            run_id='chain-1'
        )
        
        logger.flush()
        
        # Only LLM event should be logged
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        assert len(df) == 1
        assert df.iloc[0]['event_type'] == 'llm_start'
    
    def test_parent_run_id_column_always_present(self, temp_log_dir):
        """Test that parent_run_id column is always present even when empty."""
        logger = ParquetLogger(temp_log_dir, buffer_size=1)
        
        # Log event without parent_run_id
        logger.on_llm_start(
            {'kwargs': {'model_name': 'gpt-4'}},
            ['test'],
            run_id='llm-1'
            # No parent_run_id provided
        )
        
        # Check column exists and is empty string
        files = list(Path(temp_log_dir).glob("**/*.parquet"))
        df = pd.read_parquet(files[0])
        
        assert 'parent_run_id' in df.columns
        assert df.iloc[0]['parent_run_id'] == ''