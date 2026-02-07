"""
Tests for Pydantic v2 model serialization in callbacks.

This test file specifically addresses the bug where Pydantic v2 models
used with structured_output would cause a serialization error:
"argument 'by_alias': 'NoneType' object cannot be converted to 'PyBool'"
"""

import json
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from langchain_callback_parquet_logger import ParquetLogger


# Test Pydantic models
class SimpleModel(BaseModel):
    """Simple Pydantic v2 model for testing."""
    answer: str
    confidence: float


class NestedModel(BaseModel):
    """Pydantic model with nested structure."""
    name: str
    value: int
    metadata: dict


class ComplexModel(BaseModel):
    """Complex Pydantic model with various field types."""
    id: str
    count: int
    tags: List[str]
    nested: SimpleModel
    optional_field: Optional[str] = None
    aliased_field: str = Field(alias="aliasedField")


class TestPydanticChainEndSerialization:
    """Test that on_chain_end properly serializes Pydantic v2 models."""

    def test_chain_end_with_simple_pydantic_model(self):
        """Test that chain_end handles simple Pydantic models without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['chain_end'],
                partition_on=None
            )

            # Create a Pydantic model instance (simulating structured output)
            output = SimpleModel(answer="42", confidence=0.99)

            # Trigger on_chain_end with Pydantic model
            logger.on_chain_end(output, run_id='test-run-123', parent_run_id='parent-456')
            logger.flush()

            # Verify the log was written
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            assert len(parquet_files) == 1, "Expected one parquet file"

            # Read and verify the payload
            df = pd.read_parquet(parquet_files[0])
            assert len(df) == 1, "Expected one log entry"

            payload = json.loads(df.iloc[0]['payload'])

            # Check data section has serialized dict
            assert 'outputs' in payload['data'], "Expected 'outputs' in data section"
            assert isinstance(payload['data']['outputs'], dict), "Expected dict in data section"
            assert payload['data']['outputs']['answer'] == '42'
            assert payload['data']['outputs']['confidence'] == 0.99

            # Check raw section also has serialized dict
            assert 'outputs' in payload['raw'], "Expected 'outputs' in raw section"
            assert isinstance(payload['raw']['outputs'], dict), "Expected dict in raw section"
            assert payload['raw']['outputs']['answer'] == '42'

    def test_chain_end_with_nested_pydantic_model(self):
        """Test that chain_end handles nested Pydantic models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['chain_end'],
                partition_on=None
            )

            # Create a complex model with nesting
            output = ComplexModel(
                id="test-123",
                count=42,
                tags=["tag1", "tag2"],
                nested=SimpleModel(answer="nested answer", confidence=0.85),
                optional_field="optional value",
                aliasedField="alias value"
            )

            logger.on_chain_end(output, run_id='test-run-456')
            logger.flush()

            # Read and verify
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            df = pd.read_parquet(parquet_files[0])
            payload = json.loads(df.iloc[0]['payload'])

            # Verify nested structure is properly serialized
            assert payload['data']['outputs']['id'] == 'test-123'
            assert payload['data']['outputs']['count'] == 42
            assert payload['data']['outputs']['tags'] == ['tag1', 'tag2']
            assert isinstance(payload['data']['outputs']['nested'], dict)
            assert payload['data']['outputs']['nested']['answer'] == 'nested answer'
            assert payload['data']['outputs']['optional_field'] == 'optional value'
            # Note: by_alias=False means we get the Python field name, not the alias
            assert payload['data']['outputs']['aliased_field'] == 'alias value'

    def test_chain_end_with_dict_containing_pydantic_model(self):
        """Test that chain_end handles dicts containing Pydantic models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['chain_end'],
                partition_on=None
            )

            # Dict with Pydantic model as value
            output = {
                "status": "success",
                "result": SimpleModel(answer="test", confidence=1.0),
                "count": 5
            }

            logger.on_chain_end(output, run_id='test-run-789')
            logger.flush()

            # Read and verify
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            df = pd.read_parquet(parquet_files[0])
            payload = json.loads(df.iloc[0]['payload'])

            # Verify the dict structure is preserved and nested model is serialized
            assert payload['data']['outputs']['status'] == 'success'
            assert payload['data']['outputs']['count'] == 5
            assert isinstance(payload['data']['outputs']['result'], dict)
            assert payload['data']['outputs']['result']['answer'] == 'test'
            assert payload['data']['outputs']['result']['confidence'] == 1.0

    def test_chain_end_backward_compatibility_with_dict(self):
        """Test that chain_end still works with regular dicts (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['chain_end'],
                partition_on=None
            )

            # Regular dict (no Pydantic)
            output = {"answer": "42", "score": 0.95}

            logger.on_chain_end(output, run_id='test-run-backward')
            logger.flush()

            # Read and verify
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            df = pd.read_parquet(parquet_files[0])
            payload = json.loads(df.iloc[0]['payload'])

            assert payload['data']['outputs']['answer'] == '42'
            assert payload['data']['outputs']['score'] == 0.95


class TestPydanticToolEndSerialization:
    """Test that on_tool_end properly serializes Pydantic v2 models."""

    def test_tool_end_with_pydantic_model(self):
        """Test that tool_end handles Pydantic models without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['tool_end'],
                partition_on=None
            )

            # Tool returning a Pydantic model
            output = NestedModel(name="result", value=100, metadata={"key": "value"})

            logger.on_tool_end(output, run_id='test-tool-123')
            logger.flush()

            # Read and verify
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            df = pd.read_parquet(parquet_files[0])
            payload = json.loads(df.iloc[0]['payload'])

            # Verify both data and raw sections have serialized output
            assert payload['data']['output']['name'] == 'result'
            assert payload['data']['output']['value'] == 100
            assert payload['raw']['output']['name'] == 'result'

    def test_tool_end_backward_compatibility_with_string(self):
        """Test that tool_end still works with strings (backward compatibility)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['tool_end'],
                partition_on=None
            )

            # Traditional string output
            output = "Tool execution result"

            logger.on_tool_end(output, run_id='test-tool-string')
            logger.flush()

            # Read and verify
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            df = pd.read_parquet(parquet_files[0])
            payload = json.loads(df.iloc[0]['payload'])

            assert payload['data']['output'] == 'Tool execution result'
            assert payload['raw']['output'] == 'Tool execution result'


class TestPydanticDefensiveHandling:
    """Test the defensive Pydantic handling in _safe_json_dumps."""

    def test_safe_json_dumps_with_pydantic_model(self):
        """Test that _safe_json_dumps can handle Pydantic models directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(tmpdir, buffer_size=10)

            # Create a Pydantic model
            model = SimpleModel(answer="test", confidence=0.5)

            # Test the defensive handling - even if a Pydantic model somehow
            # makes it to _safe_json_dumps, it should be handled
            result_str = logger._safe_json_dumps({"data": model})
            result = json.loads(result_str)

            # The model should have been serialized via model_dump
            assert isinstance(result['data'], dict)
            assert result['data']['answer'] == 'test'
            assert result['data']['confidence'] == 0.5

    def test_safe_json_dumps_with_nested_models(self):
        """Test that _safe_json_dumps handles nested Pydantic models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(tmpdir, buffer_size=10)

            # Complex nested structure
            model = ComplexModel(
                id="id-123",
                count=5,
                tags=["a", "b"],
                nested=SimpleModel(answer="nested", confidence=0.7),
                aliasedField="value"
            )

            result_str = logger._safe_json_dumps({"wrapper": {"model": model}})
            result = json.loads(result_str)

            # Verify nested serialization worked
            assert result['wrapper']['model']['id'] == 'id-123'
            assert result['wrapper']['model']['nested']['answer'] == 'nested'


class TestEdgeCases:
    """Test edge cases with Pydantic models."""

    def test_list_of_pydantic_models(self):
        """Test handling of lists containing Pydantic models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['chain_end'],
                partition_on=None
            )

            # List of Pydantic models
            outputs = [
                SimpleModel(answer="first", confidence=0.8),
                SimpleModel(answer="second", confidence=0.9)
            ]

            logger.on_chain_end(outputs, run_id='test-list')
            logger.flush()

            # Read and verify
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            df = pd.read_parquet(parquet_files[0])
            payload = json.loads(df.iloc[0]['payload'])

            # Should be serialized as list of dicts
            assert isinstance(payload['data']['outputs'], list)
            assert len(payload['data']['outputs']) == 2
            assert payload['data']['outputs'][0]['answer'] == 'first'
            assert payload['data']['outputs'][1]['answer'] == 'second'

    def test_pydantic_model_with_none_values(self):
        """Test Pydantic models with None/optional fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ParquetLogger(
                tmpdir,
                buffer_size=1,
                event_types=['chain_end'],
                partition_on=None
            )

            # Model with None optional field
            output = ComplexModel(
                id="test",
                count=1,
                tags=[],
                nested=SimpleModel(answer="a", confidence=0.5),
                optional_field=None,  # Explicitly None
                aliasedField="b"
            )

            logger.on_chain_end(output, run_id='test-none')
            logger.flush()

            # Read and verify
            parquet_files = list(Path(tmpdir).glob("*.parquet"))
            df = pd.read_parquet(parquet_files[0])
            payload = json.loads(df.iloc[0]['payload'])

            # None should be preserved
            assert payload['data']['outputs']['optional_field'] is None
            assert payload['data']['outputs']['tags'] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
