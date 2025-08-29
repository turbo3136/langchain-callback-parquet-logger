#!/bin/bash
# Simple test runner script

echo "🧪 Running tests for langchain-callback-parquet-logger"
echo "="*60

# Install test dependencies if needed
echo "📦 Installing test dependencies..."
pip install -q pytest pytest-asyncio pytest-mock pandas

# Run tests
echo ""
echo "🏃 Running tests..."
pytest tests/ -v --tb=short

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
else
    echo ""
    echo "❌ Some tests failed"
    exit 1
fi