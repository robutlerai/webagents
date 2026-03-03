#!/bin/bash
#
# x402 Integration Test Runner
#
# Runs comprehensive integration tests for the x402 payment protocol
# Tests Agent A ↔ Agent B payment flows using Robutler tokens
#

set -e

echo "========================================"
echo "x402 Payment Protocol - Integration Tests"
echo "========================================"
echo ""

# Activate virtualenv if needed
if [ -f ~/dev/.venv/bin/activate ]; then
    source ~/dev/.venv/bin/activate
    echo "✅ Activated virtualenv: ~/dev/.venv"
else
    echo "⚠️  No virtualenv found at ~/dev/.venv"
fi

echo ""
echo "Running x402 integration tests..."
echo ""

# Run the integration tests with verbose output
python3 -m pytest tests/integration/test_x402_agent_ab_integration.py \
    -v \
    --tb=short \
    --color=yes \
    "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================"
    echo "✅ All x402 tests passed!"
    echo "========================================"
else
    echo "========================================"
    echo "❌ Some x402 tests failed (exit code: $EXIT_CODE)"
    echo "========================================"
fi

exit $EXIT_CODE

