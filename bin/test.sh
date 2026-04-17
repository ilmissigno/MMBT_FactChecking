#!/bin/bash
# =============================================================================
# Test Script
# Esegue i test del progetto
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Attiva ambiente
source "$SCRIPT_DIR/activate_env.sh"

echo ""
echo "🧪 MMBT Tests"
echo "============="

cd "$PROJECT_ROOT"

# Parametri
TEST_PATH="${1:-tests/}"
VERBOSE="${2:--v}"

# Esegui test
python -m pytest "$TEST_PATH" $VERBOSE --tb=short

echo ""
echo "✅ Test completati!"
