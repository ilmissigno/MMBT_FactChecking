#!/bin/bash
# =============================================================================
# Activate Environment Script
# Attiva l'ambiente virtuale e configura PYTHONPATH
# =============================================================================

# Compatibile sia con bash che sh
if [ -n "$BASH_SOURCE" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Carica .env se esiste
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    . "$PROJECT_ROOT/.env"
    set +a
fi

# Attiva virtual environment
if [ -d "$PROJECT_ROOT/.venv" ]; then
    . "$PROJECT_ROOT/.venv/bin/activate"
    echo "✅ Virtual environment attivato"
else
    echo "⚠️  Virtual environment non trovato. Esegui prima: ./bin/setup_env.sh"
    return 1 2>/dev/null || exit 1
fi

# Configura PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/lib:$PYTHONPATH"
export PROJECT_ROOT="$PROJECT_ROOT"

# Configura CUDA se disponibile
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

echo "📁 PROJECT_ROOT: $PROJECT_ROOT"
echo "🐍 Python: $(which python)"
