#!/bin/bash
# =============================================================================
# Setup Environment Script
# Per esecuzione diretta su sistema con NVIDIA GPU (es. container LXC)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 MMBT Fact-Checking Setup"
echo "=============================="
echo "Project root: $PROJECT_ROOT"

# Verifica NVIDIA
echo ""
echo "📊 Verifica GPU NVIDIA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo "✅ GPU NVIDIA rilevata"
else
    echo "⚠️  nvidia-smi non trovato. Continuo senza GPU..."
fi

# Verifica Python
echo ""
echo "🐍 Verifica Python..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python non trovato!"
    exit 1
fi
echo "Python: $($PYTHON_CMD --version)"

# Crea virtual environment se non esiste
VENV_PATH="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo ""
    echo "📦 Creazione virtual environment..."
    $PYTHON_CMD -m venv "$VENV_PATH"
fi

# Attiva venv
echo ""
echo "🔧 Attivazione virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Installa dipendenze
echo ""
echo "📥 Installazione dipendenze..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# Installa progetto in modalità editable
echo ""
echo "📦 Installazione progetto..."
pip install -e "$PROJECT_ROOT"

# Aggiungi lib al PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/lib:$PYTHONPATH"

echo ""
echo "✅ Setup completato!"
echo ""
echo "Per attivare l'ambiente:"
echo "  source $VENV_PATH/bin/activate"
echo "  export PYTHONPATH=\"$PROJECT_ROOT:\$PROJECT_ROOT/lib:\$PYTHONPATH\""
echo ""
echo "Oppure usa: source bin/activate_env.sh"
