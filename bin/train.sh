#!/bin/bash
# =============================================================================
# MMBT Training Script
# Esegue il training con gestione automatica del dataset/config
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Attiva ambiente
source "$SCRIPT_DIR/activate_env.sh"

# Help
show_help() {
    echo ""
    echo "🏋️ MMBT Training"
    echo "================"
    echo ""
    echo "Uso: $0 [opzioni]"
    echo ""
    echo "Opzioni:"
    echo "  --dataset NAME   - Dataset da addestrare (snopes/politifact/mumin/m3check)"
    echo "  --config FILE    - File config personalizzato (override dataset)"
    echo "  --resume         - Riprendi training da checkpoint esistente"
    echo "  --epochs N       - Numero di epoche (override config)"
    echo "  --help           - Mostra questo messaggio"
    echo ""
    echo "Esempi:"
    echo "  $0 --dataset snopes            # Training Snopes"
    echo "  $0 --dataset politifact        # Training Politifact"
    echo "  $0 --config configs/custom.conf"
    echo ""
    echo "Dataset disponibili e relativi config:"
    echo "  snopes     -> configs/snopes_config.conf"
    echo "  politifact -> configs/politifact_config.conf"
    echo "  mumin      -> configs/mumin_config.conf"
    echo ""
}

# Determina config da dataset
get_config_for_dataset() {
    local dataset_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    case "$dataset_lower" in
        snopes)
            echo "$PROJECT_ROOT/configs/snopes_config.conf"
            ;;
        politifact)
            echo "$PROJECT_ROOT/configs/politifact_config.conf"
            ;;
        mumin)
            echo "$PROJECT_ROOT/configs/mumin_config.conf"
            ;;
        mumin-multilingual|mumin_multilingual)
            echo "$PROJECT_ROOT/configs/mumin_multilingual_config.conf"
            ;;
        m3check|m3-check)
            echo "$PROJECT_ROOT/configs/m3check_config.conf"
            ;;
        m3check-multilingual|m3check_multilingual)
            echo "$PROJECT_ROOT/configs/m3check_multilingual_config.conf"
            ;;
        m3check-no-eng|m3check_no_eng)
            echo "$PROJECT_ROOT/configs/m3check_no_eng_config.conf"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Parse arguments
DATASET=""
CONFIG_FILE=""
RESUME=false
EPOCHS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d)
            DATASET=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            shift 2
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --resume|-r)
            RESUME=true
            shift
            ;;
        --epochs|-e)
            EPOCHS="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            # Backwards compatibility: primo argomento senza flag = config
            if [ -z "$CONFIG_FILE" ] && [ -f "$1" ]; then
                CONFIG_FILE="$1"
            elif [ -z "$DATASET" ]; then
                DATASET=$(echo "$1" | tr '[:upper:]' '[:lower:]')
            fi
            shift
            ;;
    esac
done

# Se non specificato, usa snopes come default
if [ -z "$DATASET" ] && [ -z "$CONFIG_FILE" ]; then
    DATASET="snopes"
fi

# Se specificato dataset, trova config corrispondente
if [ -z "$CONFIG_FILE" ] && [ -n "$DATASET" ]; then
    CONFIG_FILE=$(get_config_for_dataset "$DATASET")
    if [ -z "$CONFIG_FILE" ]; then
        echo "❌ Dataset sconosciuto: $DATASET"
        echo "   Validi: snopes, politifact, mumin, mumin-multilingual, m3check, m3check-multilingual, m3check-no-eng"
        exit 1
    fi
fi

# Verifica file config esiste
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ File configurazione non trovato: $CONFIG_FILE"
    echo ""
    echo "Configurazioni disponibili:"
    ls -la "$PROJECT_ROOT/configs/"*.conf 2>/dev/null || echo "  Nessuna"
    exit 1
fi

# Estrai nome dataset dal config se non specificato
if [ -z "$DATASET" ]; then
    DATASET=$(grep -E "^type_dataset=" "$CONFIG_FILE" | cut -d'=' -f2 | tr -d "'" | tr '[:upper:]' '[:lower:]')
fi

# Mostra info
echo ""
echo "🏋️ MMBT Training"
echo "================"
echo "Dataset: $DATASET"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: models/checkpoints/mmbt_${DATASET}/"

# Verifica checkpoint esistente
CKPT_DIR="$PROJECT_ROOT/models/checkpoints/mmbt_${DATASET}"
if [ -f "$CKPT_DIR/model_best.pt" ]; then
    echo ""
    echo "⚠️  Checkpoint esistente trovato!"
    if [ "$RESUME" = true ]; then
        echo "   → Riprendo training da checkpoint..."
    else
        echo "   → Il nuovo training sovrascriverà il checkpoint esistente"
        echo "   → Usa --resume per riprendere invece di riaddestrare"
    fi
fi

echo ""

# Prepara comandi extra
EXTRA_ARGS=""
if [ -n "$EPOCHS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max_epochs $EPOCHS"
fi

# Esegui training
cd "$PROJECT_ROOT"
echo "Avvio training..."
echo ""

python main.py -c "$CONFIG_FILE" $EXTRA_ARGS

echo ""
echo "✅ Training completato!"
echo "📦 Checkpoint salvato in: $CKPT_DIR"
