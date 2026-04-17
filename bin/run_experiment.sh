#!/bin/bash
# =============================================================================
# Run Experiments Script
# Esegue esperimenti e gestisce automaticamente i checkpoint per dataset
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Attiva ambiente
source "$SCRIPT_DIR/activate_env.sh"

# Mapping dataset -> checkpoint directory
get_checkpoint_dir() {
    local dataset_lower=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    echo "$PROJECT_ROOT/models/checkpoints/mmbt_${dataset_lower}"
}

# Verifica checkpoint esiste
check_checkpoint() {
    local dataset="$1"
    local ckpt_dir=$(get_checkpoint_dir "$dataset")
    local model_file="$ckpt_dir/model_best.pt"
    
    if [ -f "$model_file" ]; then
        echo "✅ Checkpoint trovato: $ckpt_dir"
        return 0
    else
        echo "❌ Checkpoint non trovato per '$dataset'"
        echo ""
        echo "   Esegui prima il training:"
        echo "   ./bin/train.sh configs/${dataset}_config.conf"
        echo ""
        return 1
    fi
}

# Mostra stato checkpoint
show_status() {
    echo ""
    echo "📦 Stato Checkpoint MMBT"
    echo "========================"
    
    for ds in snopes politifact mumin m3check; do
        local ckpt_dir=$(get_checkpoint_dir "$ds")
        local model_file="$ckpt_dir/model_best.pt"
        
        if [ -f "$model_file" ]; then
            local size=$(du -h "$model_file" 2>/dev/null | cut -f1)
            local date=$(stat -c %y "$model_file" 2>/dev/null | cut -d' ' -f1)
            echo "✅ ${ds^^}: $ckpt_dir"
            echo "   Size: $size | Modified: $date"
        else
            echo "❌ ${ds^^}: Non addestrato"
        fi
        echo ""
    done
}

# Mostra help
show_help() {
    echo ""
    echo "🧪 MMBT Experiments Runner"
    echo "=========================="
    echo ""
    echo "Uso: $0 <comando> [opzioni]"
    echo ""
    echo "Comandi:"
    echo "  noise       - Esperimento rumore bianco"
    echo "  mumin       - Esperimento dataset MuMiN"
    echo "  complexity  - Benchmark complessità FAISS"
    echo "  evaluate    - Valuta modello esistente"
    echo "  all         - Tutti gli esperimenti"
    echo "  status      - Mostra stato checkpoint"
    echo ""
    echo "Opzioni:"
    echo "  --dataset NAME   - Dataset (snopes/politifact/mumin/m3check)"
    echo "  --noise-levels   - Livelli rumore (es. '0.0,0.1,0.2,0.3')"
    echo "  --output DIR     - Directory output (default: outputs/)"
    echo "  --config FILE    - File configurazione esperimenti"
    echo "  --skip-check     - Salta verifica checkpoint"
    echo "  --help           - Mostra questo messaggio"
    echo ""
    echo "Esempi:"
    echo "  $0 status                           # Mostra checkpoint"
    echo "  $0 evaluate --dataset snopes        # Valuta modello Snopes"
    echo "  $0 noise --dataset snopes           # Esperimento noise"
    echo "  $0 complexity --dataset politifact  # Benchmark FAISS"
    echo ""
}

# Parse arguments
COMMAND=""
DATASET="snopes"
NOISE_LEVELS="0.0,0.1,0.2,0.3"
OUTPUT_DIR="$PROJECT_ROOT/outputs"
CONFIG_FILE="$PROJECT_ROOT/configs/experiments.yaml"
SKIP_CHECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        noise|mumin|complexity|evaluate|all|status)
            COMMAND="$1"
            shift
            ;;
        --dataset)
            DATASET=$(echo "$2" | tr '[:upper:]' '[:lower:]')
            shift 2
            ;;
        --noise-levels)
            NOISE_LEVELS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "❌ Opzione sconosciuta: $1"
            show_help
            exit 1
            ;;
    esac
done

# Se nessun comando, mostra help
if [ -z "$COMMAND" ]; then
    show_help
    exit 1
fi

# Caso speciale: status non richiede controllo
if [ "$COMMAND" = "status" ]; then
    show_status
    exit 0
fi

# Crea directory output
mkdir -p "$OUTPUT_DIR"

echo ""
echo "🧪 MMBT Experiment Runner"
echo "========================="
echo "Comando: $COMMAND"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo ""

cd "$PROJECT_ROOT"

# Verifica checkpoint (se necessario)
if [ "$SKIP_CHECK" = false ]; then
    case $COMMAND in
        noise|evaluate)
            if ! check_checkpoint "$DATASET"; then
                exit 1
            fi
            ;;
        mumin)
            # MuMiN può richiedere training o usare checkpoint esistente
            echo "ℹ️  MuMiN: verifica credenziali Twitter se necessario..."
            ;;
        complexity)
            # Complexity non richiede necessariamente un checkpoint
            echo "ℹ️  Benchmark FAISS: non richiede checkpoint preaddestrato"
            ;;
    esac
fi

echo ""

# Esegui comando
case $COMMAND in
    noise)
        echo "🔊 Esperimento White Noise"
        echo "Dataset: $DATASET"
        echo "Livelli: $NOISE_LEVELS"
        echo ""
        python scripts/run_experiments.py \
            --experiment white_noise \
            --dataset "$DATASET" \
            --noise-levels "$NOISE_LEVELS" \
            --output-dir "$OUTPUT_DIR"
        ;;
        
    evaluate)
        echo "📊 Valutazione Modello"
        echo "Dataset: $DATASET"
        echo ""
        python scripts/run_experiments.py \
            --experiment evaluate \
            --dataset "$DATASET" \
            --output-dir "$OUTPUT_DIR"
        ;;
        
    mumin)
        echo "📰 Esperimento MuMiN Dataset"
        echo ""
        python scripts/run_experiments.py \
            --experiment mumin \
            --dataset mumin \
            --output-dir "$OUTPUT_DIR"
        ;;
        
    complexity)
        echo "⚡ Benchmark Complessità FAISS"
        echo "Dataset: $DATASET"
        echo ""
        python scripts/run_experiments.py \
            --experiment complexity \
            --dataset "$DATASET" \
            --output-dir "$OUTPUT_DIR"
        ;;
        
    all)
        echo "🔄 Tutti gli Esperimenti"
        echo ""
        
        # Verifica almeno un checkpoint esiste
        FOUND=false
        for ds in snopes politifact; do
            if [ -f "$(get_checkpoint_dir $ds)/model_best.pt" ]; then
                FOUND=true
                break
            fi
        done
        
        if [ "$FOUND" = false ]; then
            echo "⚠️  Nessun checkpoint trovato!"
            echo "   Addestra almeno un modello prima di eseguire tutti gli esperimenti."
            exit 1
        fi
        
        python scripts/run_all_experiments.py \
            --config "$CONFIG_FILE" \
            --output-dir "$OUTPUT_DIR"
        ;;
esac

echo ""
echo "✅ Completato!"
echo "📊 Risultati: $OUTPUT_DIR"
