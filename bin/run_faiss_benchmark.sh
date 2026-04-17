#!/bin/bash
# Script per eseguire il benchmark FAISS corretto con warmup e ripetizioni

set -euo pipefail

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}   FAISS Scalability Benchmark${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo -e "${YELLOW}Improvements:${NC}"
echo "  ✓ Warmup runs to stabilize cache"
echo "  ✓ 5 repetitions with median aggregation"
echo "  ✓ Randomized test order"
echo "  ✓ Standard deviation tracking"
echo ""

# Attiva ambiente virtuale se non già attivo
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source /root/MMBT_TESI/.venv/bin/activate
fi

# Directory del progetto
cd /root/MMBT_TESI

# Esegui il benchmark
echo -e "${GREEN}Starting benchmark (this may take 5-10 minutes)...${NC}"
echo ""

python -m scripts.run_experiments \
    --experiment complexity \
    --dataset snopes \
    --output-dir outputs

echo ""
echo -e "${GREEN}✓ Benchmark completed!${NC}"
echo ""
echo "Results saved to:"
echo "  - outputs/snopes/complexity/scalability.json"
echo "  - outputs/snopes/complexity/REPORT.md"
echo ""
echo -e "${BLUE}View report:${NC}"
echo "  cat outputs/snopes/complexity/REPORT.md"
