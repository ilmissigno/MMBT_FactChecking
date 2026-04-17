# =============================================================================
# MMBT Fact-Checking Makefile
# Supporta sia esecuzione diretta (default) che Docker
# =============================================================================

.PHONY: help setup install train eval test lint clean \
        train-snopes train-politifact train-mumin train-mumin-multilingual \
        train-m3check train-m3check-multilingual \
        exp-noise exp-mumin exp-complexity exp-all \
        baseline-man baseline-man-mumin baseline-man-m3check baseline-man-m3check-no-eng \
        docker-build docker-train docker-eval docker-test docker-shell

# Variabili
PYTHON := python3
VENV := .venv
BIN := ./bin
CONFIG := configs/configurazione.conf
DATASET := snopes
OUTPUT_DIR := outputs
DOCKER_COMPOSE := docker-compose -f docker/docker-compose.yml
RESUME ?=
RESUME_FLAG = $(if $(RESUME),--resume,)

# Colori
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m

# Default target
.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================

help:
	@echo ""
	@echo "$(BLUE)🚀 MMBT Fact-Checking - Makefile$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Setup & Installazione:$(NC)"
	@echo "  make setup        - Setup completo ambiente (venv + dipendenze)"
	@echo "  make install      - Installa solo dipendenze Python"
	@echo ""
	@echo "$(GREEN)Training (specifica DATASET=snopes|politifact|mumin|m3check):$(NC)"
	@echo "  make train                - Training dataset default (snopes)"
	@echo "  make train DATASET=politifact"
	@echo "  make train-snopes              - Training Snopes"
	@echo "  make train-politifact          - Training Politifact"
	@echo "  make train-mumin               - Training MuMiN"
	@echo "  make train-mumin-multilingual  - Training MuMiN Multilingua (mBERT)"
	@echo "  make train-m3check             - Training M3-Check"
	@echo "  make train-m3check-multilingual - Training M3-Check Multilingua (mBERT)"
	@echo "  make train-mumin-multilingual RESUME=1  - Riprendi training MuMiN Multilingua"
	@echo ""
	@echo "$(GREEN)Evaluation & Status:$(NC)"
	@echo "  make eval DATASET=snopes  - Valuta modello"
	@echo "  make status               - Mostra stato checkpoint"
	@echo ""
	@echo "$(GREEN)Esperimenti (specifica DATASET):$(NC)"
	@echo "  make exp-noise    - Esperimento rumore bianco"
	@echo "  make exp-mumin    - Esperimento dataset MuMiN"
	@echo "  make exp-complexity - Benchmark complessità FAISS"
	@echo "  make exp-all      - Tutti gli esperimenti"
	@echo ""
	@echo "$(GREEN)Baseline MAN (Vo & Lee, EMNLP 2020):$(NC)"
	@echo "  make baseline-man DATASET=mumin   - MAN su dataset specifico"
	@echo "  make baseline-man-mumin           - MAN su MuMiN (no-EN)"
	@echo "  make baseline-man-m3check         - MAN su M3-Check multilingua"
	@echo "  make baseline-man-m3check-no-eng  - MAN su M3-Check senza inglese"
	@echo ""
	@echo "$(GREEN)Test & Quality:$(NC)"
	@echo "  make test         - Esegui test suite"
	@echo "  make lint         - Controlla stile codice"
	@echo "  make format       - Formatta codice"
	@echo ""
	@echo "$(GREEN)Docker (opzionale):$(NC)"
	@echo "  make docker-build - Build immagine Docker"
	@echo "  make docker-train - Training in container"
	@echo "  make docker-shell - Shell interattiva"
	@echo ""
	@echo "$(GREEN)Utility:$(NC)"
	@echo "  make clean        - Pulisci file temporanei"
	@echo "  make info         - Info sistema (GPU, Python)"
	@echo ""

# =============================================================================
# SETUP & INSTALL (Esecuzione Diretta)
# =============================================================================

setup:
	@echo "$(BLUE)🔧 Setup ambiente...$(NC)"
	@$(BIN)/setup_env.sh

install:
	@echo "$(BLUE)📦 Installazione dipendenze...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		. $(VENV)/bin/activate && pip install -r requirements.txt; \
	else \
		pip install -r requirements.txt; \
	fi

install-dev: install
	@echo "$(BLUE)📦 Installazione dipendenze sviluppo...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		. $(VENV)/bin/activate && pip install pytest pytest-cov black isort flake8 mypy; \
	else \
		pip install pytest pytest-cov black isort flake8 mypy; \
	fi

# =============================================================================
# TRAINING & EVALUATION (Esecuzione Diretta)
# =============================================================================

train:
	@echo "$(BLUE)🏋️ Avvio training $(DATASET)...$(NC)"
	@$(BIN)/train.sh --dataset $(DATASET) $(RESUME_FLAG)

train-snopes:
	@echo "$(BLUE)🏋️ Training Snopes...$(NC)"
	@$(BIN)/train.sh --dataset snopes $(RESUME_FLAG)

train-politifact:
	@echo "$(BLUE)🏋️ Training Politifact...$(NC)"
	@$(BIN)/train.sh --dataset politifact $(RESUME_FLAG)

train-mumin:
	@echo "$(BLUE)🏋️ Training MuMiN...$(NC)"
	@$(BIN)/train.sh --dataset mumin $(RESUME_FLAG)

train-mumin-multilingual:
	@echo "$(BLUE)🏋️ Training MuMiN Multilingual...$(NC)"
	@$(BIN)/train.sh --dataset mumin-multilingual $(RESUME_FLAG)

train-m3check:
	@echo "$(BLUE)🏋️ Training M3-Check...$(NC)"
	@$(BIN)/train.sh --dataset m3check $(RESUME_FLAG)

train-m3check-multilingual:
	@echo "$(BLUE)🏋️ Training M3-Check Multilingual...$(NC)"
	@$(BIN)/train.sh --dataset m3check-multilingual $(RESUME_FLAG)

train-m3check-no-eng:
	@echo "$(BLUE)🏋️ Training M3-Check No English...$(NC)"
	@$(BIN)/train.sh --dataset m3check-no-eng $(RESUME_FLAG)

eval:
	@echo "$(BLUE)📊 Valutazione $(DATASET)...$(NC)"
	@$(BIN)/run_experiment.sh evaluate --dataset $(DATASET)

status:
	@$(BIN)/run_experiment.sh status

# =============================================================================
# ESPERIMENTI (Esecuzione Diretta)
# =============================================================================

exp-noise:
	@echo "$(BLUE)🔊 Esperimento White Noise...$(NC)"
	@$(BIN)/run_experiment.sh noise --dataset $(DATASET) --output $(OUTPUT_DIR)

exp-mumin:
	@echo "$(BLUE)📰 Esperimento MuMiN...$(NC)"
	@$(BIN)/run_experiment.sh mumin --output $(OUTPUT_DIR)

exp-complexity:
	@echo "$(BLUE)⚡ Benchmark Complessità FAISS...$(NC)"
	@$(BIN)/run_experiment.sh complexity --dataset $(DATASET) --output $(OUTPUT_DIR)

exp-all:
	@echo "$(BLUE)🔄 Tutti gli esperimenti...$(NC)"
	@$(BIN)/run_experiment.sh all --output $(OUTPUT_DIR)

# =============================================================================
# BASELINE MAN (Vo & Lee, EMNLP 2020)
# =============================================================================

MAN_EPOCHS ?= 3
MAN_BATCH ?= 16
MAN_LR ?= 1e-4
MAN_BERT ?= /root/bert_multilingual_local
MAN_MAX_TRAIN ?= 0

baseline-man:
	@echo "$(BLUE)🧪 MAN Baseline — $(DATASET)$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) $(PYTHON) -m scripts.benchmark_baselines_man \
		--dataset $(DATASET) \
		--bert-path $(MAN_BERT) \
		--epochs $(MAN_EPOCHS) \
		--batch-size $(MAN_BATCH) \
		--lr $(MAN_LR) \
		--max-train-samples $(MAN_MAX_TRAIN)

baseline-man-mumin:
	@echo "$(BLUE)🧪 MAN Baseline — MuMiN (multilingua, no-EN)$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) $(PYTHON) -m scripts.benchmark_baselines_man \
		--dataset mumin \
		--bert-path $(MAN_BERT) \
		--epochs $(MAN_EPOCHS) \
		--batch-size $(MAN_BATCH) \
		--lr $(MAN_LR)

baseline-man-m3check:
	@echo "$(BLUE)🧪 MAN Baseline — M3-Check Multilingua$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) $(PYTHON) -m scripts.benchmark_baselines_man \
		--dataset m3check_multilingual \
		--bert-path $(MAN_BERT) \
		--epochs $(MAN_EPOCHS) \
		--batch-size $(MAN_BATCH) \
		--lr $(MAN_LR)

baseline-man-m3check-no-eng:
	@echo "$(BLUE)🧪 MAN Baseline — M3-Check senza inglese$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) $(PYTHON) -m scripts.benchmark_baselines_man \
		--dataset m3check_no_eng \
		--bert-path $(MAN_BERT) \
		--epochs $(MAN_EPOCHS) \
		--batch-size $(MAN_BATCH) \
		--lr $(MAN_LR)

# =============================================================================
# TEST & QUALITY
# =============================================================================

test:
	@echo "$(BLUE)🧪 Esecuzione test...$(NC)"
	@$(BIN)/test.sh tests/ -v

test-coverage:
	@echo "$(BLUE)📊 Test con coverage...$(NC)"
	@. $(BIN)/activate_env.sh && pytest tests/ -v --cov=mmbt --cov-report=html

lint:
	@echo "$(BLUE)🔍 Controllo stile codice...$(NC)"
	@. $(BIN)/activate_env.sh && flake8 mmbt/ scripts/ tests/ --max-line-length=120 --ignore=E501,W503

format:
	@echo "$(BLUE)✨ Formattazione codice...$(NC)"
	@. $(BIN)/activate_env.sh && black mmbt/ scripts/ tests/ && isort mmbt/ scripts/ tests/

typecheck:
	@echo "$(BLUE)🔎 Type checking...$(NC)"
	@. $(BIN)/activate_env.sh && mypy mmbt/ --ignore-missing-imports

# =============================================================================
# DOCKER (Opzionale)
# =============================================================================

docker-build:
	@echo "$(BLUE)🐳 Build Docker image...$(NC)"
	@$(DOCKER_COMPOSE) build

docker-train:
	@echo "$(BLUE)🐳 Training in Docker...$(NC)"
	@$(DOCKER_COMPOSE) run --rm train

docker-eval:
	@echo "$(BLUE)🐳 Evaluation in Docker...$(NC)"
	@$(DOCKER_COMPOSE) run --rm evaluate

docker-test:
	@echo "$(BLUE)🐳 Test in Docker...$(NC)"
	@$(DOCKER_COMPOSE) run --rm test

docker-shell:
	@echo "$(BLUE)🐳 Shell interattiva Docker...$(NC)"
	@$(DOCKER_COMPOSE) run --rm dev bash

docker-exp-noise:
	@echo "$(BLUE)🐳 Esperimento Noise in Docker...$(NC)"
	@$(DOCKER_COMPOSE) run --rm experiment-noise

docker-exp-complexity:
	@echo "$(BLUE)🐳 Esperimento Complexity in Docker...$(NC)"
	@$(DOCKER_COMPOSE) run --rm experiment-complexity

docker-down:
	@$(DOCKER_COMPOSE) down

# =============================================================================
# UTILITY
# =============================================================================

clean:
	@echo "$(BLUE)🧹 Pulizia file temporanei...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@rm -rf htmlcov/ .mypy_cache/ 2>/dev/null || true
	@echo "$(GREEN)✅ Pulizia completata$(NC)"

clean-all: clean
	@echo "$(BLUE)🧹 Pulizia completa (incluso venv e outputs)...$(NC)"
	@rm -rf $(VENV) $(OUTPUT_DIR)/* logs/* 2>/dev/null || true

tree:
	@echo "$(BLUE)📁 Struttura progetto:$(NC)"
	@tree -L 2 -I '__pycache__|*.pyc|.git|.venv|node_modules' || find . -maxdepth 2 -type d | head -30

info:
	@echo ""
	@echo "$(BLUE)ℹ️  Informazioni Sistema$(NC)"
	@echo "========================"
	@echo "Python: $$(python3 --version 2>/dev/null || echo 'Non trovato')"
	@echo "CUDA: $$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'Non disponibile')"
	@echo "GPU: $$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Non disponibile')"
	@echo "Docker: $$(docker --version 2>/dev/null || echo 'Non installato')"
	@echo ""

# =============================================================================
# DATASET PREPARATION
# =============================================================================

prepare-mumin:
	@echo "$(BLUE)📥 Preparazione dataset MuMiN...$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) python scripts/prepare_mumin_dataset_v2.py --neg-ratio 30

prepare-mumin-multilingual:
	@echo "$(BLUE)📥 Preparazione dataset MuMiN (solo non-EN)...$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) python scripts/prepare_mumin_dataset_v2.py \
		--neg-ratio 30 \
		--exclude-langs en \
		--output-dir datasets/mumin_mmbt

train-mumin-multilingual:
	@echo "$(BLUE)🏋️ Training MuMiN Multilingua (3 epoche, bert-multilingual)...$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) python main.py -c configs/mumin_multilingual_config.conf

train-mumin-multilingual:
	@echo "$(BLUE)🏋️ Training MuMiN Multilingua (3 epoche)...$(NC)"
	@. $(VENV)/bin/activate && PYTHONPATH=$(CURDIR) python main.py -c configs/mumin_multilingual_config.conf

download-models:
	@echo "$(BLUE)📥 Download modelli pre-trained...$(NC)"
	@. $(VENV)/bin/activate && python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
	@. $(BIN)/activate_env.sh && python -c "from torchvision.models import resnet152; resnet152(pretrained=True)"
