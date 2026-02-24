#!/usr/bin/env bash
# ============================================================================
# run_kaggle.sh — End-to-end pipeline for Kaggle notebooks / VMs
#
# Generates the advection dataset, installs deps, then trains + evaluates
# each model sequentially.  Designed for a single-GPU Kaggle environment
# (P100 / T4 / etc.).
#
# Usage:
#   bash scripts/run_kaggle.sh              # run all 4 models
#   bash scripts/run_kaggle.sh csdi pidm    # run only specified models
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── 0. Install dependencies ─────────────────────────────────────────────────
echo "==> Installing dependencies..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || true
pip install -q numpy scipy matplotlib pandas tqdm pyyaml scikit-learn einops \
    pytorch-lightning linear_attention_transformer ema-pytorch

# ── 1. Generate advection dataset ───────────────────────────────────────────
DATA_PATH="datasets/synthetic/advection_data_2d.npy"
if [ -f "$DATA_PATH" ]; then
    echo "==> Advection data already exists at $DATA_PATH, skipping generation."
else
    echo "==> Generating advection dataset..."
    python datasets/synthetic/advection_dataset_generator.py
    echo "==> Saved to $DATA_PATH ($(du -h "$DATA_PATH" | cut -f1))"
fi

# ── 2. Determine which models to run ────────────────────────────────────────
ALL_MODELS=(csdi cfmi pidm tmdm)
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${ALL_MODELS[@]}")
fi
echo "==> Models to run: ${MODELS[*]}"

# ── 3. Train + evaluate each model ─────────────────────────────────────────
DEVICE="auto"

for MODEL in "${MODELS[@]}"; do
    CONFIG="configs/experiment/${MODEL}_advection.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "!! Config not found: $CONFIG — skipping $MODEL"
        continue
    fi

    echo ""
    echo "================================================================"
    echo "  MODEL: $MODEL"
    echo "  CONFIG: $CONFIG"
    echo "================================================================"

    python scripts/run_experiment.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --device "$DEVICE"

    echo "==> $MODEL complete."
done

# ── 4. Summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  All done. Results:"
echo "================================================================"
for MODEL in "${MODELS[@]}"; do
    METRICS="results/metrics/${MODEL}_advection/metrics.json"
    DIAG="results/metrics/${MODEL}_advection/diagnostics.json"
    if [ -f "$METRICS" ]; then
        echo ""
        echo "--- $MODEL metrics ---"
        cat "$METRICS"
    fi
    if [ -f "$DIAG" ]; then
        echo ""
        echo "--- $MODEL diagnostics ---"
        cat "$DIAG"
    fi
done
echo ""
