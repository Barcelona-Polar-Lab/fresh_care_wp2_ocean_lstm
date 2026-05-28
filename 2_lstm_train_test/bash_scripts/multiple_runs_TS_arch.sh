#!/bin/bash

# Run from the parent directory (2_lstm_train_test/) so relative paths resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Architecture search script for Temperature + Salinity only (no Steric Height)
# Uses --output_vars 110 to disable steric_height output
# Models saved under trained_models/models_TS_arch_search/

# Activate Python environment
source /home/nicolas/python_venv/bin/activate

# Define layer configurations to test
architectures=(
    # --- Previous runs (skipped if results already exist) ---
    "26 26"
    "30 30"
    "34 34"
    "36 36"
    "38 38"
    "40 40"
    "42 42"
    "44 44"
    "46 46"
    "48 48"
    "50 50"
    "52 52"
    "54 54"
    "52 46"
    "54 48"

    # --- New: explore shrinkage around best result (52 46) ---
    # Vary scale, keep ~6-unit shrinkage
    "48 42"    # scale down by 4, same shrinkage
    "50 44"    # scale down by 2, same shrinkage
    "54 46"    # larger first layer, same second
    "56 50"    # scale up by 4, same shrinkage
    # Vary shrinkage amount, keep first layer at 52
    "52 44"    # 8-unit shrinkage
    "52 42"    # 10-unit shrinkage
    # Vary both
    "54 44"    # 10-unit shrinkage from larger first layer
    "56 46"    # 10-unit shrinkage, larger first layer

    # --- New: 3-layer experiments (small / medium / large) ---
    "32 28 24"    # small: ~72k params range
    "44 40 36"    # medium: extends mid-range with progressive shrinkage
    "52 46 40"    # large: natural extension of current best 2-layer
)

# Parent directory for all T+S architecture search models
PARENT_DIR="trained_models/models_TS_arch_search"

# Counter for tracking progress
total=${#architectures[@]}
current=0

for arch in "${architectures[@]}"; do
    current=$((current + 1))
    
    # Build model directory name (matches auto-generated format)
    units_str=$(echo "$arch" | tr ' ' '_')
    model_dir="${PARENT_DIR}/model_LSTM_${units_str}_sat_TS"
    
    if [ -f "${model_dir}/mc_test_results.nc" ]; then
        echo "[$current/$total] Skipping [$arch] — mc_test_results.nc already exists in $model_dir"
        echo "---"
        continue
    fi

    echo "============================================================"
    echo "[$current/$total] Running architecture: LSTM units = [$arch]"
    echo "  Output variables: temperature, salinity (--output_vars 110)"
    echo "  Model directory: $model_dir"
    echo "============================================================"
    
    python3 lstm_pytorch_pd_mcdo.py \
        --lstm_units $arch \
        --mode both \
        --output_vars 110 \
        --model_dir "$model_dir"
    
    echo "Completed architecture: [$arch]"
    echo "---"
done

echo "All T+S architecture configurations completed!"
