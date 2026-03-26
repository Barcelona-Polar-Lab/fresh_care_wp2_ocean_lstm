#!/bin/bash

# Architecture search script for Temperature + Salinity only (no Steric Height)
# Uses --output_vars 110 to disable steric_height output
# Models saved under trained_models/models_TS_arch_search/

# Activate Python environment
source /home/nicolas/python_venv/bin/activate

# Define layer configurations to test
architectures=(
    "26 26"
    "30 30"
    "34 34"
    "36 36"
    "38 38"
    "40 40"
    "42 42"
    "44 44"
    "48 48"
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
