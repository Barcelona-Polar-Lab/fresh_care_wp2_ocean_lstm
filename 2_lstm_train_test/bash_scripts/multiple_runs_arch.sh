#!/bin/bash

# Run from the parent directory (2_lstm_train_test/) so relative paths resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Architecture search script - testing different LSTM layer configurations
# Hyperparameters remain at default values from Config class

# Define layer configurations to test

# First set of architectures to test
# architectures=(
#     "30 30"
#     "32 32"
#     "35 35"
#     "38 38"
#     "40 40"
# )

# Second set of architectures to test
architectures=(
    "26 26"
    "28 28"
    "42 42"
    "45 45"
    "42 38"
    "45 42"
)




# Counter for tracking progress
total=${#architectures[@]}
current=0

for arch in "${architectures[@]}"; do
    current=$((current + 1))
    
    echo "[$current/$total] Running architecture: LSTM units = [$arch]"
    python3 lstm_pytorch_pd.py --lstm_units $arch --mode both
    echo "Completed architecture: [$arch]"
    echo "---"
done

echo "All architecture configurations completed!"
