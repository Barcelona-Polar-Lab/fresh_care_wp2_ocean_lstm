#!/bin/bash

# Script to compare model performance using satellite vs GLORYS surface data
# Runs the LSTM model with both surface data sources

echo "=========================================="
echo "Surface Data Source Comparison"
echo "=========================================="

# Define surface data sources
surface_sources=("satellite" "glorys")

# Counter for tracking progress
total=${#surface_sources[@]}
current=0

for surface_ts in "${surface_sources[@]}"; do
    current=$((current + 1))
    
    echo ""
    echo "[$current/$total] Running with surface T/S source: $surface_ts"
    echo "------------------------------------------"
    
    # Run training and testing with the specified surface data source
    python3 lstm_pytorch_pd.py --mode both --surface_ts $surface_ts
    
    if [ $? -eq 0 ]; then
        echo "✓ Completed: $surface_ts"
    else
        echo "✗ Failed: $surface_ts"
        exit 1
    fi
    
    echo "------------------------------------------"
done

echo ""
echo "=========================================="
echo "All surface data source runs completed!"
echo "=========================================="
