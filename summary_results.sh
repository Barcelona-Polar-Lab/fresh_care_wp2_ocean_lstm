#!/bin/bash

# Script to extract and display RMSEs_sum from all LSTM model directories

echo "==============================================="
echo "LSTM Model Results Summary - RMSEs_sum"
echo "==============================================="
echo ""

# Find all model directories matching the pattern and sort them
for dir in $(find models_arch_search_synth -maxdepth 1 -type d -name "model_LSTM_*_*" | sort); do
    # Get directory name
    dir_name="$dir"
    
    # Check if test_results.nc exists
    results_file="${dir}/test_results.nc"
    
    if [ -f "$results_file" ]; then
        # Extract RMSEs_sum using ncdump
        rmse_sum=$(ncdump -h "$results_file" 2>/dev/null | grep ":RMSEs_sum" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        
        if [ -n "$rmse_sum" ]; then
            printf "%-20s  RMSEs_sum = %s\n" "$dir_name:" "$rmse_sum"
        else
            printf "%-20s  Could not extract RMSEs_sum\n" "$dir_name:"
        fi
    else
        printf "%-20s  test_results.nc not found\n" "$dir_name:"
    fi
done

echo ""
echo "==============================================="
