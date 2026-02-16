#!/bin/bash

# Script to extract and display RMSE values from all LSTM model directories

echo "================================================================="
echo "LSTM Model Results Summary - RMSE Values"
echo "================================================================="
echo ""

# Find all model directories matching the pattern and sort them
for dir in $(find models_arch_search -maxdepth 1 -type d -name "model_LSTM_*_*" | sort); do
    # Get directory name
    dir_name=$(basename "$dir")
    
    # Check if test_results.nc exists
    results_file="${dir}/test_results.nc"
    
    if [ -f "$results_file" ]; then
        # Extract RMSE values using ncdump
        t_rmse=$(ncdump -h "$results_file" 2>/dev/null | grep ":T_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        s_rmse=$(ncdump -h "$results_file" 2>/dev/null | grep ":S_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        sh_rmse=$(ncdump -h "$results_file" 2>/dev/null | grep ":SH_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        rmse_sum=$(ncdump -h "$results_file" 2>/dev/null | grep ":RMSEs_sum" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        
        if [ -n "$t_rmse" ] && [ -n "$s_rmse" ] && [ -n "$sh_rmse" ] && [ -n "$rmse_sum" ]; then
            # Format values to 4 decimal places
            t_rmse_fmt=$(printf "%.4f" "$t_rmse")
            s_rmse_fmt=$(printf "%.4f" "$s_rmse")
            sh_rmse_fmt=$(printf "%.4f" "$sh_rmse")
            rmse_sum_fmt=$(printf "%.4f" "$rmse_sum")
            
            printf "%-20s  T: %-7s   S: %-7s   SH: %-7s   Sum: %-7s\n" "$dir_name" "$t_rmse_fmt" "$s_rmse_fmt" "$sh_rmse_fmt" "$rmse_sum_fmt"
        else
            printf "%-20s  Could not extract RMSE values\n" "$dir_name"
        fi
    else
        printf "%-20s  test_results.nc not found\n" "$dir_name"
    fi
done

echo ""
echo "================================================================="