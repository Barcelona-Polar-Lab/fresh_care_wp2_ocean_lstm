#!/bin/bash

# Run from the parent directory (2_lstm_train_test/) so relative paths resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Script to extract and display RMSE values from all LSTM model directories

# Set numeric locale to C to ensure decimal point formatting
export LC_NUMERIC=C

# Base directory containing the trained_models tree
# Local: sibling of this script's parent (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
# Remote (bec112 server)
#BASE_DIR="/data/FRESH-CARE/data_for_LSTM/models"

# Pattern families to search (glob patterns relative to BASE_DIR)
# Local layout: trained_models/wg_daily_strat_real_only_mcdo_val*/*
# Remote layout: wg_daily_strat_real_only_mcdo_val*/*  (BASE_DIR already ends at models/)
PATTERN_FAMILIES=(
    "trained_models/wg_daily_strat_real*/*"
)

echo "================================================================="
echo "LSTM Model Results Summary - RMSE Values"
echo "================================================================="
echo ""

# Collect all matching directories across all pattern families
dirs=()
for pattern in "${PATTERN_FAMILIES[@]}"; do
    for d in ${BASE_DIR}/${pattern}; do
        [ -d "$d" ] && dirs+=("$d")
    done
done

# Collect results into sortable lines: "rmse_sum|display_line"
result_lines=()
no_results_lines=()

for dir in $(printf '%s\n' "${dirs[@]}" | sort); do
    dir_name=$(basename "$dir")
    results_file="${dir}/mc_test_results.nc"
    
    if [ -f "$results_file" ]; then
        t_rmse=$(ncdump -h "$results_file" 2>/dev/null | grep ":T_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        s_rmse=$(ncdump -h "$results_file" 2>/dev/null | grep ":S_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        sh_rmse=$(ncdump -h "$results_file" 2>/dev/null | grep ":SH_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        rmse_sum=$(ncdump -h "$results_file" 2>/dev/null | grep ":RMSEs_sum" | awk -F' = ' '{print $2}' | sed 's/ ;//')
        
        if [ -n "$t_rmse" ] && [ -n "$s_rmse" ] && [ -n "$rmse_sum" ]; then
            t_rmse_fmt=$(printf "%.4f" "$t_rmse")
            s_rmse_fmt=$(printf "%.4f" "$s_rmse")
            sh_rmse_fmt=$([ -n "$sh_rmse" ] && printf "%.4f" "$sh_rmse" || echo "N/A    ")
            rmse_sum_fmt=$(printf "%.4f" "$rmse_sum")
            
            train_time=$(ncdump -h "$results_file" 2>/dev/null | grep ":training_time_seconds" | awk -F' = ' '{print $2}' | sed 's/ ;//')
            if [ -n "$train_time" ]; then
                train_hrs=$(printf "%.2f" "$(echo "$train_time / 3600" | bc -l)")
                train_time_str="${train_hrs}h"
            else
                train_time_str="N/A"
            fi
            
            display_line=$(printf "%-30s  T: %-7s   S: %-7s   SH: %-7s   Sum: %-7s   Train: %-8s" "$dir_name" "$t_rmse_fmt" "$s_rmse_fmt" "$sh_rmse_fmt" "$rmse_sum_fmt" "$train_time_str")
            result_lines+=("${rmse_sum}|${display_line}")
        else
            no_results_lines+=("$(printf "%-30s  Could not extract RMSE values" "$dir_name")")
        fi
    else
        no_results_lines+=("$(printf "%-30s  mc_test_results.nc not found" "$dir_name")")
    fi
done

# Print results sorted by rmse_sum ascending (best = smallest on top)
printf '%s\n' "${result_lines[@]}" | sort -t'|' -k1 -n | cut -d'|' -f2-

# Print entries without valid results at the bottom
for line in "${no_results_lines[@]}"; do
    echo "$line"
done

echo ""
echo "================================================================="