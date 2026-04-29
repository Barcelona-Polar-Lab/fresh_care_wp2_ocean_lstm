#!/bin/bash

# Hyperparameter search for T+S LSTM (architecture: 52 46)
# Updated for:
#   - GLORYS-daily baseline
#   - Real-data-only training
#   - MC-Dropout dev early stopping (PATIENCE_EVALS × MC_DEV_EVERY epochs)
#
# Fixed:
#   - LSTM units = [52, 46]  (previously optimal; not in scope here)
#   - batch_size = 16        (previously optimal)
#   - dropout    = 0.2       (held constant for simplicity)
#   - mc_dev_every = 5       (default; actual patience in epochs = PATE * 5)
#
# Swept: Learning rate × Patience_evals (3 × 3 = 9 runs)
#
# All models saved under trained_models/wg_daily_strat_real_only/
# Naming: model_LSTM_{units}_bs{bs}_lr{lr}_pate{pate}_do{do}
#
# The script is idempotent: re-running skips completed experiments.

# Activate Python environment
source /home/nicolas/python_venv/bin/activate
export LC_NUMERIC=C

# ============================================================
# CONFIGURATION
# ============================================================
LSTM_UNITS="52 46"
UNITS_STR="52_46"
#PARENT_DIR="trained_models/wg_daily_strat_real_only_mcdo_val"  # local
PARENT_DIR="/data/FRESH-CARE/data_for_LSTM/models/wg_daily_strat_real_only_mcdo_val"  # bec112 server

FIXED_BS=16
FIXED_DROPOUT="0.2"
FIXED_MC_DEV_EVERY=5  # matches Config.MC_DEV_EVERY default; used only for dir naming

# Swept axes
LEARNING_RATES=(2e-4 1e-4 5e-5)
PATEVAL_VALUES=(6 10)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

build_model_dir() {
    local lr=$1 pate=$2
    echo "${PARENT_DIR}/model_LSTM_${UNITS_STR}_bs${FIXED_BS}_lr${lr}_pat${pate}x${FIXED_MC_DEV_EVERY}_do${FIXED_DROPOUT}"
}

run_experiment() {
    local lr=$1 pate=$2 current=$3 total=$4
    local model_dir
    model_dir=$(build_model_dir "$lr" "$pate")

    if [ -f "${model_dir}/mc_test_results.nc" ]; then
        echo "[$current/$total] Skipping — results exist: $(basename "$model_dir")"
        return
    fi

    echo "============================================================"
    echo "[$current/$total] LR=$lr PATE=$pate (DO=$FIXED_DROPOUT BS=$FIXED_BS)"
    echo "  Model directory: $model_dir"
    echo "============================================================"

    python3 lstm_pytorch_pd_mcdo.py \
        --lstm_units $LSTM_UNITS \
        --batch_size "$FIXED_BS" \
        --learning_rate "$lr" \
        --patience_evals "$pate" \
        --dropout_rate "$FIXED_DROPOUT" \
        --mode both \
        --model_dir "$model_dir"

    echo "Completed: $(basename "$model_dir")"
    echo "---"
}

# Get RMSEs_sum from a results file
get_rmse_sum() {
    local results_file=$1
    ncdump -h "$results_file" 2>/dev/null \
        | grep ":RMSEs_sum" \
        | awk -F' = ' '{print $2}' \
        | sed 's/ ;//'
}

# Find top model directory by lowest RMSEs_sum
find_best() {
    local dirs=("$@")
    local lines=()
    for dir in "${dirs[@]}"; do
        if [ -f "${dir}/mc_test_results.nc" ]; then
            local rmse
            rmse=$(get_rmse_sum "${dir}/mc_test_results.nc")
            if [ -n "$rmse" ]; then
                lines+=("${rmse}|${dir}")
            fi
        fi
    done
    printf '%s\n' "${lines[@]}" | sort -t'|' -k1 -n | head -n 1 | cut -d'|' -f2
}

# Print a sorted summary table for a set of model directories
print_summary() {
    local stage_name=$1
    shift
    local dirs=("$@")

    echo ""
    echo "--- ${stage_name} (sorted by RMSEs_sum) ---"

    local lines=()
    for dir in "${dirs[@]}"; do
        if [ -f "${dir}/mc_test_results.nc" ]; then
            local rmse t_rmse s_rmse
            rmse=$(get_rmse_sum "${dir}/mc_test_results.nc")
            t_rmse=$(ncdump -h "${dir}/mc_test_results.nc" 2>/dev/null | grep ":T_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
            s_rmse=$(ncdump -h "${dir}/mc_test_results.nc" 2>/dev/null | grep ":S_rmse_total" | awk -F' = ' '{print $2}' | sed 's/ ;//')
            if [ -n "$rmse" ] && [ -n "$t_rmse" ] && [ -n "$s_rmse" ]; then
                local display
                display=$(printf "  %-65s  T: %.4f  S: %.4f  Sum: %.4f" "$(basename "$dir")" "$t_rmse" "$s_rmse" "$rmse")
                lines+=("${rmse}|${display}")
            fi
        fi
    done

    if [ ${#lines[@]} -eq 0 ]; then
        echo "  (no results found)"
    else
        printf '%s\n' "${lines[@]}" | sort -t'|' -k1 -n | cut -d'|' -f2
    fi
    echo ""
}


# ============================================================
# GRID SEARCH: Learning rate × Patience_evals
# Fixed: bs=16, dropout=0.2
# ============================================================
echo ""
echo "================================================================="
echo "GRID SEARCH: Learning rate × Patience_evals"
echo "  Grid: LR={${LEARNING_RATES[*]}} × PATE={${PATEVAL_VALUES[*]}}"
echo "  Fixed: LSTM=[${LSTM_UNITS}], BS=${FIXED_BS}, DO=${FIXED_DROPOUT}"
echo "  Runs: $((${#LEARNING_RATES[@]} * ${#PATEVAL_VALUES[@]}))"
echo "================================================================="

all_dirs=()
total=$((${#LEARNING_RATES[@]} * ${#PATEVAL_VALUES[@]}))
current=0

for lr in "${LEARNING_RATES[@]}"; do
    for pate in "${PATEVAL_VALUES[@]}"; do
        current=$((current + 1))
        run_experiment "$lr" "$pate" "$current" "$total"
        all_dirs+=("$(build_model_dir "$lr" "$pate")")
    done
done


# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "================================================================="
echo "FINAL SUMMARY"
echo "================================================================="

print_summary "All runs" "${all_dirs[@]}"

overall_best=$(find_best "${all_dirs[@]}")
if [ -n "$overall_best" ]; then
    overall_rmse=$(get_rmse_sum "${overall_best}/mc_test_results.nc")
    echo "================================================================="
    echo "OVERALL BEST MODEL:"
    echo "  $(basename "$overall_best")"
    echo "  RMSEs_sum: ${overall_rmse}"
    echo "================================================================="
fi

echo ""
echo "Hyperparameter search completed!"

