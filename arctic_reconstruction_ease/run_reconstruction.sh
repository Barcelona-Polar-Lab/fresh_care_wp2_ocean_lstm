#!/usr/bin/env bash
# ===========================================================================
# run_reconstruction.sh — Arctic Reconstruction Pipeline Orchestrator
#
# The pipeline is split into five stages so that the heavy LSTM step (D)
# can be moved to a remote GPU server without shipping the 700 GB GLORYS
# archive. Steps A, B, C and E need GLORYS / satellite raw data;
# Step D only needs the small intermediates produced by A, B and C.
#
#   A   — Ocean mask + bathymetry on EASE grid (static, run once).
#   B   — Regrid satellite SST/SSS/ADT to EASE grid (run once / resolution).
#   C   — Regrid GLORYS surface (depth=0) to EASE grid, per date.
#   D   — LSTM + MC-Dropout anomaly inference, per date.
#   E   — Combine D anomalies with full 3-D GLORYS + satellite, per date,
#         producing the final published NetCDF (schema unchanged).
#
# Usage:
#   bash run_reconstruction.sh [--mode=MODE] [--no-preflight | --preflight-keep-all] \
#       <config1.yaml> [config2.yaml ...]
#
# Modes:
#   local            (default) Run A, B, C, D, E end-to-end on this machine.
#   server-prep      Run A, B, C only — produces the small intermediate
#                    bundle to ship to the GPU server.
#   server-anomalies Run D only — needs A/B/C intermediates already in place.
#                    Use this on the remote GPU server.
#   local-finalize   Run E only — needs anomalies + raw GLORYS available.
#
# Options:
#   --mode=MODE              See above (default: local).
#   --no-preflight           Skip interactive checks; run everything fresh.
#   --preflight-keep-all     Skip interactive checks; only fill gaps.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Parse options -------------------------------------------------------
NO_PREFLIGHT=false
KEEP_ALL=false
MODE="local"
CONFIGS=()

for arg in "$@"; do
    case "$arg" in
        --no-preflight) NO_PREFLIGHT=true ;;
        --preflight-keep-all) KEEP_ALL=true ;;
        --mode=*) MODE="${arg#--mode=}" ;;
        --mode) echo "ERROR: --mode requires =VALUE form, e.g. --mode=local" >&2; exit 1 ;;
        -*) echo "ERROR: Unknown option: $arg" >&2; exit 1 ;;
        *) CONFIGS+=("$arg") ;;
    esac
done

case "$MODE" in
    local|server-prep|server-anomalies|local-finalize) ;;
    *)
        echo "ERROR: Unknown --mode='$MODE'. Valid: local | server-prep | server-anomalies | local-finalize" >&2
        exit 1
        ;;
esac

if [[ "$NO_PREFLIGHT" == true && "$KEEP_ALL" == true ]]; then
    echo "ERROR: --no-preflight and --preflight-keep-all are mutually exclusive" >&2
    exit 1
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--mode=MODE] [--no-preflight | --preflight-keep-all] <config1.yaml> ..." >&2
    exit 1
fi

run_step_A=false
run_step_B=false
run_step_C=false
run_step_D=false
run_step_E=false
case "$MODE" in
    local)            run_step_A=true; run_step_B=true; run_step_C=true; run_step_D=true; run_step_E=true ;;
    server-prep)      run_step_A=true; run_step_B=true; run_step_C=true ;;
    server-anomalies) run_step_D=true ;;
    local-finalize)   run_step_E=true ;;
esac

# ---- Process each config -------------------------------------------------
TOTAL=${#CONFIGS[@]}
for (( i=0; i<TOTAL; i++ )); do
    CFG_RAW="${CONFIGS[$i]}"
    N=$(( i + 1 ))

    if [[ ! -f "$CFG_RAW" ]]; then
        echo "ERROR: Config file not found: $CFG_RAW" >&2
        exit 1
    fi

    # Resolve absolute path
    CONFIG="$(cd "$(dirname "$CFG_RAW")" && pwd)/$(basename "$CFG_RAW")"

    echo ""
    echo "============================================================"
    echo " Arctic Reconstruction Pipeline  [$N/$TOTAL]   mode=$MODE"
    echo " Config: $CONFIG"
    echo "============================================================"

    # ---- Preflight -------------------------------------------------------
    if [[ "$NO_PREFLIGHT" == false && "$KEEP_ALL" == false ]]; then
        echo ""
        echo ">>> Preflight: checking existing outputs"
        PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python -c "
from config_utils import load_config, run_preflight
import sys
run_preflight(load_config(sys.argv[1]))
" "$CONFIG"
    elif [[ "$KEEP_ALL" == true ]]; then
        # Keep all existing files, only process what's missing
        PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python -c "
from config_utils import load_config, get_pipeline_plan_path, get_static_data_path
import json, sys
cfg = load_config(sys.argv[1])
plan = {
    'run_step_A': True,
    'run_step_B': {'SST': True, 'SSS': True, 'ADT': True},
    'overwrite_glorys_surface': False,
    'overwrite_anomalies': False,
    'overwrite_reconstruction': False,
}
# Skip Step A entirely if static file already exists
if get_static_data_path(cfg).exists():
    plan['run_step_A'] = False
plan_path = get_pipeline_plan_path(cfg)
plan_path.parent.mkdir(parents=True, exist_ok=True)
with open(plan_path, 'w') as f:
    json.dump(plan, f, indent=2)
print('  Preflight: keep-all mode — reusing existing files, filling gaps')
" "$CONFIG"
    else
        # No preflight — run everything fresh
        PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}" python -c "
from config_utils import load_config, get_pipeline_plan_path
import json, sys
cfg = load_config(sys.argv[1])
plan = {
    'run_step_A': True,
    'run_step_B': {'SST': True, 'SSS': True, 'ADT': True},
    'overwrite_glorys_surface': True,
    'overwrite_anomalies': True,
    'overwrite_reconstruction': True,
}
plan_path = get_pipeline_plan_path(cfg)
plan_path.parent.mkdir(parents=True, exist_ok=True)
with open(plan_path, 'w') as f:
    json.dump(plan, f, indent=2)
print('  Preflight skipped — running fresh (overwrite all)')
" "$CONFIG"
    fi

    # ---- Step A: Static data (mask + bathymetry) -------------------------
    if [[ "$run_step_A" == true ]]; then
        echo ""
        echo ">>> Step A: Ocean mask & bathymetry"
        python "$SCRIPT_DIR/A_create_ocean_mask.py" --config "$CONFIG"
    fi

    # ---- Step B: Satellite regridding ------------------------------------
    if [[ "$run_step_B" == true ]]; then
        echo ""
        echo ">>> Step B: Satellite data → EASE grid"
        python "$SCRIPT_DIR/B_surf_data_to_EASE.py" --config "$CONFIG"
    fi

    # ---- Step C: GLORYS surface regridding ------------------------------
    if [[ "$run_step_C" == true ]]; then
        echo ""
        echo ">>> Step C: GLORYS surface → EASE grid"
        python "$SCRIPT_DIR/C_glorys_surface_to_EASE.py" --config "$CONFIG"
    fi

    # ---- Step D: Anomaly inference ---------------------------------------
    if [[ "$run_step_D" == true ]]; then
        echo ""
        echo ">>> Step D: LSTM anomaly inference"
        python "$SCRIPT_DIR/D_arctic_reconstruction.py" --config "$CONFIG"
    fi

    # ---- Step E: Finalize with full GLORYS -------------------------------
    if [[ "$run_step_E" == true ]]; then
        echo ""
        echo ">>> Step E: Finalize with full GLORYS reference"
        python "$SCRIPT_DIR/E_finalize_with_glorys.py" --config "$CONFIG"
    fi

    echo ""
    echo "============================================================"
    echo " Pipeline finished for config $N/$TOTAL: $(basename "$CONFIG")"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo " All $TOTAL pipelines completed (mode=$MODE)."
echo "============================================================"
