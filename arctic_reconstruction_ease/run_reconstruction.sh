#!/usr/bin/env bash
# ===========================================================================
# run_reconstruction.sh — Arctic Reconstruction Pipeline Orchestrator
#
# Usage:
#   bash run_reconstruction.sh [--no-preflight] <config1.yaml> [config2.yaml ...]
#
# Examples:
#   # Pan-Arctic 25 km run (interactive preflight):
#   bash run_reconstruction.sh configs/config_arctic_25km.yaml
#
#   # Pan-Arctic 6.25 km run (heavy — see config header):
#   bash run_reconstruction.sh configs/config_arctic_6p25km.yaml
#
#   # Run a single region interactively (will ask what to reuse/redo):
#   bash run_reconstruction.sh configs/config_bering.yaml
#
#   # Run all four regions unattended, keeping existing files (fill gaps only):
#   bash run_reconstruction.sh --preflight-keep-all configs/config_bering.yaml configs/config_davis.yaml configs/config_fram.yaml configs/config_barents.yaml
#
#   # Run all four regions from scratch (overwrites everything):
#   bash run_reconstruction.sh --no-preflight configs/config_bering.yaml configs/config_davis.yaml configs/config_fram.yaml configs/config_barents.yaml
#
# Steps executed (per config):
#   A  — Create ocean mask + bathymetry on EASE grid  (static, run once)
#   B  — Regrid satellite surface data to EASE grid   (run once per resolution)
#   C  — Run LSTM reconstruction for every target date
#
# Options:
#   --no-preflight       Skip interactive checks; run everything fresh (overwrites existing).
#   --preflight-keep-all Skip interactive checks; keep all existing files, only fill gaps.
#
# D_build_model_input.py is deprecated; its logic is absorbed into C.
# GLORYS regridding is called per-timestep from inside C.
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Parse options -------------------------------------------------------
NO_PREFLIGHT=false
KEEP_ALL=false
CONFIGS=()

for arg in "$@"; do
    case "$arg" in
        --no-preflight) NO_PREFLIGHT=true ;;
        --preflight-keep-all) KEEP_ALL=true ;;
        -*) echo "ERROR: Unknown option: $arg" >&2; exit 1 ;;
        *) CONFIGS+=("$arg") ;;
    esac
done

if [[ "$NO_PREFLIGHT" == true && "$KEEP_ALL" == true ]]; then
    echo "ERROR: --no-preflight and --preflight-keep-all are mutually exclusive" >&2
    exit 1
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "Usage: $0 [--no-preflight] <config1.yaml> [config2.yaml ...]" >&2
    exit 1
fi

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
    echo " Arctic Reconstruction Pipeline  [$N/$TOTAL]"
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
    echo ""
    echo ">>> Step A: Ocean mask & bathymetry"
    python "$SCRIPT_DIR/A_create_ocean_mask.py" --config "$CONFIG"

    # ---- Step B: Satellite regridding ------------------------------------
    echo ""
    echo ">>> Step B: Satellite data → EASE grid"
    python "$SCRIPT_DIR/B_surf_data_to_EASE.py" --config "$CONFIG"

    # ---- Step C: Reconstruction ------------------------------------------
    echo ""
    echo ">>> Step C: LSTM reconstruction"
    python "$SCRIPT_DIR/C_arctic_reconstruction.py" --config "$CONFIG"

    echo ""
    echo "============================================================"
    echo " Pipeline finished for config $N/$TOTAL: $(basename "$CONFIG")"
    echo "============================================================"
done

echo ""
echo "============================================================"
echo " All $TOTAL reconstructions completed."
echo "============================================================"
