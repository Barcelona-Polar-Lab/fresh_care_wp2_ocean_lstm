"""
Quick diagnostic: where are the 4 satellite/GLORYS-surface inputs missing
on the ocean mask, and how do they combine into the LSTM-input NaN map?

For one chosen date, plots 6 panels:
    1. T_glorys_surf  (NaN over ocean)
    2. S_glorys_surf
    3. SST
    4. SSS
    5. ADT
    6. UNION (any of the 5 above is NaN)  vs  ocean mask

Output: plots/missing_inputs_<DATE>.png
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Make pipeline modules importable
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "arctic_reconstruction_ease"))

from config_utils import (  # type: ignore[import-not-found]
    load_config, load_satellite_for_time,
    get_glorys_surface_file, get_static_data_path,
    get_satellite_ease_dirs, SAT_VARS,
)

# --- user-chosen date and config ---
TARGET   = datetime(2011, 1, 3)
CFG_PATH = HERE / "arctic_reconstruction_ease" / "configs" / "config_arctic_25km.yaml"
PLOT_DIR = HERE / "plots"

# ---------------------------------------------------------------------------

cfg = load_config(str(CFG_PATH))

# ocean mask
sd    = xr.open_dataset(get_static_data_path(cfg))
ocean = (sd["ocean_mask"].values == 1)
n_oc  = int(ocean.sum())

# 5 inputs as fed to grid_to_profiles
gs    = xr.open_dataset(get_glorys_surface_file(cfg, TARGET))
T_g   = np.asarray(gs["T_glorys_surf"].values).squeeze()
S_g   = np.asarray(gs["S_glorys_surf"].values).squeeze()

window  = cfg["processing"]["time_window_days"]
sat_dir = get_satellite_ease_dirs(cfg)
SST = load_satellite_for_time(sat_dir["SST"], TARGET, window, "SST", SAT_VARS["SST"]) - 273.15
SSS = load_satellite_for_time(sat_dir["SSS"], TARGET, window, "SSS", SAT_VARS["SSS"])
ADT = load_satellite_for_time(sat_dir["ADT"], TARGET, window, "ADT", SAT_VARS["ADT"])

inputs = {
    "T_glorys_surf": T_g,
    "S_glorys_surf": S_g,
    "SST":           SST,
    "SSS":           SSS,
    "ADT":           ADT,
}

# Build NaN-on-ocean mask per input  (1 = missing on ocean, 0 = either land or has data)
def nan_on_ocean(arr):
    m = np.zeros_like(arr, dtype=np.float32)
    m[ocean & np.isnan(arr)] = 1.0
    m[~ocean] = np.nan   # land → blank
    return m

# union: any-input-NaN on ocean
any_nan = np.zeros_like(ocean, dtype=bool)
for a in inputs.values():
    any_nan |= np.isnan(a)
union_map = np.zeros_like(ocean, dtype=np.float32)
union_map[ocean & any_nan] = 1.0
union_map[~ocean] = np.nan

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
axes = axes.ravel()

for i, (name, arr) in enumerate(inputs.items()):
    nan_o = int(np.isnan(arr[ocean]).sum())
    axes[i].imshow(nan_on_ocean(arr), origin="lower", cmap="Reds", vmin=0, vmax=1)
    axes[i].set_title(f"{name} missing on ocean\n{nan_o}/{n_oc} ({100*nan_o/n_oc:.1f}%)")
    axes[i].set_xticks([]); axes[i].set_yticks([])

n_union = int((any_nan & ocean).sum())
axes[5].imshow(union_map, origin="lower", cmap="Reds", vmin=0, vmax=1)
axes[5].set_title(f"UNION (any input missing on ocean)\n{n_union}/{n_oc} ({100*n_union/n_oc:.1f}%)")
axes[5].set_xticks([]); axes[5].set_yticks([])

fig.suptitle(f"Missing inputs on ocean — {TARGET:%Y-%m-%d}", y=1.00)
fig.tight_layout()

PLOT_DIR.mkdir(parents=True, exist_ok=True)
out = PLOT_DIR / f"missing_inputs_{TARGET:%Y%m%d}.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved → {out}")
