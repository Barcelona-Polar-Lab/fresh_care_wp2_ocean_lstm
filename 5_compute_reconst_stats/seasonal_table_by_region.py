#!/usr/bin/env python3
"""
Build the seasonal regional ΔT / ΔS table (LaTeX tabularx) from the
seasonal mean files produced by ``compute_reconst_stats.py``.

For each Arctic region (S1..S14 from the GeoJSON) and each season
(DJF / MAM / JJA / SON):

    1. Open every ``seasonal_YYYY_<SEASON>*.nc`` file.
    2. Take ``T_anom_pred_mean`` and ``S_anom_pred_mean``
       (these ARE the anomalies vs GLORYS — exactly what the table wants).
    3. Average over the upper 100 m (depth-weighted, midpoint rule).
    4. Mask cells inside the region polygon (lat/lon, shapely).
    5. Area-mean across the region (simple cell mean — EASE is
       equal-area, so cells already have equal weight).
    6. Mean across years.

Output: the LaTeX ``tabularx`` snippet only (matches the template the
user provided). Printed to stdout, optionally written via ``--output``.

Usage
-----
    python seasonal_table_by_region.py \\
        --stats-dir /home/nicolas/SACO/FRESH-CARE/Data_lstm_reconstruction/arctic_25km/product_stats \\
        --regions  /home/nicolas/SACO/FRESH-CARE/Arctic_masks/geojson_masks/arctic_regions.json \\
        [--depth-max 100] [--skip-partial] [--output table.tex]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from shapely.geometry import Point, shape
from shapely.prepared import prep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SEASONS = ["DJF", "MAM", "JJA", "SON"]
FNAME_RE = re.compile(r"^seasonal_(\d{4})_(DJF|MAM|JJA|SON)(_partial)?\.nc$")


# ---------------------------------------------------------------------------
# Region masks
# ---------------------------------------------------------------------------

def load_regions(path: Path) -> dict[str, tuple[str, object]]:
    """Return ``{id: (name_en, prepared_polygon)}`` preserving GeoJSON order."""
    with open(path) as f:
        gj = json.load(f)
    out: dict[str, tuple[str, object]] = {}
    for feat in gj["features"]:
        rid = feat["properties"]["id"]
        rname = feat["properties"]["name_en"]
        poly = shape(feat["geometry"])
        out[rid] = (rname, prep(poly))
    return out


def build_region_masks(lat2d: np.ndarray, lon2d: np.ndarray,
                       regions: dict[str, tuple[str, object]],
                       ocean_mask: np.ndarray | None
                       ) -> dict[str, np.ndarray]:
    """For each region id, a 2D boolean mask on the EASE grid."""
    ny, nx = lat2d.shape
    masks = {rid: np.zeros((ny, nx), dtype=bool) for rid in regions}
    # Only consider ocean cells if ocean_mask is provided
    if ocean_mask is not None:
        ocean = ocean_mask.astype(bool)
    else:
        ocean = np.ones((ny, nx), dtype=bool)
    # Iterate once over candidate cells
    flat_idx = np.argwhere(ocean & np.isfinite(lat2d) & np.isfinite(lon2d))
    logger.info(f"Building region masks over {len(flat_idx)} ocean cells...")
    for j, i in flat_idx:
        pt = Point(float(lon2d[j, i]), float(lat2d[j, i]))
        for rid, (_, poly) in regions.items():
            if poly.contains(pt):
                masks[rid][j, i] = True
                break
    for rid, m in masks.items():
        logger.info(f"  {rid}: {int(m.sum()):6d} cells")
    return masks


# ---------------------------------------------------------------------------
# Depth integration
# ---------------------------------------------------------------------------

def depth_weights(depth: np.ndarray, dmax: float) -> np.ndarray:
    """Layer thicknesses (midpoint rule) for ``depth <= dmax``; 0 elsewhere.

    Top thickness goes from 0 to midpoint with next level (or to dmax if
    only one level). Bottom thickness is clipped at dmax.
    """
    depth = np.asarray(depth, dtype=float)
    n = len(depth)
    w = np.zeros(n)
    sel = np.where(depth <= dmax)[0]
    if sel.size == 0:
        return w
    sel = sel[: sel[-1] + 1]  # contiguous from top
    for k_local, k in enumerate(sel):
        if k_local == 0:
            top = 0.0
        else:
            top = 0.5 * (depth[k - 1] + depth[k])
        if k_local == len(sel) - 1:
            bot = dmax
        else:
            bot = 0.5 * (depth[k] + depth[k + 1])
        bot = min(bot, dmax)
        w[k] = max(bot - top, 0.0)
    return w


def column_mean_upper(field: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Depth-weighted mean over leading depth axis; NaN-aware per cell.

    field : (D, ny, nx)
    w     : (D,)
    """
    valid = ~np.isnan(field)
    w3 = w[:, None, None]
    num = np.nansum(field * w3, axis=0)
    den = np.sum(np.where(valid, w3, 0.0), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_files(stats_dir: Path, skip_partial: bool
                   ) -> dict[str, list[tuple[int, Path]]]:
    season_dir = stats_dir / "seasonal"
    if not season_dir.is_dir():
        logger.error(f"Seasonal stats dir not found: {season_dir}")
        sys.exit(1)
    by_season: dict[str, list[tuple[int, Path]]] = {s: [] for s in SEASONS}
    for p in sorted(season_dir.glob("seasonal_*.nc")):
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        year = int(m.group(1))
        season = m.group(2)
        is_partial = m.group(3) is not None
        if is_partial and skip_partial:
            continue
        by_season[season].append((year, p))
    return by_season


def main():
    ap = argparse.ArgumentParser(
        description="Produce the seasonal ΔT/ΔS regional table (LaTeX).")
    ap.add_argument("--stats-dir", required=True, type=Path,
                    help="Dir containing seasonal/ subdir with seasonal_*.nc")
    ap.add_argument("--regions", required=True, type=Path,
                    help="Path to arctic_regions.json (GeoJSON FeatureCollection)")
    ap.add_argument("--depth-max", type=float, default=100.0,
                    help="Upper-column depth (m) to average over [default: 100]")
    ap.add_argument("--include-partial", action="store_true",
                    help="Include *_partial.nc seasonal files "
                         "(default: skip them)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional .tex file to write the tabularx snippet to")
    ap.add_argument("--t-decimals", type=int, default=2)
    ap.add_argument("--s-decimals", type=int, default=2)
    args = ap.parse_args()

    regions = load_regions(args.regions)
    logger.info(f"Loaded {len(regions)} regions from {args.regions.name}")

    files_by_season = discover_files(args.stats_dir,
                                      skip_partial=not args.include_partial)
    for s in SEASONS:
        years = [y for y, _ in files_by_season[s]]
        logger.info(f"  {s}: {len(years)} file(s) "
                    f"{(f'years {min(years)}..{max(years)}' if years else '(none)')}")

    if not any(files_by_season.values()):
        logger.error("No seasonal files found.")
        sys.exit(1)

    # Build masks lazily from the first file we touch
    region_masks: dict[str, np.ndarray] | None = None
    depth_w_cache: dict[int, np.ndarray] = {}
    # results[rid][season] = (mean_dT, mean_dS, n_years)
    results: dict[str, dict[str, tuple[float, float, int]]] = {
        rid: {s: (np.nan, np.nan, 0) for s in SEASONS} for rid in regions
    }

    for season in SEASONS:
        per_year_dT: dict[str, list[float]] = {rid: [] for rid in regions}
        per_year_dS: dict[str, list[float]] = {rid: [] for rid in regions}
        for year, path in files_by_season[season]:
            logger.info(f"[{season} {year}] {path.name}")
            with xr.open_dataset(path) as ds:
                if region_masks is None:
                    if "latitude" not in ds or "longitude" not in ds:
                        logger.error(f"{path.name} missing latitude/longitude")
                        sys.exit(1)
                    lat2d = np.asarray(ds["latitude"].values)
                    lon2d = np.asarray(ds["longitude"].values)
                    ocean_mask = (np.asarray(ds["ocean_mask"].values)
                                  if "ocean_mask" in ds else None)
                    region_masks = build_region_masks(lat2d, lon2d, regions,
                                                     ocean_mask)
                if "T_anom_pred_mean" not in ds or "S_anom_pred_mean" not in ds:
                    logger.warning(f"  missing T_anom_pred_mean / S_anom_pred_mean — skip")
                    continue
                depth = np.asarray(ds["depth"].values)
                key = int(round(depth.sum() * 1000))  # cheap cache key
                if key not in depth_w_cache:
                    depth_w_cache[key] = depth_weights(depth, args.depth_max)
                w = depth_w_cache[key]
                if w.sum() <= 0:
                    logger.warning(f"  no depth levels within {args.depth_max} m — skip")
                    continue
                dT3 = np.asarray(ds["T_anom_pred_mean"].values, dtype=float).squeeze()
                dS3 = np.asarray(ds["S_anom_pred_mean"].values, dtype=float).squeeze()
                dT2 = column_mean_upper(dT3, w)   # (ny, nx)
                dS2 = column_mean_upper(dS3, w)
            for rid, mask in region_masks.items():
                cells = mask & np.isfinite(dT2)
                if cells.any():
                    per_year_dT[rid].append(float(np.nanmean(dT2[cells])))
                cells = mask & np.isfinite(dS2)
                if cells.any():
                    per_year_dS[rid].append(float(np.nanmean(dS2[cells])))

        for rid in regions:
            tvals = per_year_dT[rid]
            svals = per_year_dS[rid]
            mT = float(np.mean(tvals)) if tvals else np.nan
            mS = float(np.mean(svals)) if svals else np.nan
            results[rid][season] = (mT, mS, max(len(tvals), len(svals)))

    # ---- Render LaTeX tabularx ----
    def fmt(x: float, nd: int) -> str:
        if not np.isfinite(x):
            return "--"
        return f"{x:+.{nd}f}"

    lines = []
    lines.append(r"\begin{tabularx}{\textwidth}{")
    lines.append(r"  l")
    lines.append(r"  *{4}{")
    lines.append(r"    @{\hspace{0.7em}}")
    lines.append(r"    >{\centering\arraybackslash}X")
    lines.append(r"    @{\hspace{-1em}}")
    lines.append(r"    >{\centering\arraybackslash}X")
    lines.append(r"    @{\hspace{1.15em}}")
    lines.append(r"  }")
    lines.append(r"}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Region \textbackslash{} Mean. Seas. Anom. ($^\circ$C)} & "
        r"\textbf{DJF $\Delta T$} & \textbf{DJF $\Delta S$} & "
        r"\textbf{MAM $\Delta T$} & \textbf{MAM $\Delta S$} & "
        r"\textbf{JJA $\Delta T$} & \textbf{JJA $\Delta S$} & "
        r"\textbf{SON $\Delta T$} & \textbf{SON $\Delta S$} \\"
    )
    lines.append(r"\hline")
    for rid, (rname, _) in regions.items():
        row = [f"{rid} {rname}"]
        for s in SEASONS:
            mT, mS, _ = results[rid][s]
            row.append(fmt(mT, args.t_decimals))
            row.append(fmt(mS, args.s_decimals))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabularx}")
    snippet = "\n".join(lines) + "\n"

    print(snippet)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(snippet)
        logger.info(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
