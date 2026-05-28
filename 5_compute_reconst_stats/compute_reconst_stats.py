#!/usr/bin/env python3
"""
Compute temporal statistics (means + stds) from a reconstruction output dir
produced by Step E (``TS_currents_lstm_YYYYMMDD.nc`` per-date files).

For each requested period type (yearly / seasonal / monthly) and each period
that has at least one day, we accumulate per-cell mean and variance with
Welford's online algorithm, then write a NetCDF file in a sibling
``product_stats/{yearly,seasonal,monthly}/`` directory.

Statistics computed (per grid cell, over the time dimension):

* Scalar vars (`T_anom_pred`, `S_anom_pred`, `T_recon`, `S_recon`,
  `T_glorys`, `S_glorys`, `ADH`):
    - ``<var>_mean``
    - ``<var>_std``         (sample std with Bessel correction)
* Velocity components (`u_gos`, `v_gos`, `vel_gos_x`, `vel_gos_y`):
    - ``<var>_mean``
    - ``<var>_std``
* Velocity covariance ellipse (from `u_gos`, `v_gos`):
    - ``vel_cov_uu``, ``vel_cov_vv``, ``vel_cov_uv``
    - ``vel_std_major``, ``vel_std_minor``  (sqrt of eigenvalues)
    - ``vel_ellipse_angle_deg``             (orientation of major axis,
                                              degrees CCW from +x / east)

Anomaly std and reconstructed std are computed independently because the
means differ (the variance of ``T_recon = T_anom + T_glorys`` differs from
the variance of ``T_anom`` alone).

The "std" produced here is the *temporal* spread across days within the
period — it is NOT the MC-Dropout std stored in the per-date files
(`T_anom_std`, `S_anom_std`), which quantifies model uncertainty.

Periods that are not fully covered (e.g. partial first/last year, a season
missing days) are still written but get a ``_partial`` suffix in the
filename and a ``partial: True`` global attribute.

Usage
-----
::

    python compute_reconst_stats.py --reconst-dir /path/to/arctic_25km \\
        --yearly --seasonal --monthly [--overwrite]

The script expects ``--reconst-dir`` to contain ``TS_currents_lstm/``;
output goes to ``<reconst-dir>/product_stats/{yearly,seasonal,monthly}/``.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration: variables and seasons
# ---------------------------------------------------------------------------

# Scalar variables: per-cell mean + per-cell std
SCALAR_VARS = [
    "T_anom_pred",
    "S_anom_pred",
    "T_recon",
    "S_recon",
    "T_glorys",
    "S_glorys",
    "ADH",
]

# Velocity components: per-component mean + per-component std
VEL_VARS = ["u_gos", "v_gos", "vel_gos_x", "vel_gos_y"]

# Covariance ellipse is computed from this pair (geographic eastward/northward)
VEL_PAIR = ("u_gos", "v_gos")

# Static vars copied verbatim from the first file
STATIC_VARS = ["ocean_mask", "elevation", "latitude", "longitude",
               "ease_grid_mapping"]

SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}
SEASON_LABEL = {"DJF": "Winter (DJF)", "MAM": "Spring (MAM)",
                "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}

DATE_RE = re.compile(r"TS_currents_lstm_(\d{8})\.nc$")


# ---------------------------------------------------------------------------
# Period assignment & coverage
# ---------------------------------------------------------------------------

def season_of(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    raise ValueError(month)


def season_year(d: date) -> int:
    """Year a date belongs to for DJF/MAM/JJA/SON grouping.

    DJF of year Y = Dec(Y-1) + Jan(Y) + Feb(Y).
    """
    if d.month == 12:
        return d.year + 1
    return d.year


def days_in_year(y: int) -> int:
    return 366 if calendar.isleap(y) else 365


def days_in_season(season: str, y: int) -> int:
    months = SEASONS[season]
    total = 0
    for m in months:
        # DJF: Dec belongs to previous calendar year
        yr = y - 1 if (season == "DJF" and m == 12) else y
        total += calendar.monthrange(yr, m)[1]
    return total


def days_in_month(y: int, m: int) -> int:
    return calendar.monthrange(y, m)[1]


def assign_periods(dates: list[date], kinds: list[str]
                   ) -> dict[str, dict[tuple, list[date]]]:
    """Return ``{kind: {period_key: [dates...]}}`` for requested kinds."""
    out: dict[str, dict[tuple, list[date]]] = {k: defaultdict(list) for k in kinds}
    if "full" in kinds and dates:
        full_key = (min(dates).year, max(dates).year)
    for d in dates:
        if "yearly" in kinds:
            out["yearly"][(d.year,)].append(d)
        if "seasonal" in kinds:
            out["seasonal"][(season_year(d), season_of(d.month))].append(d)
        if "monthly" in kinds:
            out["monthly"][(d.year, d.month)].append(d)
        if "full" in kinds:
            out["full"][full_key].append(d)
    return out


def expected_days(kind: str, key: tuple) -> int:
    if kind == "yearly":
        return days_in_year(key[0])
    if kind == "seasonal":
        return days_in_season(key[1], key[0])
    if kind == "monthly":
        return days_in_month(key[0], key[1])
    if kind == "full":
        y0, y1 = key
        return sum(days_in_year(y) for y in range(y0, y1 + 1))
    raise ValueError(kind)


def period_bounds(kind: str, key: tuple) -> tuple[date, date]:
    """Calendar start/end dates of a period (resolution-independent)."""
    if kind == "yearly":
        y = key[0]
        return date(y, 1, 1), date(y, 12, 31)
    if kind == "seasonal":
        y, season = key
        if season == "DJF":
            return date(y - 1, 12, 1), date(y, 2, calendar.monthrange(y, 2)[1])
        m0 = SEASONS[season][0]
        m1 = SEASONS[season][-1]
        return date(y, m0, 1), date(y, m1, calendar.monthrange(y, m1)[1])
    if kind == "monthly":
        y, m = key
        return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])
    if kind == "full":
        # 'full' uses the actual dataset coverage — it can never be partial
        # by construction. Bounds are overridden in process_period().
        y0, y1 = key
        return date(y0, 1, 1), date(y1, 12, 31)
    raise ValueError(kind)


def period_filename(kind: str, key: tuple, partial: bool) -> str:
    if kind == "yearly":
        base = f"yearly_{key[0]:04d}"
    elif kind == "seasonal":
        base = f"seasonal_{key[0]:04d}_{key[1]}"
    elif kind == "monthly":
        base = f"monthly_{key[0]:04d}{key[1]:02d}"
    elif kind == "full":
        base = f"mean_{key[0]:04d}_{key[1]:04d}"
    else:
        raise ValueError(kind)
    if partial:
        base += "_partial"
    return base + ".nc"


def period_label(kind: str, key: tuple) -> str:
    if kind == "yearly":
        return f"{key[0]}"
    if kind == "seasonal":
        return f"{SEASON_LABEL[key[1]]} {key[0]}"
    if kind == "monthly":
        return f"{calendar.month_name[key[1]]} {key[0]}"
    if kind == "full":
        return f"{key[0]}–{key[1]} (full record)"
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Welford accumulators
# ---------------------------------------------------------------------------

class WelfordScalar:
    """Per-cell online mean + M2 for one scalar variable, NaN-aware."""

    __slots__ = ("n", "mean", "m2", "units", "long_name", "extra_attrs",
                 "dims", "_init")

    def __init__(self):
        self.n = None        # int32 array, count of non-NaN observations per cell
        self.mean = None     # float32
        self.m2 = None       # float32 (sum of squared deviations from running mean)
        self.units = None
        self.long_name = None
        self.extra_attrs = {}
        self.dims = None
        self._init = False

    def _initialize(self, da: xr.DataArray):
        shape = da.shape
        self.n = np.zeros(shape, dtype=np.int32)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.dims = da.dims
        # Preserve metadata from the first file
        self.units = da.attrs.get("units")
        self.long_name = da.attrs.get("long_name")
        for k, v in da.attrs.items():
            if k not in ("units", "long_name"):
                self.extra_attrs[k] = v
        self._init = True

    def update(self, da: xr.DataArray):
        if not self._init:
            self._initialize(da)
        x = np.asarray(da.values, dtype=np.float64)
        mask = ~np.isnan(x)
        # Welford update on valid cells only
        self.n[mask] += 1
        n_arr = self.n
        # delta = x - mean (only where mask)
        delta = np.where(mask, x - self.mean, 0.0)
        # mean += delta / n  (only where mask, n>0 guaranteed there)
        safe_n = np.where(mask, n_arr, 1)
        self.mean += delta / safe_n
        delta2 = np.where(mask, x - self.mean, 0.0)
        self.m2 += delta * delta2

    def finalize(self):
        """Return (mean, std) float32 arrays. std is NaN where n < 2."""
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(self.n > 0, self.mean, np.nan).astype(np.float32)
            var = np.where(self.n > 1, self.m2 / (self.n - 1), np.nan)
            std = np.sqrt(var).astype(np.float32)
        return mean, std


class WelfordCovariance:
    """Per-cell online co-moment accumulator for two scalar fields (u, v)."""

    __slots__ = ("n", "mean_u", "mean_v", "m2_u", "m2_v", "c_uv", "_init",
                 "dims")

    def __init__(self):
        self.n = None
        self.mean_u = None
        self.mean_v = None
        self.m2_u = None
        self.m2_v = None
        self.c_uv = None
        self.dims = None
        self._init = False

    def _initialize(self, da_u: xr.DataArray):
        shape = da_u.shape
        self.n = np.zeros(shape, dtype=np.int32)
        self.mean_u = np.zeros(shape, dtype=np.float64)
        self.mean_v = np.zeros(shape, dtype=np.float64)
        self.m2_u = np.zeros(shape, dtype=np.float64)
        self.m2_v = np.zeros(shape, dtype=np.float64)
        self.c_uv = np.zeros(shape, dtype=np.float64)
        self.dims = da_u.dims
        self._init = True

    def update(self, da_u: xr.DataArray, da_v: xr.DataArray):
        if not self._init:
            self._initialize(da_u)
        u = np.asarray(da_u.values, dtype=np.float64)
        v = np.asarray(da_v.values, dtype=np.float64)
        mask = ~(np.isnan(u) | np.isnan(v))
        self.n[mask] += 1
        n_arr = self.n
        safe_n = np.where(mask, n_arr, 1)
        du = np.where(mask, u - self.mean_u, 0.0)
        dv = np.where(mask, v - self.mean_v, 0.0)
        self.mean_u += du / safe_n
        self.mean_v += dv / safe_n
        du2 = np.where(mask, u - self.mean_u, 0.0)
        dv2 = np.where(mask, v - self.mean_v, 0.0)
        self.m2_u += du * du2
        self.m2_v += dv * dv2
        # Co-moment (Welford 2-pass-equivalent): use post-update delta of one,
        # pre-update delta of the other, multiplied. Standard form:
        #   C_{n} = C_{n-1} + (u - mean_u_new) * (v - mean_v_old)
        # Both are equivalent; we use du2 * dv (post-u, pre-v).
        self.c_uv += du2 * dv

    def finalize(self):
        """Return dict with cov components + ellipse params (float32)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            denom = self.n - 1
            valid = self.n > 1
            cov_uu = np.where(valid, self.m2_u / denom, np.nan)
            cov_vv = np.where(valid, self.m2_v / denom, np.nan)
            cov_uv = np.where(valid, self.c_uv / denom, np.nan)
            # 2x2 symmetric eigendecomposition (analytic):
            #   eig = (trace ± sqrt(trace^2 - 4*det)) / 2
            trace = cov_uu + cov_vv
            det = cov_uu * cov_vv - cov_uv * cov_uv
            disc = np.maximum(trace * trace - 4.0 * det, 0.0)
            sqrt_disc = np.sqrt(disc)
            lam1 = 0.5 * (trace + sqrt_disc)   # major
            lam2 = 0.5 * (trace - sqrt_disc)   # minor
            std_major = np.sqrt(np.maximum(lam1, 0.0))
            std_minor = np.sqrt(np.maximum(lam2, 0.0))
            # Orientation of major axis (radians, CCW from +x):
            #   theta = 0.5 * atan2(2*cov_uv, cov_uu - cov_vv)
            angle = 0.5 * np.arctan2(2.0 * cov_uv, cov_uu - cov_vv)
            angle_deg = np.degrees(angle)
        return {
            "vel_cov_uu": cov_uu.astype(np.float32),
            "vel_cov_vv": cov_vv.astype(np.float32),
            "vel_cov_uv": cov_uv.astype(np.float32),
            "vel_std_major": std_major.astype(np.float32),
            "vel_std_minor": std_minor.astype(np.float32),
            "vel_ellipse_angle_deg": angle_deg.astype(np.float32),
        }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def parse_date_from_path(p: Path) -> date | None:
    m = DATE_RE.search(p.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def discover_files(input_dir: Path) -> dict[date, Path]:
    out = {}
    for p in sorted(input_dir.glob("TS_currents_lstm_*.nc")):
        d = parse_date_from_path(p)
        if d is not None:
            out[d] = p
    return out


# ---------------------------------------------------------------------------
# Per-period processing
# ---------------------------------------------------------------------------

def process_period(kind: str, key: tuple, dates: list[date],
                   files_by_date: dict[date, Path],
                   out_dir: Path, overwrite: bool,
                   first_file_for_coords: Path,
                   dataset_min: date, dataset_max: date) -> None:
    n_expected = expected_days(kind, key)
    n_have = len(dates)
    p_start, p_end = period_bounds(kind, key)
    if kind == "full":
        # The 'full' period IS the dataset coverage by definition.
        p_start, p_end = dataset_min, dataset_max
        partial = False
    else:
        # Partial only when the period extends beyond the dataset's time
        # coverage — NOT when n_have < n_expected (the daily product has
        # sub-daily resolution gaps by design; e.g. 3-day cadence ⇒ ~122
        # files/year is "full").
        partial = (p_start < dataset_min) or (p_end > dataset_max)

    out_name = period_filename(kind, key, partial)
    out_path = out_dir / out_name
    if out_path.exists() and not overwrite:
        logger.info(f"  [{kind}] {out_name} exists — skipping")
        return

    logger.info(f"  [{kind}] {period_label(kind, key)}: "
                f"{n_have} file(s) "
                f"[{p_start}..{p_end}] "
                f"({'PARTIAL' if partial else 'full'})")

    # Allocate accumulators
    scalar_accums: dict[str, WelfordScalar] = {v: WelfordScalar() for v in SCALAR_VARS}
    vel_accums: dict[str, WelfordScalar] = {v: WelfordScalar() for v in VEL_VARS}
    vel_cov = WelfordCovariance()

    # Iterate files (one day at a time → minimal RAM)
    coord_template = None
    src_global_attrs = None
    static_arrays: dict[str, xr.DataArray] = {}
    time_start = min(dates)
    time_end = max(dates)
    used_paths: list[str] = []

    for d in sorted(dates):
        path = files_by_date[d]
        used_paths.append(path.name)
        with xr.open_dataset(path) as ds:
            if coord_template is None:
                # Capture coords / global attrs / static fields once
                coord_template = {
                    "depth": ds["depth"] if "depth" in ds.coords else None,
                    "y_ease": ds["y_ease"],
                    "x_ease": ds["x_ease"],
                }
                src_global_attrs = dict(ds.attrs)
                for sv in STATIC_VARS:
                    if sv in ds:
                        static_arrays[sv] = ds[sv].load()
            # Drop the time dim (single-element) from data variables before update
            for v in SCALAR_VARS:
                if v in ds:
                    da = ds[v].isel(time=0) if "time" in ds[v].dims else ds[v]
                    scalar_accums[v].update(da)
            for v in VEL_VARS:
                if v in ds:
                    da = ds[v].isel(time=0) if "time" in ds[v].dims else ds[v]
                    vel_accums[v].update(da)
            if VEL_PAIR[0] in ds and VEL_PAIR[1] in ds:
                du = ds[VEL_PAIR[0]]
                dv = ds[VEL_PAIR[1]]
                if "time" in du.dims:
                    du = du.isel(time=0)
                    dv = dv.isel(time=0)
                vel_cov.update(du, dv)

    # ---- Build output dataset ----
    coords = {}
    if coord_template["depth"] is not None:
        coords["depth"] = coord_template["depth"]
    coords["y_ease"] = coord_template["y_ease"]
    coords["x_ease"] = coord_template["x_ease"]

    ds_out = xr.Dataset(coords=coords)

    def _store_mean_std(varname: str, accum: WelfordScalar):
        if not accum._init:
            return
        mean, std = accum.finalize()
        base_long = accum.long_name or varname
        base_units = accum.units or ""
        common_attrs = {k: v for k, v in accum.extra_attrs.items()}
        mean_attrs = dict(common_attrs)
        mean_attrs["long_name"] = f"Temporal mean of {base_long}"
        mean_attrs["units"] = base_units
        mean_attrs["cell_methods"] = "time: mean"
        std_attrs = dict(common_attrs)
        std_attrs["long_name"] = f"Temporal standard deviation of {base_long}"
        std_attrs["units"] = base_units
        std_attrs["cell_methods"] = "time: standard_deviation"
        ds_out[f"{varname}_mean"] = xr.DataArray(mean, dims=accum.dims,
                                                 attrs=mean_attrs)
        ds_out[f"{varname}_std"] = xr.DataArray(std, dims=accum.dims,
                                                attrs=std_attrs)

    for v in SCALAR_VARS:
        _store_mean_std(v, scalar_accums[v])
    for v in VEL_VARS:
        _store_mean_std(v, vel_accums[v])

    # Velocity covariance ellipse
    if vel_cov._init:
        cov = vel_cov.finalize()
        dims = vel_cov.dims
        u_units = vel_accums[VEL_PAIR[0]].units or "m s-1"
        cov_units = f"({u_units})^2"
        ellipse_attrs_common = {
            "source_components": f"{VEL_PAIR[0]}, {VEL_PAIR[1]}",
            "cell_methods": "time: covariance",
        }
        ds_out["vel_cov_uu"] = xr.DataArray(
            cov["vel_cov_uu"], dims=dims,
            attrs={"long_name": f"Variance of {VEL_PAIR[0]}",
                   "units": cov_units, **ellipse_attrs_common})
        ds_out["vel_cov_vv"] = xr.DataArray(
            cov["vel_cov_vv"], dims=dims,
            attrs={"long_name": f"Variance of {VEL_PAIR[1]}",
                   "units": cov_units, **ellipse_attrs_common})
        ds_out["vel_cov_uv"] = xr.DataArray(
            cov["vel_cov_uv"], dims=dims,
            attrs={"long_name": f"Covariance of ({VEL_PAIR[0]}, {VEL_PAIR[1]})",
                   "units": cov_units, **ellipse_attrs_common})
        ds_out["vel_std_major"] = xr.DataArray(
            cov["vel_std_major"], dims=dims,
            attrs={"long_name": "Velocity variability ellipse: major semi-axis "
                                "(sqrt of largest eigenvalue of 2x2 covariance)",
                   "units": u_units, **ellipse_attrs_common})
        ds_out["vel_std_minor"] = xr.DataArray(
            cov["vel_std_minor"], dims=dims,
            attrs={"long_name": "Velocity variability ellipse: minor semi-axis "
                                "(sqrt of smallest eigenvalue of 2x2 covariance)",
                   "units": u_units, **ellipse_attrs_common})
        ds_out["vel_ellipse_angle_deg"] = xr.DataArray(
            cov["vel_ellipse_angle_deg"], dims=dims,
            attrs={"long_name": "Orientation of velocity variability ellipse "
                                "major axis, CCW from eastward",
                   "units": "degree", **ellipse_attrs_common})

    # Static fields (copied verbatim, keep attrs incl. grid_mapping)
    for sv, da in static_arrays.items():
        ds_out[sv] = da

    # Per-cell sample count (useful diagnostic; same shape as scalar accum)
    any_scalar = next((a for a in scalar_accums.values() if a._init), None)
    if any_scalar is not None:
        ds_out["n_samples"] = xr.DataArray(
            any_scalar.n.astype(np.int32), dims=any_scalar.dims,
            attrs={"long_name": "Number of valid daily samples per cell",
                   "units": "1"})

    # Global attributes: preserve source, override title and add period info
    g = dict(src_global_attrs) if src_global_attrs else {}
    g.pop("reconstruction_date", None)
    g["title"] = (f"Temporal statistics ({kind}) — {period_label(kind, key)} "
                  f"— derived from per-date LSTM reconstruction")
    g["history"] = (f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} — "
                    f"compute_reconst_stats.py: {kind} mean & std over "
                    f"{n_have} daily files")
    g["period_type"] = kind
    g["period_label"] = period_label(kind, key)
    if kind == "yearly":
        g["period_year"] = int(key[0])
    elif kind == "seasonal":
        g["period_year"] = int(key[0])
        g["period_season"] = key[1]
    elif kind == "monthly":
        g["period_year"] = int(key[0])
        g["period_month"] = int(key[1])
    elif kind == "full":
        g["period_year_start"] = int(key[0])
        g["period_year_end"] = int(key[1])
    g["n_files_used"] = int(n_have)
    g["n_days_in_period"] = int(n_expected)
    g["period_start"] = p_start.strftime("%Y-%m-%d")
    g["period_end"] = p_end.strftime("%Y-%m-%d")
    g["partial"] = "true" if partial else "false"
    g["time_coverage_start"] = time_start.strftime("%Y-%m-%d")
    g["time_coverage_end"] = time_end.strftime("%Y-%m-%d")
    g["source_files_first"] = used_paths[0]
    g["source_files_last"] = used_paths[-1]
    ds_out.attrs = g

    # Encoding: compressed float32 for data vars
    encoding = {}
    for vname, da in ds_out.data_vars.items():
        if np.issubdtype(da.dtype, np.floating):
            encoding[vname] = {"zlib": True, "complevel": 4,
                               "_FillValue": np.float32(np.nan)}
        elif np.issubdtype(da.dtype, np.integer):
            encoding[vname] = {"zlib": True, "complevel": 4}

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    ds_out.to_netcdf(tmp_path, encoding=encoding)
    tmp_path.replace(out_path)
    ds_out.close()
    logger.info(f"  [{kind}] Saved → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compute yearly/seasonal/monthly statistics from an LSTM "
                    "reconstruction output directory.")
    ap.add_argument("--reconst-dir", required=True, type=Path,
                    help="Parent dir containing TS_currents_lstm/ "
                         "(e.g. .../arctic_25km). product_stats/ will be "
                         "created as a sibling of TS_currents_lstm/.")
    ap.add_argument("--yearly", action="store_true",
                    help="Compute yearly means + stds.")
    ap.add_argument("--seasonal", action="store_true",
                    help="Compute seasonal (DJF/MAM/JJA/SON) means + stds.")
    ap.add_argument("--monthly", action="store_true",
                    help="Compute monthly means + stds.")
    ap.add_argument("--full", action="store_true",
                    help="Compute a single mean + std over the full dataset "
                         "time coverage (file: mean_YYYY_YYYY.nc).")
    ap.add_argument("--input-subdir", default="TS_currents_lstm",
                    help="Subdir name under --reconst-dir holding daily files "
                         "(default: TS_currents_lstm).")
    ap.add_argument("--output-subdir", default="product_stats",
                    help="Subdir name under --reconst-dir for outputs "
                         "(default: product_stats).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing period output files.")
    args = ap.parse_args()

    kinds = []
    if args.yearly:
        kinds.append("yearly")
    if args.seasonal:
        kinds.append("seasonal")
    if args.monthly:
        kinds.append("monthly")
    if args.full:
        kinds.append("full")
    if not kinds:
        ap.error("Pass at least one of --yearly / --seasonal / --monthly / --full.")

    input_dir = args.reconst_dir / args.input_subdir
    if not input_dir.is_dir():
        logger.error(f"Input dir not found: {input_dir}")
        sys.exit(1)

    files_by_date = discover_files(input_dir)
    if not files_by_date:
        logger.error(f"No TS_currents_lstm_YYYYMMDD.nc files in {input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(files_by_date)} daily files in {input_dir}")
    logger.info(f"  Range: {min(files_by_date)} → {max(files_by_date)}")

    out_root = args.reconst_dir / args.output_subdir
    first_file = files_by_date[min(files_by_date)]

    groups = assign_periods(sorted(files_by_date.keys()), kinds)
    dataset_min = min(files_by_date)
    dataset_max = max(files_by_date)

    for kind in kinds:
        out_dir = out_root / kind
        keys = sorted(groups[kind].keys())
        logger.info("=" * 60)
        logger.info(f"[{kind}] {len(keys)} period(s) → {out_dir}")
        logger.info("=" * 60)
        for key in keys:
            dates = groups[kind][key]
            process_period(kind, key, dates, files_by_date,
                           out_dir, args.overwrite, first_file,
                           dataset_min, dataset_max)

    logger.info("Done.")


if __name__ == "__main__":
    main()
