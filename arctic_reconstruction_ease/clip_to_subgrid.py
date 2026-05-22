#!/usr/bin/env python3
"""
Clip existing pipeline outputs / intermediates to a smaller subgrid that
matches an updated YAML configuration.

The new YAML defines a smaller EASE grid (centered, multiple-of-resolution
extent, e.g. 280x280 instead of 350x350) and OPTIONALLY a lat/lon bounding
box under ``grid:`` (``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``).
Cells inside the smaller rectangle but outside the bounding box are
masked: float variables get NaN-ed and the ``ocean_mask`` uint8 var is
AND-ed with the bounding-box mask (so downstream code that filters by
``ocean_mask == 1`` automatically restricts to the new domain).

Public primitives — intended to be reused later from the pipeline itself
(``A_create_ocean_mask.py`` / ``E_finalize_with_glorys.py``) so the clip
logic stays single-sourced:

    compute_latlon_mask(lat2d, lon2d, cfg) -> 2D bool | None
        Return the keep-mask from optional grid.lat_min/lat_max/
        lon_min/lon_max bounds, or None if none are set.
    compute_subgrid_slice(x_src, y_src, x_tgt, y_tgt) -> (slice_y, slice_x)
        Symmetric central integer slice such that
        ``x_src[slice_x] == x_tgt`` (and same for y) to ~1e-5 m.
    clip_dataset(ds, slice_y, slice_x, mask2d, attrs_extra) -> Dataset
        Slice spatial vars to the subgrid; apply mask2d (NaN for floats,
        AND for ocean_mask); update global attrs.
    clip_one_file(src, dst, cfg, mask_cache, force) -> str
        Open + clip + write atomically. Returns "ok" / "skip" / "error".

CLI (smoke test + bulk clip):
    python clip_to_subgrid.py --config CONFIG.yaml \\
            --src-root SRC --dst-root DST \\
            [--subtree {all,product,intermediates,static,anomalies,
                        glorys_surface,satellite,final}] \\
            [--limit K] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time as _time
from pathlib import Path

import numpy as np
import xarray as xr

# Local imports (kept light so the pipeline can import these primitives).
from config_utils import (
    atomic_to_netcdf,
    build_var_encoding,
    compute_latlon_grids,
    create_ease_grid,
    load_config,
    format_eta,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# Tolerance for matching slice coordinates against freshly-built target
# coordinates. EASE coords are 25 km integers; 1e-5 m == 1e-8 km is way
# below any meaningful precision and rules out off-by-one slicing.
_COORD_ATOL_M = 1e-5

# Latitude tolerance for non-strict inequalities (lat_min etc.).
_LATLON_ATOL_DEG = 1e-5


# ----------------------------------------------------------------------------
# Reusable primitives
# ----------------------------------------------------------------------------


def compute_latlon_mask(lat2d: np.ndarray, lon2d: np.ndarray, cfg: dict):
    """Build a 2-D boolean keep-mask from cfg['grid'] lat/lon bounds.

    Non-strict inequalities (``>=`` / ``<=``) with a small tolerance so
    that a boundary value of e.g. ``lat_min: 60`` keeps cells at
    lat ≈ 59.99999. Returns ``None`` if no bound is set.

    Recognised keys under ``cfg['grid']``: ``lat_min``, ``lat_max``,
    ``lon_min``, ``lon_max`` (any subset; missing = unbounded on that
    side).
    """
    grid = cfg.get('grid', {})
    lat_min = grid.get('lat_min')
    lat_max = grid.get('lat_max')
    lon_min = grid.get('lon_min')
    lon_max = grid.get('lon_max')
    if all(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
        return None

    mask = np.ones(lat2d.shape, dtype=bool)
    if lat_min is not None:
        mask &= lat2d >= (lat_min - _LATLON_ATOL_DEG)
    if lat_max is not None:
        mask &= lat2d <= (lat_max + _LATLON_ATOL_DEG)
    if lon_min is not None:
        mask &= lon2d >= (lon_min - _LATLON_ATOL_DEG)
    if lon_max is not None:
        mask &= lon2d <= (lon_max + _LATLON_ATOL_DEG)
    return mask


def compute_subgrid_slice(x_src: np.ndarray, y_src: np.ndarray,
                          x_tgt: np.ndarray, y_tgt: np.ndarray):
    """Return ``(slice_y, slice_x)`` such that ``x_src[slice_x] == x_tgt``.

    Requires a symmetric central placement (which is the case when both
    grids share the same ``center_x_m/center_y_m`` and resolution).
    Raises ``ValueError`` if no exact match can be made within
    ``_COORD_ATOL_M`` meters.
    """
    def _one(src, tgt, name):
        if len(tgt) > len(src):
            raise ValueError(f"{name}: target ({len(tgt)}) is larger than "
                             f"source ({len(src)}) — cannot clip.")
        n_drop = len(src) - len(tgt)
        if n_drop % 2 != 0:
            raise ValueError(f"{name}: asymmetric drop ({n_drop}) — source "
                             "and target grids do not share the same center.")
        i0 = n_drop // 2
        i1 = i0 + len(tgt)
        sub = src[i0:i1]
        if not np.allclose(sub, tgt, atol=_COORD_ATOL_M, rtol=0):
            raise ValueError(f"{name}: central slice does not match target "
                             f"(max diff {np.max(np.abs(sub - tgt))} m).")
        return slice(i0, i1)

    return _one(y_src, y_tgt, 'y_ease'), _one(x_src, x_tgt, 'x_ease')


# Variables that are pure geometry / metadata — we slice them spatially
# (if they have x_ease/y_ease dims) but we do NOT apply the latlon mask
# to them. Bathymetry and lat/lon grids are valid everywhere on the grid.
_GEOMETRY_VARS = {'latitude', 'longitude', 'elevation', 'ease_grid_mapping'}


def _apply_mask_to_var(da: xr.DataArray, mask2d: np.ndarray,
                       var_name: str) -> xr.DataArray:
    """Apply a 2-D ``mask2d`` (True = keep) to a DataArray with x/y dims.

    - ``ocean_mask`` (uint8): bitwise AND with the mask.
    - Float dtypes: set masked cells to NaN (broadcasted across leading
      dims).
    - Other integer dtypes (e.g. flag vars we don't ship): leave alone.
    """
    if var_name in _GEOMETRY_VARS:
        return da
    if var_name == 'ocean_mask':
        new = da.values.copy()
        new[~mask2d] = 0
        return xr.DataArray(new, dims=da.dims, coords=da.coords,
                            attrs=da.attrs)
    if np.issubdtype(da.dtype, np.floating):
        new = da.values.copy()
        # Broadcast mask across any leading dims (time, depth, ...).
        new[..., ~mask2d] = np.nan
        return xr.DataArray(new, dims=da.dims, coords=da.coords,
                            attrs=da.attrs)
    return da


def _is_spatial(da: xr.DataArray) -> bool:
    return ('x_ease' in da.dims) and ('y_ease' in da.dims)


def clip_dataset(ds: xr.Dataset, slice_y: slice, slice_x: slice,
                 mask2d: np.ndarray | None,
                 attrs_extra: dict | None = None,
                 cfg: dict | None = None) -> xr.Dataset:
    """Slice + (optionally) mask a single dataset.

    - Spatial vars (with both x_ease and y_ease): subset with isel,
      then apply ``mask2d`` (NaN for floats, AND for ocean_mask).
    - Non-spatial vars / coords (time, depth, ease_grid_mapping, DOY):
      pass through unchanged.
    - Global attrs: preserved; ``attrs_extra`` merged on top.
    - If ``cfg`` is provided and bounds are set, ``geospatial_lat_min``
      / ``lat_max`` / ``lon_min`` / ``lon_max`` global attrs are added.
    """
    ds_sub = ds.isel(y_ease=slice_y, x_ease=slice_x)

    if mask2d is not None:
        new_vars = {}
        for v in ds_sub.data_vars:
            da = ds_sub[v]
            if _is_spatial(da):
                new_vars[v] = _apply_mask_to_var(da, mask2d, v)
            else:
                new_vars[v] = da
        ds_sub = xr.Dataset(new_vars, coords=ds_sub.coords, attrs=ds_sub.attrs)

    # Tag ocean_mask attrs so a downstream reader knows it now reflects
    # both the shapefile ocean mask AND the lat/lon domain bounds.
    if mask2d is not None and 'ocean_mask' in ds_sub.data_vars:
        oa = dict(ds_sub['ocean_mask'].attrs)
        comment = oa.get('comment', '')
        tag = ('Restricted to the configured lat/lon domain '
               '(grid.lat_min/lat_max/lon_min/lon_max in YAML).')
        if tag not in comment:
            oa['comment'] = (comment + ' ' if comment else '') + tag
        ds_sub['ocean_mask'] = ds_sub['ocean_mask'].assign_attrs(oa)

    # Global attrs: copy + refresh geometry attrs from the actual sliced
    # shape (otherwise ``grid_size`` etc. would still reflect the source
    # 350x350 grid) + add geospatial bounds.
    new_attrs = dict(ds.attrs)
    ny_new = ds_sub.sizes.get('y_ease')
    nx_new = ds_sub.sizes.get('x_ease')
    if nx_new is not None and ny_new is not None:
        # Only overwrite if the attr was already present (we don't want to
        # inject new attrs into files that didn't carry them originally).
        if 'grid_size' in new_attrs:
            new_attrs['grid_size'] = f'{nx_new} x {ny_new}'
    if cfg is not None:
        grid = cfg.get('grid', {})
        for key_yaml, key_attr in (
            ('lat_min', 'geospatial_lat_min'),
            ('lat_max', 'geospatial_lat_max'),
            ('lon_min', 'geospatial_lon_min'),
            ('lon_max', 'geospatial_lon_max'),
        ):
            if grid.get(key_yaml) is not None:
                new_attrs[key_attr] = float(grid[key_yaml])
    if attrs_extra:
        new_attrs.update(attrs_extra)
    ds_sub.attrs = new_attrs

    return ds_sub


# ----------------------------------------------------------------------------
# Mask cache (computed once from the YAML)
# ----------------------------------------------------------------------------


class _MaskCache:
    """Builds and caches the target-grid slice + lat/lon mask from cfg.

    The slice is computed against the *source* x_ease/y_ease of each
    file (lazily, on first use). The lat/lon mask is computed once on
    the target grid.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.x_tgt, self.y_tgt, _ = create_ease_grid(cfg)
        lon2d_tgt, lat2d_tgt = compute_latlon_grids(self.x_tgt, self.y_tgt, cfg)
        self.mask2d = compute_latlon_mask(lat2d_tgt, lon2d_tgt, cfg)
        self._slice = None  # (slice_y, slice_x), cached after first file

    def get_slice(self, x_src: np.ndarray, y_src: np.ndarray):
        if self._slice is None:
            self._slice = compute_subgrid_slice(
                x_src, y_src, self.x_tgt, self.y_tgt)
            sy, sx = self._slice
            logger.info(
                f"  Subgrid slice: y_ease[{sy.start}:{sy.stop}], "
                f"x_ease[{sx.start}:{sx.stop}] "
                f"(source {len(y_src)}x{len(x_src)} -> "
                f"target {len(self.y_tgt)}x{len(self.x_tgt)})")
            if self.mask2d is not None:
                n_keep = int(self.mask2d.sum())
                n_tot = self.mask2d.size
                logger.info(
                    f"  lat/lon mask: keep {n_keep}/{n_tot} cells "
                    f"({100.0 * n_keep / n_tot:.1f}%)")
        return self._slice


# ----------------------------------------------------------------------------
# Per-file processing
# ----------------------------------------------------------------------------


def clip_one_file(src: Path, dst: Path, cfg: dict,
                  mask_cache: _MaskCache, force: bool = False) -> str:
    """Clip a single .nc file. Returns 'ok' / 'skip' / 'error'."""
    if dst.exists() and not force:
        return 'skip'
    try:
        # decode_times=False keeps the raw numeric time values + the
        # 'units'/'calendar' attributes, so we can copy them verbatim
        # into the output file (no decode/encode round-trip = no risk of
        # changing the epoch or dtype).
        with xr.open_dataset(src, decode_times=False) as ds:
            x_src = ds['x_ease'].values
            y_src = ds['y_ease'].values
            slice_y, slice_x = mask_cache.get_slice(x_src, y_src)

            ds_clip = clip_dataset(
                ds, slice_y, slice_x, mask_cache.mask2d, cfg=cfg)
            ds_clip.load()  # materialize before closing source

        # Encoding: rebuild via the same helper the pipeline uses so the
        # output file shares the int16 quantization scheme. Time is left
        # alone: because we opened with decode_times=False, the time
        # variable is plain numeric and its 'units'/'calendar' attrs are
        # written verbatim by to_netcdf — no decode/encode round-trip,
        # so the original epoch and calendar are preserved bit-for-bit.
        encoding = build_var_encoding(ds_clip)

        dst.parent.mkdir(parents=True, exist_ok=True)
        atomic_to_netcdf(ds_clip, dst, encoding=encoding)
        ds_clip.close()
        return 'ok'
    except Exception as e:
        logger.error(f"  FAILED {src.name}: {e}")
        return 'error'


# ----------------------------------------------------------------------------
# Tree walking
# ----------------------------------------------------------------------------


# Subtree -> relative path (under root). 'final' = product files.
_SUBTREES = {
    'static':         'intermediate_files/static',
    'glorys_surface': 'intermediate_files/glorys_surface',
    'anomalies':      'intermediate_files/anomalies',
    'satellite':      'intermediate_files/satellite_ease',
    'final':          'final_TS_reconstruction',
}
_GROUPS = {
    'product':        ['final'],
    'intermediates':  ['static', 'glorys_surface', 'anomalies', 'satellite'],
    'all':            ['static', 'glorys_surface', 'anomalies', 'satellite',
                       'final'],
}


def _expand_subtree(name: str) -> list[str]:
    if name in _GROUPS:
        return _GROUPS[name]
    if name in _SUBTREES:
        return [name]
    raise ValueError(f"Unknown subtree {name!r}. Pick from "
                     f"{sorted(set(_SUBTREES) | set(_GROUPS))}.")


def _iter_files(src_root: Path, subtree_name: str):
    rel = _SUBTREES[subtree_name]
    sub = src_root / rel
    if not sub.exists():
        logger.warning(f"  [{subtree_name}] missing in source: {sub}")
        return
    for f in sorted(sub.rglob('*.nc')):
        yield subtree_name, rel, f


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True,
                   help="YAML config describing the TARGET (clipped) grid.")
    p.add_argument('--src-root', required=True,
                   help="Source root, e.g. .../arctic_25km/")
    p.add_argument('--dst-root', required=True,
                   help="Destination root, e.g. .../arctic_25km_clipped_staging/")
    p.add_argument('--subtree', default='all',
                   help="Subtree(s) to clip. Choices: all, product, "
                        "intermediates, static, anomalies, glorys_surface, "
                        "satellite, final. Comma-separated allowed.")
    p.add_argument('--limit', type=int, default=None,
                   help="Per-subtree file cap (smoke testing).")
    p.add_argument('--force', action='store_true',
                   help="Overwrite existing files at destination.")
    args = p.parse_args()

    cfg = load_config(args.config)
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if src_root == dst_root:
        sys.exit("ERROR: --src-root and --dst-root must differ.")

    subtrees = []
    for tok in args.subtree.split(','):
        subtrees.extend(_expand_subtree(tok.strip()))
    subtrees = list(dict.fromkeys(subtrees))  # de-dupe, preserve order

    logger.info("=" * 60)
    logger.info("Clip to subgrid")
    logger.info(f"  Config:    {args.config}")
    logger.info(f"  Source:    {src_root}")
    logger.info(f"  Dest:      {dst_root}")
    logger.info(f"  Subtrees:  {subtrees}")
    logger.info(f"  Target:    {cfg['grid']['n_cells_x']}x{cfg['grid']['n_cells_y']} "
                f"@ {cfg['grid']['resolution_km']} km")
    g = cfg['grid']
    bounds = {k: g.get(k) for k in ('lat_min', 'lat_max', 'lon_min', 'lon_max')}
    logger.info(f"  Bounds:    {bounds}")
    logger.info(f"  limit={args.limit}  force={args.force}")
    logger.info("=" * 60)

    mask_cache = _MaskCache(cfg)

    grand_ok = grand_skip = grand_err = 0
    t_total = _time.monotonic()

    for st in subtrees:
        files = list(_iter_files(src_root, st))
        if args.limit is not None:
            files = files[:args.limit]
        if not files:
            logger.info(f"[{st}] no files.")
            continue
        logger.info(f"[{st}] {len(files)} files")
        n_ok = n_skip = n_err = 0
        t0 = _time.monotonic()
        for i, (_, rel, src) in enumerate(files, 1):
            rel_path = src.relative_to(src_root / rel)
            dst = dst_root / rel / rel_path
            status = clip_one_file(src, dst, cfg, mask_cache, force=args.force)
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            else:
                n_err += 1
            if i % 50 == 0 or i == len(files):
                logger.info(
                    f"  [{st}] {i}/{len(files)}  "
                    f"ok={n_ok} skip={n_skip} err={n_err}  "
                    f"{format_eta(t0, i, len(files), skipped=n_skip)}")
        grand_ok += n_ok; grand_skip += n_skip; grand_err += n_err

    elapsed = _time.monotonic() - t_total
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.1f}s — ok={grand_ok} "
                f"skip={grand_skip} err={grand_err}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
