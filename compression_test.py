"""
Compression test for reconstruction netCDF files.
Tries multiple encoding strategies and reports resulting file sizes.
"""

import os
import time
import numpy as np
import xarray as xr
import netCDF4 as nc4

SRC = (
    "/home/nicolas/SACO/FRESH-CARE/Data_lstm_reconstruction/"
    "first_test_2012_monthly/reconstruction_outputs/reconstruction_data/"
    "reconstruction_20120115.nc"
)
OUT_DIR = "/home/nicolas/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/temp_files"
SRC_SIZE = os.path.getsize(SRC) / 1e6

print(f"Source file: {SRC_SIZE:.1f} MB\n")
print(f"{'Config':<40} {'Size (MB)':>10} {'Ratio':>8} {'Time (s)':>9}")
print("-" * 72)

results = []

# ── helpers ──────────────────────────────────────────────────────────────────

def _enc_for_var(da, zlib=True, complevel=4, chunksizes=None, dtype=None,
                 scale_factor=None, add_offset=None):
    """Build an encoding dict for a single DataArray."""
    enc = {"zlib": zlib, "complevel": complevel, "shuffle": True}
    if chunksizes is not None:
        enc["chunksizes"] = chunksizes
    if dtype is not None:
        enc["dtype"] = dtype
    if scale_factor is not None:
        enc["scale_factor"] = scale_factor
        enc["add_offset"] = add_offset if add_offset is not None else 0.0
        enc["_FillValue"] = np.iinfo(np.int16).min  # -32768
    return enc


def build_encoding(ds, complevel=4, chunk_xy=50, chunk_depth=17,
                   quantize=False):
    """Generate per-variable encoding dicts."""
    encoding = {}

    # Which vars are (time, depth, y, x)?  Which are (time, y, x)?
    vars_4d = [v for v in ds.data_vars if ds[v].ndim == 4]
    vars_3d = [v for v in ds.data_vars if ds[v].ndim == 3]
    vars_2d = [v for v in ds.data_vars if ds[v].ndim == 2]
    vars_scalar = [v for v in ds.data_vars if ds[v].ndim <= 1]

    for v in vars_4d:
        chunks = (1, chunk_depth, chunk_xy, chunk_xy)
        if quantize and ds[v].dtype == np.float32:
            # Estimate range ignoring NaN
            arr = ds[v].values
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            # int16 range (excluding fill): -32767 .. 32767
            sf = (vmax - vmin) / 65534.0
            offset = vmin + sf * 32767  # centre
            encoding[v] = _enc_for_var(ds[v], complevel=complevel,
                                        chunksizes=chunks,
                                        dtype="int16",
                                        scale_factor=sf,
                                        add_offset=offset)
        else:
            encoding[v] = _enc_for_var(ds[v], complevel=complevel,
                                        chunksizes=chunks)

    for v in vars_3d:
        chunks = (1, chunk_xy, chunk_xy)
        encoding[v] = _enc_for_var(ds[v], complevel=complevel,
                                    chunksizes=chunks)

    for v in vars_2d:
        chunks = (chunk_xy, chunk_xy)
        encoding[v] = _enc_for_var(ds[v], complevel=complevel,
                                    chunksizes=chunks)

    for v in vars_scalar:
        encoding[v] = {"zlib": False}

    return encoding


def test_config(label, out_name, ds, encoding):
    out_path = os.path.join(OUT_DIR, out_name)
    t0 = time.time()
    ds.to_netcdf(out_path, encoding=encoding)
    elapsed = time.time() - t0
    size = os.path.getsize(out_path) / 1e6
    ratio = SRC_SIZE / size
    print(f"{label:<40} {size:>10.1f} {ratio:>8.2f}x {elapsed:>9.1f}s")
    results.append((label, size, ratio, out_path))
    return out_path


# ── load source once ──────────────────────────────────────────────────────────
ds = xr.open_dataset(SRC)

# ── Test 1: int16 quantization + zlib=4, original chunks ─────────────────────
enc1 = build_encoding(ds, complevel=4, chunk_xy=50, chunk_depth=17, quantize=True)
test_config("int16 quant + zlib=4, chunks [1,17,50,50]", "compressed_reconst_test_1.nc", ds, enc1)

# ── Test 2: int16 quantization + zlib=4, larger spatial chunks ───────────────
enc2 = build_encoding(ds, complevel=4, chunk_xy=100, chunk_depth=17, quantize=True)
test_config("int16 quant + zlib=4, chunks [1,17,100,100]", "compressed_reconst_test_2.nc", ds, enc2)

ds.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
best = min(results, key=lambda x: x[1])
print(f"Best: {best[0]}  →  {best[1]:.1f} MB  ({best[2]:.2f}x smaller)")
print(f"      {best[3]}")
