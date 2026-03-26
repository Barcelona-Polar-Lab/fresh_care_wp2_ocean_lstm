## Plan: Arctic LSTM Reconstruction Pipeline

**TL;DR**: Create a memory-efficient pipeline that transforms gridded input data (350×350×102) to profile format, runs MC Dropout predictions in ~5k profile chunks, and reconstructs Arctic T/S/SH fields. Will create `arctic_reconstruction.py` and `lstm_pytorch_utils.py` (shared utilities). Processing ~40-60k ocean pixels per timestep × 12 months.

**Key insight**: The model takes **surface anomalies as input features** (satellite - GLORYS surface) and **outputs predicted profile anomalies**. For prediction steps (A-D), we only need GLORYS surface values (depth=0). Full GLORYS 3D fields are only loaded in the final reconstruction step (E), optimizing memory usage.

**Steps**

1. **Create [lstm_pytorch_utils.py](lstm_pytorch_utils.py)** - Extract shared utilities from [lstm_pytorch_pd_mcdo.py](lstm_pytorch_pd_mcdo.py):
   - `OceanLSTM` class (lines 100-240) - model architecture
   - `normalize_data()` / `denormalize_data()` (lines 1792-1831) - normalization helpers
   - `load_model_checkpoint()` - new function to load model + norm_params
   - `mc_dropout_predict()` - new function encapsulating MC Dropout inference logic (adapted from lines 765-890)
   - Keep imports minimal; import these utilities back into the training script

2. **Create [arctic_reconstruction/arctic_reconstruction.py](arctic_reconstruction/arctic_reconstruction.py)** with configurable CLI args:
   - `--input_dir` (default: `/home/nico/Desktop/AUX_DIR_FRESH_CARE/model_input/`)
   - `--output_dir` (default: `./arctic_reconstruction/output/`)
   - `--model_path` (default: `model_LSTM_40_40_sat_znorm/model.pth`)
   - `--chunk_size` (default: 5000 profiles, tuned for <8GB RAM)
   - `--n_mc_samples` (default: 50)

3. **Step A - Grid to profiles** (prepare model INPUT features): For each input file `model_input_YYYY_MM.nc`:
   - Load `ocean_mask` → get valid (y, x) indices where mask==1
   - Extract surface data: `SST`, `SSS`, `ADT` (satellite), `X_EASE`, `Y_EASE`, `DOY`
   - Extract GLORYS **surface only** (depth=0): `T_glorys[0,:,:]`, `S_glorys[0,:,:]`, `SH_glorys[0,:,:]`
   - Compute **input features** (surface anomalies): `sst_input = SST - T_glorys_surface`, etc.
   - Compute seasonal features: `cos(2π×DOY/365+1)`, `sin(2π×DOY/365+1)`
   - Reshape to `(n_ocean_pixels, n_depths, 7)` where 7 input features = [sst_input, sss_input, adt_input, x_ease, y_ease, cos, sin]
   - Store pixel indices `(y_idx, x_idx)` for regridding later
   - **Memory note**: Do NOT load full GLORYS 3D arrays here - only surface slices needed

4. **Step B - Chunked MC Dropout predictions** (model OUTPUT = predicted profile anomalies):
   - Split ocean profiles into chunks of 5000
   - For each chunk:
     - Apply z-score normalization using `norm_params` from checkpoint
     - Run 50 MC Dropout forward passes (model in `train()` mode)
     - Model outputs **predicted anomaly profiles**: `T_anom_pred`, `S_anom_pred`, `SH_anom_pred` at all depths
     - Compute mean, std, CI bounds (2.5%, 97.5%) across MC samples
     - Denormalize predictions
   - Use `tqdm` progress bars (outer: chunks, inner: MC samples)

5. **Step C - Save predicted anomaly profiles** to `output_dir/predicted_anom_prof/`:
   - Filename: `anom_profiles_YYYYMMDD.nc`
   - Variables: `SH_anom_pred`, `T_anom_pred`, `S_anom_pred` (means), `*_std` (uncertainty), `*_ci_lower`, `*_ci_upper`
   - Coords: `(profile, depth)` + metadata: `x_idx`, `y_idx`, `x_ease_val`, `y_ease_val`, `time`
   - Use chunked encoding for depth dimension

6. **Step D - Regrid predicted anomalies to grid format** to `output_dir/predicted_anom_grid/`:
   - Map profiles back to grid using stored `(y_idx, x_idx)` indices
   - Output shape: `(time, depth, y_ease, x_ease)` matching input file structure
   - Initialize with NaN, fill valid ocean locations with predicted anomalies
   - Filename: `anom_grid_YYYYMMDD.nc`
   - Chunked encoding: `{'depth': 17, 'y_ease': 50, 'x_ease': 50}`

7. **Step E - Final reconstruction** to `output_dir/reconstruction_data/`:
   - **Now** load full GLORYS 3D fields: `T_glorys`, `S_glorys`, `SH_glorys` (all depths)
   - Apply NaN mask: where GLORYS is NaN, set predicted anomalies to NaN
   - Reconstruct full profiles: `T = T_anom_pred + T_glorys`, `S = S_anom_pred + S_glorys`, `SH = SH_anom_pred + SH_glorys`
   - Add all variables to output dataset (surface data, GLORYS climatology, predicted anomalies, reconstructed profiles)
   - Filename: `reconstruction_YYYYMMDD.nc`

8. **Update [lstm_pytorch_pd_mcdo.py](lstm_pytorch_pd_mcdo.py)** with minimal changes:
   - Add import: `from lstm_pytorch_utils import OceanLSTM, normalize_data, denormalize_data`
   - Remove duplicated class/function definitions
   - Keep existing test logic intact using imported utilities

**Verification**
- Run on a single timestep first: `python arctic_reconstruction.py --input_dir ... --chunk_size 5000`
- Check output shapes: profile files should have ~40-60k profiles, grid files 350×350
- Verify reconstruction by opening one output and plotting a depth slice
- Compare memory usage with `htop` during execution
- Run existing test suite to verify `lstm_pytorch_pd_mcdo.py` still works after refactor

**Decisions**
- Chunk size: 5000 profiles (vs 10000) to fit in <8GB RAM with MC Dropout overhead
- Utility file: Shared `lstm_pytorch_utils.py` to avoid code duplication
- Output structure: 3 subdirectories (profiles → grid → reconstruction) for step-by-step debugging
- File naming: `YYYYMMDD` timestamp from mid-month date in input files
