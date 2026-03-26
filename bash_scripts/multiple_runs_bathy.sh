#!/bin/bash
# Multiple runs to test bathymetry and GLORYS surface anomalies
#
# INPUT_VAR_ORDER (12 variables):
#   0: sst_anomaly, 1: sss_anomaly, 2: sst_glorys_anomaly, 3: sss_glorys_anomaly,
#   4: adt, 5: seasonal_cos, 6: seasonal_sin,
#   7: latitude, 8: longitude, 9: x_ease, 10: y_ease, 11: bathymetry
#
# Default binary: 110011100110 (sst, sss, adt, cos, sin, x_ease, y_ease)

PYTHON="/home/FRESH-CARE/Codes/python_gen_venv/bin/python"
SCRIPT="/home/FRESH-CARE/Codes/WP2_lstm/lstm_pytorch_pd_mcdo.py"

echo "=============================================="
echo "Run 1: Default + Bathymetry"
echo "Binary: 110011100111"
echo "=============================================="
$PYTHON $SCRIPT --input_vars 110011100111 --model_dir /data/FRESH-CARE/data_for_LSTM/models/model_LSTM_40_40_sat_znorm_bathy

echo ""
echo "=============================================="
echo "Run 2: GLORYS surface anomalies (no satellite SST/SSS, no bathymetry)"
echo "Binary: 001111100110"
echo "=============================================="
$PYTHON $SCRIPT --input_vars 001111100110 --model_dir /data/FRESH-CARE/data_for_LSTM/models/model_LSTM_40_40_glor_znorm

echo ""
echo "All runs completed!"


