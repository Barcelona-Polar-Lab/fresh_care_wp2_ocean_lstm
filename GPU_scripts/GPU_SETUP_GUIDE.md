# GPU Troubleshooting Guide

## Step 1: Run Diagnostics

On the remote server, run:
```bash
bash diagnose_gpu.sh
```

## Common Issues and Fixes

### Issue 1: NVIDIA Drivers Not Installed

**Symptoms:** `nvidia-smi` command not found

**Fix:**
```bash
# Check Ubuntu version
lsb_release -a

# For Ubuntu 22.04, install recommended driver
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# Or install specific driver version
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Reboot required
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi
```

### Issue 2: NVIDIA Kernel Module Not Loaded

**Symptoms:** `lsmod | grep nvidia` shows nothing, or `/dev/nvidia*` devices missing

**Fix:**
```bash
# Load NVIDIA modules manually
sudo modprobe nvidia
sudo modprobe nvidia-uvm

# If that fails, rebuild modules
sudo apt install dkms
sudo dkms install -m nvidia -v $(modinfo nvidia | grep ^version | awk '{print $2}')
```

### Issue 3: CUDA Toolkit Not Installed (Optional)

**Note:** PyTorch includes its own CUDA libraries, so this is usually not required.

If you need the toolkit:
```bash
# For CUDA 12.8 (matching your PyTorch)
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.15_linux.run
sudo sh cuda_12.8.0_550.54.15_linux.run
```

### Issue 4: Permission Issues

**Symptoms:** GPU detected but can't access devices

**Fix:**
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Check device permissions
ls -la /dev/nvidia*

# Log out and back in for group changes to take effect
```

### Issue 5: Wrong PyTorch Version

**Symptoms:** PyTorch installed without CUDA support

**Fix:**
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA support (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Or for CUDA 12.1 (more compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Quick Verification After Fixes

```bash
# Check driver
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run your test
python3 gpu_test.py
```

## Most Likely Solution

For Ubuntu 22.04 with GPU but no drivers, the most common fix is:

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

Then verify with:
```bash
nvidia-smi
python3 gpu_test.py
```
