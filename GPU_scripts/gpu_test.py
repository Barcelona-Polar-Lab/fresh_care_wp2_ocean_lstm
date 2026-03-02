import torch
import time

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    
    # Test actual GPU computation
    print("\nTesting GPU computation...")
    device = torch.device('cuda')
    
    # Create large tensors on GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    start_time = time.time()
    for i in range(100):
        z = torch.mm(x, y)
    torch.cuda.synchronize()  # Wait for GPU operations to complete
    gpu_time = time.time() - start_time
    
    print(f"GPU memory after computation: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    print(f"GPU computation time: {gpu_time:.3f}s")
    
    # Compare with CPU
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    
    start_time = time.time()
    for i in range(100):
        z_cpu = torch.mm(x_cpu, y_cpu)
    cpu_time = time.time() - start_time
    
    print(f"CPU computation time: {cpu_time:.3f}s")
    print(f"GPU speedup: {cpu_time/gpu_time:.1f}x")
else:
    print("CUDA not available!")
