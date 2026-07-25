"""GPU sanity check — runs inside trainer-gpu container."""
import sys
try:
    import torch
except ImportError:
    print("torch       : NOT INSTALLED")
    sys.exit(1)

print(f"torch         : {torch.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"capability    : sm_{cap[0]}{cap[1]}")
    print(f"device_name   : {torch.cuda.get_device_name(0)}")
    print(f"device_count  : {torch.cuda.device_count()}")
    # tiny tensor op to confirm kernels actually load
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print(f"matmul OK     : c.shape={tuple(c.shape)}  c.device={c.device}")
    print(f"mem_allocated : {torch.cuda.memory_allocated()/1e6:.1f} MB")
else:
    print("CUDA not available")
    sys.exit(2)
print("OK")
