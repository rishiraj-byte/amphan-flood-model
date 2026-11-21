import sys

try:
    import torch
except ImportError:
    sys.exit("PyTorch not installed: run setup_laptop.bat")
print("python ", sys.version.split()[0])
print("torch  ", torch.__version__, "| CUDA build:", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU    {p.name}, {p.total_memory / 2**30:.1f} GB VRAM  -> OK")
    x = torch.randn(2048, 2048, device="cuda")
    print("matmul test:", float((x @ x).mean()) is not None)
    if p.total_memory < 3.5 * 2**30:
        print("small GPU: use  --batch 4 --base 24")
else:
    print("NO CUDA GPU visible to PyTorch.")
    print(" - NVIDIA laptop: update the driver, then reinstall torch with the CUDA command from")
    print("   https://pytorch.org/get-started/locally/  (pip, Windows, CUDA)")
    print(" - AMD / Intel GPU: CUDA is not available; training falls back to CPU (slow)")
