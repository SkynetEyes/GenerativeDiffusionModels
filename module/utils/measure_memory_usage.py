import torch
import psutil


def measure_memory_usage(tag=""):
    """Print current RAM and VRAM usage."""
    process = psutil.Process()
    ram_used = process.memory_info().rss / 1024**3
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / 1024**3
        vram_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{tag}] RAM used: {ram_used:.2f} GB | VRAM allocated: {vram_alloc:.2f} GB | VRAM reserved: {vram_reserved:.2f} GB")
    else:
        print(f"[{tag}] RAM used: {ram_used:.2f} GB | VRAM not available")