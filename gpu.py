import torch

print("Allocated VRAM (GB):", torch.cuda.memory_allocated()/1024**3)
print("Reserved VRAM (GB):", torch.cuda.memory_reserved()/1024**3)
print("CUDA available:", torch.cuda.is_available())
print("Current GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
