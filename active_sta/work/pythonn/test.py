import torch

if torch.cuda.is_available():
    print(f"CUDA is available! GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA is not available.")
