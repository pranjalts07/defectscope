import torch


def get_device() -> torch.device:
    """
    Returns the best available device: CUDA > MPS (Apple Silicon) > CPU.
    MPS is the Metal Performance Shaders backend — works on M1/M2/M3 Macs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
