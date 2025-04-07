import os
import sys
import platform
import torch
import torch.version
from HelloLM.utils.logger import logger
from importlib.metadata import version


@logger.catch
def ensure_directory(path: str):
    resolved = path
    if not os.path.isabs(path):
        resolved = os.path.join(os.getcwd(), path)

    if not os.path.exists(path):
        os.makedirs(resolved, exist_ok=True)

@logger.catch
def to_abs_path(path: str):
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    return path

@logger.catch
def log_env_metadata(debug=False):
    log = logger.info
    if debug:
        log = logger.debug

    log("--- System Information ---")
    log(f"OS: {platform.system()} {platform.release()}")
    log(f"Architecture: {platform.machine()}")

    log("--- Python Information ---")
    log(f"Python version: {sys.version}")
    log(f"Executable path: {sys.executable}")

    log("--- Package Version ---")
    log(f"PyTorch version: {version('torch')}")
    log(f"Tiktoken version: {version('tiktoken')}")
    log(f"Plotly version: {version('plotly')}")

    log("--- Runtime Devices ---")

    if torch.cuda.is_available():
        log("NVIDIA CUDA is available")
        log(f"-- Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log(f"     Device {i}: {torch.cuda.get_device_name(i)}")
        log(f'-- CUDA runtime version: {torch.version.cuda}')
        log(f'Total memory: {torch.cuda.memory_summary}')
    
    if torch.mps.is_available():
        log("macOS MPS is available")
        log(f"-- Number of MPS devices: {torch.mps.device_count()}")
        log(f'-- Recommended allocatable memory {torch.mps.recommended_max_memory() / (1024 ** 3)} GB')

    if torch.cpu.is_available():
        log("CPU calculated is available")
        log(f'-- Number of CPU devices: {torch.cpu.device_count()}')
