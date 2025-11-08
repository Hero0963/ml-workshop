# src/stt/check_gpu.py
import torch
from loguru import logger
import sys

# Configure loguru to show only INFO level and above
logger.remove()
logger.add(sys.stderr, level="INFO")


def check_gpu_availability():
    """
    Checks for PyTorch and CUDA availability and prints the status using loguru.
    """
    logger.info(f"PyTorch version: {torch.__version__}")

    is_available = torch.cuda.is_available()

    if is_available:
        logger.success("✅ CUDA is available. PyTorch can use the GPU.")
        device_count = torch.cuda.device_count()
        logger.info(f"Found {device_count} CUDA-enabled device(s).")
        for i in range(device_count):
            logger.info(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.error("❌ CUDA is not available. PyTorch will run on the CPU.")
        logger.warning(
            "If you have an NVIDIA GPU, please check your CUDA drivers and PyTorch installation."
        )


if __name__ == "__main__":
    check_gpu_availability()
