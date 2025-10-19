#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
gpu_check.py
Verifies CUDA GPU availability and VRAM capacity on Sparx.
Logs detailed diagnostics with emojis and raises EnvironmentError if unmet.
"""

import torch
import logging
import datetime

# ---------------------------------------------------------------------
# Configure logging
# ---------------------------------------------------------------------
LOG_PATH = "/home/craigtrim/projects/alias-label-retriever/logs/gpu_check.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("gpu_check")

# ---------------------------------------------------------------------
# Verify GPU availability
# ---------------------------------------------------------------------
def verify_gpu(min_vram_gb: int = 10) -> str:
    logger.info("🔍 Starting GPU environment verification...")

    if not torch.cuda.is_available():
        details = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda,
        }
        logger.error("❌ CUDA device not detected. Full diagnostic below:")
        for k, v in details.items():
            logger.error(f"   • {k}: {v}")

        raise EnvironmentError(
            "🚨 No CUDA GPU detected. This process requires a local GPU.\n"
            "Please verify NVIDIA drivers, CUDA toolkit, and PyTorch installation."
        )

    device_count = torch.cuda.device_count()
    logger.info(f"🧮 Detected {device_count} CUDA device(s).")

    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        pct = (free_gb / total_gb) * 100

        logger.info(f"✅ GPU[{i}] {name}")
        logger.info(f"   • VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total ({pct:.0f}% free)")

        if free_gb < min_vram_gb:
            raise EnvironmentError(
                f"⚠️ Insufficient free GPU memory: {free_gb:.1f} GB "
                f"(minimum required: {min_vram_gb} GB). "
                "Close other processes or switch to a higher-memory GPU."
            )

    logger.info("🎯 GPU environment verified successfully ✅")
    logger.info(f"🕒 Verification complete at {datetime.datetime.now().isoformat()}")
    return "cuda"


if __name__ == "__main__":
    device = verify_gpu()
    print(f"✅ Ready to train on device: {device}")
