# src/core/rl/verify_torch.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
一個用來驗證 PyTorch 安裝狀態並檢查 GPU (CUDA) 支援的腳本。
執行方式：
在你的虛擬環境中，運行 `python verify_torch.py`
"""

import sys
import torch


def main():
    """主執行函數"""
    print("--- PyTorch 安裝環境檢測 ---")
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print("-" * 35)

    # 核心檢測：CUDA 是否可用
    is_cuda_available = torch.cuda.is_available()

    if is_cuda_available:
        print("✅ 恭喜！你的 PyTorch 已成功啟用 GPU 加速。")
        print("-" * 35)
        # 獲取並印出詳細的 GPU 資訊
        print(f"PyTorch 編譯時使用的 CUDA 版本: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"偵測到的 GPU 數量: {gpu_count}")
        for i in range(gpu_count):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️ 注意：你的 PyTorch 目前只能使用 CPU 進行運算。")
        print("-" * 35)
        print("這可能由以下原因造成：")
        print("  1. 你安裝到的是 PyTorch 的 CPU-only 版本。")
        print("  2. NVIDIA 驅動程式未正確安裝或版本過舊。")
        print("  3. 系統的 CUDA Toolkit 版本與 PyTorch 編譯版本不匹配。")

        # 檢查版本號是否包含 cpu 關鍵字
        if "cpu" in torch.__version__:
            print(
                "\n提示：你的 PyTorch 版本號中包含 'cpu'，這代表你安裝的是純 CPU 版本。"
            )
            print("請參考 PyTorch 官網的指令，使用 `extra-index-url` 來安裝 GPU 版本。")

    print("-" * 35)


if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("❌ 錯誤：找不到 PyTorch 模組。")
        print("請先在目前的虛擬環境中安裝 PyTorch。")
