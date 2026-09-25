# diffusion-models-course 專案指南

> 供接手的 agent／開發者快速上手：架構、模組職責、怎麼執行、怎麼驗證。
> Last Updated: 2026-09-25

## 產品概述

**高中生版**：一門教你「電腦怎麼從一片雜訊畫出圖」的課。每一課先用白話講直覺，再推數學，最後親手寫程式在筆電上跑出結果。

**專業版**：擴散模型的自學課程（繁體中文），從 DDPM 的推導到 score／SDE、DDIM、classifier-free guidance、flow matching、DiT，並附 2026 年前沿地圖。
所有演算法在 `src/diffusion_course/` 有經過測試的參考實作，每個實驗 notebook 直接使用它。

## 目錄結構

```
diffusion-models-course/
├── lessons/          # 講義 00–11 ＋ 附錄 A（繁中 Markdown）
├── notebooks/        # 已執行的實驗 02–10（輸出是實跑紀錄）
├── src/diffusion_course/
│   ├── schedules.py      # NoiseSchedule：β、α、ᾱ、後驗變異數、SNR
│   ├── ddpm.py           # q_sample、ddpm_loss、predict_x0、posterior_mean、ddpm_sample
│   ├── ddim.py           # ddim_timesteps、ddim_sample（eta 控制隨機性）
│   ├── score.py          # eps_to_score、langevin_dynamics、annealed_langevin_dynamics
│   ├── flow_matching.py  # interpolate、flow_matching_loss、flow_sample（Euler／Heun）
│   ├── guidance.py       # drop_labels、ClassifierFreeGuidance
│   ├── models/           # ToyMLP、UNet、TinyDiT、embeddings；build_model
│   ├── data.py           # 2D 玩具資料、GaussianMixture（精確 score）、MNIST 載入
│   ├── training.py       # train（AdamW＋clip＋EMA）、save/load_checkpoint
│   ├── evaluation.py     # DigitClassifier、get_digit_classifier（快取）、judge_samples
│   ├── viz.py            # 畫點、軌跡、向量場、影像格、loss
│   └── utils.py          # PROJECT_ROOT、CHECKPOINT_DIR、get_device、set_seed、notebook_logging
├── scripts/          # train_mnist.py、sample_mnist.py（GPU 長訓練用）
├── tests/            # pytest；多數測試拿封閉解對答案
├── references.md     # 參考資料（查證日期、一手／二手）
├── NOTICE.md         # 第三方授權與標示
└── ai-collab/        # 本資料夾
```

被 git 忽略、執行時才產生：`data/`（MNIST）、`checkpoints/`（實驗存的權重）、`outputs/`（scripts 的圖）。

## 核心設計

**統一的模型介面** `model(x, t, y=None)`：

- `t` 一律是 $[0, 1)$ 的浮點數。DDPM 的整數步 $i$ 由 `NoiseSchedule.timestep_input` 換成 $i/T$；flow matching 直接用連續時間。
- `y=None` 表示無條件；有類別的模型會把 `None` 對應到保留的「空標籤」（索引 = `num_classes`），這就是 CFG 的無條件分支。
- 因為介面一致，`ClassifierFreeGuidance` 是一層包裝，所有取樣器都能直接用。

**時間方向**：DDPM 相關程式用 DDPM 慣例（$x_0$ 是資料）；`flow_matching.py` 用「$t=0$ 雜訊、$t=1$ 資料」，變數命名為 `noise`／`data`。

**零初始化輸出層**：U-Net 的 head 與 DiT 的 adaLN／最終投影都初始化為 0（測試會檢查），模型一開始輸出 0。

## 執行

```bash
cd diffusion-models-course
uv add torch torchvision numpy matplotlib loguru ipykernel   # 一次性，由本人執行
uv add --dev pytest
uv run pytest                                               # 50 個測試
```

notebook 用本子專案的 `.venv` 當 kernel。**Lab 07 需要 Lab 06 存下的權重**（`checkpoints/lab06_mnist_ddpm_unet.pt`）；評估用分類器第一次使用時訓練並快取到 `checkpoints/digit_classifier.pt`。

雲端 4 核心 CPU 的實測執行時間（2026-09-25）：

| notebook | 時間 | 備註 |
|---|---|---|
| 02、03 | < 1 分鐘 | 不訓練（03 會下載 MNIST） |
| 04、05 | 約 2 分鐘 | 2D MLP，5000 步 |
| 06 | 約 15–20 分鐘 | U-Net 16 通道，2000 步 |
| 07 | 約 5 分鐘 | 載入 06 的模型 |
| 08 | 約 15–20 分鐘 | 條件 U-Net，2000 步 |
| 09 | 約 5 分鐘 | 2D，含 reflow |
| 10 | 約 15 分鐘 | TinyDiT，3000 步 |

## 測試策略

- 演算法測試盡量用**封閉解**：單點資料的最佳去噪器／速度場（`tests/oracles.py`）必須讓取樣器精確回到那個點；後驗均值與 20 萬樣本的線性迴歸比對；常態混合的 score 與 autograd 比對；Langevin 對標準常態的穩態。
- 模型測試：輸出形狀、零初始化、`y=None` 等同空標籤、DiT 的 unpatchify 是 patchify 排列的逆。
- `ruff`：本子專案的 `pyproject.toml` 排除 `notebooks/`（與 repo 根的 pre-commit 一致）。

## 修改 notebook 的注意事項

notebook 的輸出是實跑紀錄，改了程式碼就要**整本重新執行**再存檔，避免輸出與程式不一致。輸出中不應出現本機絕對路徑。
