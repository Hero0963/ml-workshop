# Roadmap — diffusion-models-course

> **新 session 第一站**：現況、下一步、已定案不要再重開的決策。
> 做完一件事就更新本檔（現況＋下一步），細節寫進 [`dev_log.md`](dev_log.md)。架構與執行方式見 [`project_guide.md`](project_guide.md)。
> Last Updated: 2026-09-25

## 現況（2026-09-25）

- **課程 v1 完成**：12 課講義（`lessons/00`–`11` ＋ 附錄 A）、9 個已執行的實驗 notebook（`notebooks/02`–`10`）、參考實作 `src/diffusion_course/`、參考資料 `references.md`、授權說明 `NOTICE.md`。
- 所有 notebook 都在**雲端 4 核心 CPU**上實跑過（CPU 版 torch 2.14.0），輸出就是那次執行的紀錄；MNIST 實驗為了能在 CPU 跑完，模型與步數都刻意縮小。
- 本人的機器上**還沒有建過環境**：`pyproject.toml` 的 `dependencies` 是空的，等本人 `uv add`（見「下一步」第 1 項）。

## 下一步

1. **本人建立基線** — `cd diffusion-models-course && uv add torch torchvision numpy matplotlib loguru ipykernel && uv add --dev pytest && uv run pytest`；指令已在雲端用 `uv add --no-sync` 驗證會解析到 `torch 2.14.0+cu126`（含 Windows wheel）。
2. **有 GPU 時重跑 MNIST 實驗** — 把 Lab 06／08／10 開頭的步數與寬度調大（notebook 裡有建議值），或用 `scripts/train_mnist.py` 長訓練，比較品質。
3. **補練習題參考解答** — 目前每課只有題目；可放在 `lessons/solutions/`。
4. **第 11 課保鮮** — 前沿地圖約半年重查一次（照 `AGENTS.md` §4 的 survey 保鮮紀律），並更新查證日期。
5. **決定授權** — 子專案尚未放 `LICENSE`；要不要比照 `thread-the-grid`（Apache-2.0）由本人決定。

## 已定案（不要再重開）

| 日期 | 決策 | 理由 |
|---|---|---|
| 2026-09-25 | 講義用 Markdown（繁中）＋ 已執行的 notebook；程式與註解用英文 | 與 repo 其他教材一致（`rules.md`） |
| 2026-09-25 | 所有模型共用 `model(x, t, y)` 介面，$t \in [0, 1)$ | DDPM／DDIM／flow matching／CFG 可以任意組合；講義也依此解說 |
| 2026-09-25 | flow matching 一律用「$t=0$ 雜訊、$t=1$ 資料」，程式變數叫 `noise`／`data` | 文獻時間方向不一致（第 09 課 §2.7），程式碼避免 $x_0$／$x_1$ 歧義 |
| 2026-09-25 | 實驗以 CPU 能跑完為準：U-Net `base_channels=16`、2000 步；DiT 3000 步 | 4 核心 CPU 上 U-Net 32 通道一步 1.2 秒，16 通道 0.47 秒；DiT（49 token）0.28 秒 |
| 2026-09-25 | 相依套件由本人 `uv add`；`pyproject.toml` 預先把 `torch`／`torchvision` 導向 cu126 explicit index | repo 規則 `AGENTS.md` §5；cu126 同時有 Linux 與 Windows 的 torch 2.14.0 wheel |
| 2026-09-25 | 程式依論文公式自寫，不複製任何官方實作；圖全部由 notebook 產生 | 著作權與授權（部分官方實作是 CC BY-NC）；見 `NOTICE.md` |
| 2026-09-25 | MNIST 授權依較嚴格的 CC BY-SA 3.0 處理 | 官方頁面當天連不上，二手來源說 CC BY-SA 3.0、HF 資料卡寫 MIT，取嚴格者 |
| 2026-09-25 | 評估用「1 分鐘訓練的數字分類器」而非 FID | 28×28 數字上 FID 意義有限；分類器可量 accuracy（有條件時）、confidence、class entropy |
