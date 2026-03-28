# Agent 交接用

> Last Updated: 2026-03-28

---

## 當前狀態

- **分支**：`deep-learning-tutorial`
- **最新進度**：教材撰寫完成 + 工具腳本測試通過
- **階段**：Phase 1 完成

---

## 專案概述

這是一個深度學習教材專案，基於 Andrej Karpathy 的教學影片系列。
涵蓋兩大主題：GPT Tokenizer (minBPE) 和 nanoGPT。

**教材特色**：
1. 每個概念都有「高中生版」和「專業版」雙層講解
2. 附帶可執行的 Python 程式碼範例
3. 循序漸進，從基礎到進階

---

## 技術棧

| 項目 | 版本/位置 |
|------|----------|
| Python | 3.9+ |
| Package Manager | uv |
| PyTorch | 2.0+ |
| 參考 repo | references/minbpe, references/nanoGPT |

---

## 已完成

- [x] 建立 `deep-learning-tutorial` 分支
- [x] 建立專案資料夾結構
- [x] uv 初始化 + 依賴設定
- [x] Git clone minbpe 和 nanoGPT 參考 repo
- [x] 建立 ai-collab 文件架構
- [x] 撰寫 Tokenizer 教材 (4 篇)
- [x] 撰寫 nanoGPT 教材 (4 篇)
- [x] 建立 Python 工具腳本 (4 個)
- [x] 測試 train_tokenizer.py 和 compare_tokenizers.py

---

## TODO List

### P1 (重要)
- [ ] 測試 train_nanogpt.py (需要 GPU 或耐心等 CPU 訓練)
- [ ] Jupyter Notebook 互動式教材

### P2 (優化)
- [ ] 更多練習題與解答
- [ ] GPU 訓練優化指南
- [ ] 進階主題：attention visualization、model surgery

---

## Codebase 結構

```
deep-learning-karpathy/
├── tutorials/
│   ├── 01_tokenizer/           # Part 1: GPT Tokenizer (4 篇)
│   └── 02_nanogpt/             # Part 2: nanoGPT (4 篇)
├── scripts/                    # Python 工具 (4 個已測試)
│   ├── train_tokenizer.py
│   ├── compare_tokenizers.py
│   ├── train_nanogpt.py
│   └── sample_text.py
├── references/                 # 參考 repos
│   ├── minbpe/
│   └── nanoGPT/
├── ai-collab/
├── pyproject.toml
└── README.md
```

---

## 常用指令

```bash
cd deep-learning-karpathy/ && uv sync

uv run python scripts/train_tokenizer.py
uv run python scripts/compare_tokenizers.py
uv run python scripts/train_nanogpt.py
uv run python scripts/sample_text.py
```
