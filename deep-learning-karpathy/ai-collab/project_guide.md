# Deep Learning Karpathy 專案指南

> 供新進 Agent/開發者快速入手
> Last Updated: 2026-03-28

---

## 產品概述

### 高中生版解釋
想像你想教一台電腦「讀懂」人類的文字並且自己寫出文章。這個專案就是帶你一步步理解這件事是怎麼做到的：
1. **Tokenizer（分詞器）**：教電腦把文字切成小塊（像拼圖碎片）
2. **GPT（語言模型）**：教電腦學習這些碎片之間的關係，然後自己組出新的文章

### 專業術語版
**產品類型**：Deep Learning Tutorial / Educational Codebase

**學習路徑**：
| 順序 | 主題 | 來源 | 核心概念 |
|------|------|------|----------|
| 1 | GPT Tokenizer | Karpathy - minBPE | BPE 演算法、byte-level tokenization、regex splitting |
| 2 | nanoGPT | Karpathy - nanoGPT | Transformer 架構、self-attention、GPT-2 訓練 |

**為什麼這個順序？**
- Tokenizer 是 LLM 的「前處理」步驟，必須先理解
- 理解 tokenizer 後，才能理解 GPT 模型的 vocab_size、embedding 等概念

---

## 技術棧

| 項目 | 技術 |
|-----|------|
| Language | Python 3.10+ |
| Package Manager | uv |
| Deep Learning | PyTorch 2.0+ |
| Tokenizer | tiktoken (GPT-4 comparison) |
| Pretrained Models | HuggingFace Transformers |
| Notebooks | Jupyter |
| Visualization | Matplotlib |

---

## Codebase 結構

```
deep-learning-karpathy/
├── tutorials/                    # 教材文件 (繁體中文)
│   ├── 01_tokenizer/             # Part 1: GPT Tokenizer (minBPE)
│   │   ├── 01_bpe_basics.md      # BPE 演算法基礎
│   │   ├── 02_basic_tokenizer.md # BasicTokenizer 實作
│   │   ├── 03_regex_tokenizer.md # RegexTokenizer + GPT-4
│   │   └── 04_exercises.md       # 練習題
│   └── 02_nanogpt/               # Part 2: nanoGPT
│       ├── 01_transformer_basics.md   # Transformer 基礎
│       ├── 02_gpt_architecture.md     # GPT 模型架構
│       ├── 03_training_shakespeare.md # 訓練莎士比亞文本
│       └── 04_exercises.md            # 練習題
├── scripts/                      # Python 工具腳本 (uv 管理)
│   ├── train_tokenizer.py        # 訓練自己的 tokenizer
│   ├── compare_tokenizers.py     # 比較不同 tokenizer
│   ├── train_nanogpt.py          # 訓練 nanoGPT
│   └── sample_text.py            # 用訓練好的模型生成文字
├── notebooks/                    # Jupyter 互動式教材
├── references/                   # 參考 repo (git clone)
│   ├── minbpe/                   # karpathy/minbpe
│   └── nanoGPT/                  # karpathy/nanoGPT
├── ai-collab/                    # AI 協作文件
├── pyproject.toml                # Python 依賴定義 (uv)
└── README.md                     # 專案說明
```

---

## 開發與測試指南

### 環境需求
- Python 3.10+
- uv (Python package manager)
- GPU (可選，nanoGPT 訓練會更快；CPU 也能跑)

### 快速開始

```bash
# 1. 安裝依賴
cd deep-learning-karpathy/
uv sync

# 2. 執行範例腳本
uv run python scripts/train_tokenizer.py      # 訓練 tokenizer
uv run python scripts/compare_tokenizers.py   # 比較 tokenizer
uv run python scripts/train_nanogpt.py        # 訓練 nanoGPT (需要 GPU 或耐心)
uv run python scripts/sample_text.py          # 生成文字

# 3. 啟動 Jupyter Notebook
uv run jupyter notebook notebooks/
```

### 常用指令

```bash
# Python 套件管理
uv sync                              # 安裝依賴
uv add <package>                     # 新增套件
uv run python <script>               # 執行腳本
uv run pytest                        # 執行測試

# Jupyter
uv run jupyter notebook              # 啟動 notebook server
```
