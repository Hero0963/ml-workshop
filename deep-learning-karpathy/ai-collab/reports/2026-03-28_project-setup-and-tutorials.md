# 任務報告：Deep Learning Karpathy 專案建立與教材撰寫

> 日期：2026-03-28
> 分支：`deep-learning-tutorial`
> 狀態：Phase 1 完成

---

## 任務摘要

建立了一個完整的深度學習教材專案，基於 Andrej Karpathy 的教學影片系列，涵蓋 GPT Tokenizer (minBPE) 和 nanoGPT 兩大主題。

---

## 完成項目

### 1. 專案基礎建設
- 在 `ml-workshop` repo 建立 `deep-learning-tutorial` 分支（從 main）
- 建立 `deep-learning-karpathy/` 專案資料夾結構
- 用 `uv` 初始化 Python 專案（pyproject.toml），設定所有依賴
- Git clone 了 `karpathy/minbpe` 和 `karpathy/nanoGPT` 作為參考
- 建立完整的 `ai-collab/` 文件架構（參照 app_puzzle 模式）

### 2. Tokenizer 教材（4 篇）

| 檔案 | 主題 | 重點內容 |
|------|------|----------|
| `01_bpe_basics.md` | BPE 演算法基礎 | 什麼是 Tokenizer、BPE 原理、手動執行範例 |
| `02_basic_tokenizer.md` | BasicTokenizer 實作 | 從零建構完整的 BPE Tokenizer |
| `03_regex_tokenizer.md` | RegexTokenizer & GPT-4 | Regex splitting、Special tokens、tiktoken 對比 |
| `04_exercises.md` | 練習題 | 5 個由淺到深的練習 |

### 3. nanoGPT 教材（4 篇）

| 檔案 | 主題 | 重點內容 |
|------|------|----------|
| `01_transformer_basics.md` | Transformer 基礎 | Self-Attention、MLP、Residual、LayerNorm |
| `02_gpt_architecture.md` | GPT 架構完整解析 | GPTConfig、Forward Pass、Weight Tying、Generation |
| `03_training_shakespeare.md` | 訓練莎士比亞 | 資料準備→DataLoader→Training Loop→文字生成 |
| `04_exercises.md` | 練習題 | 6 個由淺到深的練習（含 Scaling Laws） |

### 4. Python 工具腳本（4 個）

| 檔案 | 功能 | 測試狀態 |
|------|------|---------|
| `train_tokenizer.py` | 訓練 BPE tokenizer | ✓ 通過 |
| `compare_tokenizers.py` | 比較 Char/BPE/GPT-4 tokenizer | ✓ 通過 |
| `train_nanogpt.py` | 在 Shakespeare 上訓練 GPT | 已寫好，待 GPU 測試 |
| `sample_text.py` | 從訓練好的模型生成文字 | 已寫好，依賴 train 產出 |

### 5. ai-collab 文件

| 檔案 | 內容 |
|------|------|
| `rules.md` | 開發規範（溝通風格、程式碼風格、協作流程） |
| `project_guide.md` | 專案架構指南 |
| `commands.txt` | AI/Human 上手指南 |
| `dev_log.md` | 開發日誌 |
| `handover.md` | Agent 交接文件 |

---

## 教材設計特色

1. **雙層講解**：每個概念都有「高中生版」（用日常比喻解釋）和「專業版」（用準確術語和數學描述）
2. **具體例子**：所有概念都附帶可執行的 Python 程式碼
3. **循序漸進**：Tokenizer → nanoGPT 的學習路徑，符合 Karpathy 的教學順序
4. **動手練習**：每個主題都有 4-6 個練習題

---

## 依賴套件

```
torch>=2.0.0, tiktoken>=0.5.0, transformers>=4.30.0, datasets>=2.14.0
numpy>=1.24.0, matplotlib>=3.7.0, jupyter>=1.0.0, regex>=2023.0
pytest>=7.0.0, tqdm>=4.60.0, ipykernel>=6.20.0
```

全部透過 `uv sync` 安裝，已驗證可用。

---

## 測試結果

### train_tokenizer.py
```
Training text: 185,561 characters (Taylor Swift Wikipedia)
vocab_size: 512 (256 bytes + 256 merges)
Compression ratio: 1.68x
Encode/Decode: 100% 正確
```

### compare_tokenizers.py
```
English: Char 1.00x → BPE-500 1.62x → GPT-4 4.85x
Chinese: Char 1.00x → BPE-500 1.00x → GPT-4 1.76x
Code:    Char 1.00x → BPE-500 1.16x → GPT-4 3.29x
```

---

## 已知限制

1. **train_nanogpt.py** 尚未在 GPU 上測試（CPU 訓練可能需要 30+ 分鐘）
2. **Windows 相容性**：已處理 cp950 console encoding 問題
3. **Python 版本**：降為 3.9+ 以相容 ml-workshop 的 workspace 設定
4. **nanoGPT 已 deprecated**：原作者建議使用 nanochat，但作為教學用途仍然很好

---

## 下一步建議

### P1
1. 在 GPU 上測試 `train_nanogpt.py`
2. 建立 Jupyter Notebook 互動版教材
3. 加入更多練習解答

### P2
1. Attention 視覺化工具
2. 進階主題：model surgery、finetuning
3. 加入 nanochat 的比較教材

---

## 變更檔案清單

```
deep-learning-karpathy/              # 全新專案
├── pyproject.toml                   # 新增
├── tutorials/01_tokenizer/          # 新增 (4 篇)
├── tutorials/02_nanogpt/            # 新增 (4 篇)
├── scripts/                         # 新增 (4 個腳本)
├── references/minbpe/               # git clone
├── references/nanoGPT/              # git clone
└── ai-collab/                       # 新增 (完整文件架構)
```
