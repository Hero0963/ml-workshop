# Deep Learning Karpathy 開發日誌

> 逆時序排列，最新在最上面

---

## 2026-03-28

### 教材與工具腳本完成 (完成)
- **Tokenizer 教材** (4 篇)：
  - `tutorials/01_tokenizer/01_bpe_basics.md` - BPE 演算法基礎
  - `tutorials/01_tokenizer/02_basic_tokenizer.md` - BasicTokenizer 實作
  - `tutorials/01_tokenizer/03_regex_tokenizer.md` - RegexTokenizer 與 GPT-4
  - `tutorials/01_tokenizer/04_exercises.md` - 練習題
- **nanoGPT 教材** (4 篇)：
  - `tutorials/02_nanogpt/01_transformer_basics.md` - Transformer 基礎
  - `tutorials/02_nanogpt/02_gpt_architecture.md` - GPT 模型架構
  - `tutorials/02_nanogpt/03_training_shakespeare.md` - 訓練莎士比亞文本
  - `tutorials/02_nanogpt/04_exercises.md` - 練習題
- **Python 工具腳本** (4 個，已測試通過)：
  - `scripts/train_tokenizer.py` - 訓練 BPE tokenizer
  - `scripts/compare_tokenizers.py` - 比較不同 tokenizer
  - `scripts/train_nanogpt.py` - 訓練 nanoGPT (Shakespeare)
  - `scripts/sample_text.py` - 用訓練好的模型生成文字
- 每篇教材都包含**高中生版**和**專業版**雙層講解
- 附帶具體程式碼範例，可直接執行

### 專案初始化 (完成)
- 建立 `deep-learning-tutorial` 分支 (從 main)
- 建立 `deep-learning-karpathy/` 專案資料夾結構
- uv 初始化 + 設定依賴 (torch, tiktoken, transformers, etc.)
- Git clone 參考 repo: `karpathy/minbpe`, `karpathy/nanoGPT`
- 建立 ai-collab 文件架構 (rules.md, project_guide.md, dev_log.md, handover.md, commands.txt)
