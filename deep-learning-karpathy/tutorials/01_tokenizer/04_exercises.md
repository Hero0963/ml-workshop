# 04 - Tokenizer 練習題

> 動手做才學得會！按照 `references/minbpe/exercise.md` 的建議路徑

---

## 練習 1：從零實作 BasicTokenizer（基礎）

### 任務
不看參考程式碼，自己實作 `BasicTokenizer`，包含：
- `train(text, vocab_size, verbose=False)`
- `encode(text) → list[int]`
- `decode(ids) → str`

### 驗證方式

```python
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3)

# 驗證 1: encode/decode 一致
assert tokenizer.decode(tokenizer.encode(text)) == text

# 驗證 2: 正確的 token 數量
assert tokenizer.encode(text) == [258, 100, 258, 97, 99]

print("✓ 全部通過！")
```

### 提示
1. `get_stats()` 和 `merge()` 兩個輔助函數先寫好
2. train 的核心就是一個 for loop，每次找最頻繁的 pair 並合併
3. encode 的關鍵：合併順序要和 train 一致（用 merge index 排序）
4. decode 最簡單：查表 + 串接 bytes

---

## 練習 2：加入 Regex Splitting（進階）

### 任務
將 BasicTokenizer 升級為 RegexTokenizer：
1. 使用 GPT-4 的 regex pattern 切割文字
2. 在每個 chunk 內分別做 BPE
3. 確保不會跨類別合併

### GPT-4 Regex Pattern
```python
import regex
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

### 驗證方式

```python
tokenizer = RegexTokenizer()
text = "Hello world! It's a beautiful day. 123 + 456 = 579"
tokenizer.train(text * 100, vocab_size=300)

# 驗證: encode/decode 一致
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
assert decoded == text

# 驗證: 新 token 不應跨類別
for idx in range(256, len(tokenizer.vocab)):
    token_text = tokenizer.vocab[idx].decode("utf-8", errors="replace")
    # 檢查 token 是否只包含同類字元
    print(f"Token {idx}: '{token_text}'")

print("✓ 全部通過！")
```

---

## 練習 3：比對 GPT-4 (tiktoken)（挑戰）

### 任務
載入 GPT-4 的 merges，讓你的 RegexTokenizer 產出與 `tiktoken` 完全相同的結果。

### 步驟
1. 安裝 tiktoken: `uv add tiktoken`
2. 從 `tiktoken` 取出 merges（參考 `references/minbpe/minbpe/gpt4.py` 的 `recover_merges`）
3. 處理 byte shuffle（GPT-4 的歷史遺留問題）
4. 比對結果

### 驗證方式

```python
import tiktoken

text = "hello123!!!? (안녕하세요!) 😉"

# tiktoken
enc = tiktoken.get_encoding("cl100k_base")
expected = enc.encode(text)

# 你的實作
# tokenizer = GPT4Tokenizer()  # 你的實作
# actual = tokenizer.encode(text)

# assert expected == actual, f"不匹配！expected={expected}, actual={actual}"
# print("✓ 與 GPT-4 完全匹配！")
```

### 已知難點
1. **恢復 merges**：tiktoken 存的是 rank，不是直接的 merge pairs
2. **Byte shuffle**：GPT-4 對前 256 個 byte token 做了排列置換
3. **Special tokens**：需要正確處理 `<|endoftext|>` 等

---

## 練習 4：壓縮率分析（數據分析）

### 任務
分析不同設定下的壓縮率。

```python
import matplotlib.pyplot as plt

# 準備測試文本
texts = {
    "English": "The quick brown fox jumps over the lazy dog. " * 100,
    "Chinese": "快速的棕色狐狸跳過了懶惰的狗。" * 100,
    "Code": "def hello():\n    print('hello world')\n" * 100,
    "Numbers": "3.14159265358979323846264338327950288419716939937510" * 100,
}

vocab_sizes = [260, 280, 300, 400, 500, 1000]

# 對每種文本和 vocab_size 計算壓縮率
results = {}
for name, text in texts.items():
    results[name] = []
    original_len = len(text.encode("utf-8"))
    for vs in vocab_sizes:
        tok = BasicTokenizer()
        tok.train(text, vocab_size=vs)
        compressed_len = len(tok.encode(text))
        ratio = original_len / compressed_len
        results[name].append(ratio)
        print(f"{name} (vs={vs}): {ratio:.2f}x compression")

# 畫圖
for name, ratios in results.items():
    plt.plot(vocab_sizes, ratios, marker='o', label=name)
plt.xlabel("Vocab Size")
plt.ylabel("Compression Ratio")
plt.title("BPE Compression Ratio vs Vocab Size")
plt.legend()
plt.grid(True)
plt.savefig("compression_analysis.png")
plt.show()
```

### 思考問題
1. 為什麼中文的壓縮率和英文不同？
2. 程式碼（Code）的壓縮率為什麼特別高？
3. 更大的 vocab_size 是否總是更好？有什麼 trade-off？

---

## 練習 5：訓練自己的 Tokenizer（綜合）

### 任務
在一個你感興趣的文本上訓練 tokenizer：
1. 選擇文本（小說、程式碼、新聞、維基百科等）
2. 嘗試不同的 vocab_size
3. 觀察學到的 tokens 是否有意義
4. 比較 Basic vs Regex 的差異

### 建議文本
- `references/minbpe/tests/taylorswift.txt`（Taylor Swift Wikipedia 頁面）
- `references/nanoGPT/data/shakespeare_char/input.txt`（Shakespeare）
- 任何你感興趣的文本（從網路下載）

### 可執行腳本
使用 `scripts/train_tokenizer.py` 來訓練和測試。

---

## 延伸閱讀

1. [原始 BPE 論文 (Sennrich et al. 2015)](https://arxiv.org/abs/1508.07909)
2. [GPT-2 論文 Section 2.2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [tiktoken](https://github.com/openai/tiktoken) - OpenAI 的高效 BPE 實作
4. [SentencePiece](https://github.com/google/sentencepiece) - Google 的 tokenizer（Llama 使用）
5. [Karpathy 的 minBPE lecture notes](../references/minbpe/lecture.md)
