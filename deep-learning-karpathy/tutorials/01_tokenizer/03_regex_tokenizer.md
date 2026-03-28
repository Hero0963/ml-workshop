# 03 - RegexTokenizer 與 GPT-4 Tokenizer

> 加入 regex splitting，實作 GPT-4 級別的 tokenizer
> 對應 `references/minbpe/minbpe/regex.py` 和 `references/minbpe/minbpe/gpt4.py`

---

## 為什麼需要 Regex？

### 高中生版

BasicTokenizer 有一個問題：它會把**不同類別**的東西合併在一起。

比如說，假如 "dog." 在文本中常出現，BPE 可能把 "dog." （包含句號）合併成一個 token。但這不太合理——"dog" 是一個單字，"." 是標點符號，它們不應該被混在一起。

**Regex Tokenizer 的做法**：先把文字按類別切開（字母歸字母、數字歸數字、標點歸標點），然後在每個類別**內部**分別做 BPE。這樣就不會跨類別合併了。

### 專業版

GPT-2 論文引入了一個關鍵的前處理步驟：在 BPE 之前，先用正則表達式（regex）將輸入文字按語言類別分割。這確保了 merge 不會跨越語義邊界。

**GPT-4 使用的 regex pattern**：

```python
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

這個 pattern 會將文字分成：
| Pattern | 匹配內容 | 範例 |
|---------|---------|------|
| `'(?i:[sdmt]\|ll\|ve\|re)` | 英文縮寫 | 's, 't, 'll, 've, 're |
| `[^\r\n\p{L}\p{N}]?+\p{L}+` | 字母單詞（可選前導標點）| hello, .world |
| `\p{N}{1,3}` | 1-3 位數字 | 123, 42, 7 |
| ` ?[^\s\p{L}\p{N}]++[\r\n]*` | 標點符號 | !!!, ., --- |
| `\s*[\r\n]` | 換行 | \n |
| `\s+(?!\S)\|\s+` | 空白字元 | spaces, tabs |

---

## RegexTokenizer 實作

### 與 BasicTokenizer 的差異

```
BasicTokenizer:
  "Hello world! 123" → 整段一起做 BPE

RegexTokenizer:
  "Hello world! 123" → 先切割：["Hello", " world", "!", " 123"]
                      → 每段分別做 BPE
                      → 合併結果
```

### 核心程式碼

```python
import regex  # 注意：用 regex 套件，不是內建的 re（支援 \p{L} 等 Unicode 類別）

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer:

    def __init__(self, pattern=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = regex.compile(self.pattern)
        self.merges = {}
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 關鍵差異：先用 regex 切割文字
        text_chunks = regex.findall(self.compiled_pattern, text)

        # 每個 chunk 分別轉為 bytes
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # 統計所有 chunk 的 pair（跨 chunk 不會合併！）
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            # 在每個 chunk 中分別做 merge
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]})")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        # 先用 regex 切割
        text_chunks = regex.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = list(chunk_bytes)
            # 在每個 chunk 內做 merge
            while len(chunk_ids) >= 2:
                stats = get_stats(chunk_ids)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = merge(chunk_ids, pair, idx)
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
```

---

## Basic vs Regex：實際對比

### 具體例子

```python
text = "Hello world! Hello there! 123 + 456 = 579"

# BasicTokenizer：可能學到跨類別的 merge
basic = BasicTokenizer()
basic.train(text * 100, vocab_size=280)
basic_tokens = basic.encode(text)

# RegexTokenizer：只在同類別內 merge
regex_tok = RegexTokenizer()
regex_tok.train(text * 100, vocab_size=280)
regex_tokens = regex_tok.encode(text)

print(f"Basic: {len(basic_tokens)} tokens")
print(f"Regex: {len(regex_tokens)} tokens")

# 觀察差異：Regex 版不會有 "d!" 或 "o " 這種跨類別的 token
for idx in range(256, len(regex_tok.vocab)):
    token = regex_tok.vocab[idx]
    print(f"  Token {idx}: {token}")
```

---

## Special Tokens

### 高中生版
Special tokens 是一些「暗號」，告訴 AI 一些特殊的事情：
- `<|endoftext|>` = 「這段文字結束了」
- `<|im_start|>` = 「接下來是某個角色的發言」

這些「暗號」不是從文字中學到的，而是手動加入的。

### 專業版

Special tokens 是在 BPE merge 之外額外註冊的 token。它們不參與 merge 過程，而是在 encode 時被直接匹配和替換。

```python
def register_special_tokens(self, special_tokens: dict[str, int]):
    """
    註冊 special tokens。
    例如: {"<|endoftext|>": 100257}
    """
    self.special_tokens = special_tokens
    self.vocab = self._build_vocab()  # 重建 vocab 以包含 special tokens
```

在 encode 時，需要先用 special token 切割文字，再對每個片段做 BPE：

```python
def encode(self, text: str, allowed_special="none") -> list[int]:
    # 處理 special tokens
    special = None
    if allowed_special == "all":
        special = self.special_tokens
    elif allowed_special == "none":
        special = {}
    elif isinstance(allowed_special, set):
        special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

    if not special:
        # 沒有 special tokens，直接做 BPE
        return self._encode_chunk(text)

    # 用 special tokens 切割文字
    # 例如: "hello<|endoftext|>world" → ["hello", "<|endoftext|>", "world"]
    special_pattern = "(" + "|".join(regex.escape(k) for k in special) + ")"
    special_chunks = regex.split(special_pattern, text)

    ids = []
    for part in special_chunks:
        if part in special:
            ids.append(special[part])
        else:
            ids.extend(self._encode_chunk(part))
    return ids
```

---

## 與 GPT-4 (tiktoken) 對比

### 驗證我們的實作

```python
# 需要安裝: pip install tiktoken
import tiktoken

text = "hello123!!!? (안녕하세요!) 😉"

# GPT-4 的 tokenizer (tiktoken)
enc = tiktoken.get_encoding("cl100k_base")
tiktoken_ids = enc.encode(text)
print(f"tiktoken: {tiktoken_ids}")

# 我們的 RegexTokenizer (載入 GPT-4 的 merges)
# 注意：要完全匹配需要額外處理 byte shuffle
# 詳見 references/minbpe/minbpe/gpt4.py
```

**GPT-4 Tokenizer 的特殊之處**：
1. vocab_size = 100,257（100K+ tokens）
2. 使用 `cl100k_base` 編碼
3. 有 byte shuffle（歷史原因造成的 byte 排列置換）
4. 包含多個 special tokens（`<|endoftext|>` 等）

---

## 壓縮率比較

### 具體數據

```python
# 用 Shakespeare 文本測試
with open("references/nanoGPT/data/shakespeare_char/input.txt", "r") as f:
    shakespeare = f.read()

# Character-level (no BPE)
char_tokens = len(shakespeare.encode("utf-8"))

# BasicTokenizer
basic = BasicTokenizer()
basic.train(shakespeare, vocab_size=500)
basic_tokens = len(basic.encode(shakespeare))

# RegexTokenizer
regex_tok = RegexTokenizer()
regex_tok.train(shakespeare, vocab_size=500)
regex_tokens = len(regex_tok.encode(shakespeare))

print(f"原始 bytes:        {char_tokens:,}")
print(f"BasicTokenizer:    {basic_tokens:,} (壓縮率 {char_tokens/basic_tokens:.1f}x)")
print(f"RegexTokenizer:    {regex_tokens:,} (壓縮率 {char_tokens/regex_tokens:.1f}x)")
```

---

## 重點整理

| 特性 | BasicTokenizer | RegexTokenizer |
|------|---------------|----------------|
| 前處理 | 無 | Regex pattern splitting |
| 跨類別合併 | 可能 | 不可能 |
| Special tokens | 不支援 | 支援 |
| GPT-4 相容 | 否 | 是 (加上 byte shuffle) |
| 適用場景 | 學習用/簡單實驗 | 生產環境 |

**下一篇**：練習題——自己動手實作 BPE Tokenizer！
