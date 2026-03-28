# 02 - BasicTokenizer 實作

> 從零建構一個最基本的 BPE Tokenizer
> 對應 `references/minbpe/minbpe/basic.py`

---

## 目標

### 高中生版
上一篇我們學了 BPE 的原理。這篇我們要真的寫出一個 Tokenizer 程式，它能做三件事：
1. **學習**（train）：看一堆文字，學會怎麼切
2. **編碼**（encode）：把文字變成數字
3. **解碼**（decode）：把數字變回文字

### 專業版
實作 `BasicTokenizer` 類別，包含 `train()`、`encode()`、`decode()` 三個核心方法。這是最簡單的 BPE 實作，直接在 raw UTF-8 bytes 上操作，不做任何前處理。

---

## 基礎工具函數

首先我們需要兩個輔助函數（在上一篇已經見過）：

```python
def get_stats(ids: list[int], counts: dict | None = None) -> dict:
    """
    統計所有相鄰 pair 的出現次數。

    高中生版：數一數每兩個相鄰數字的組合出現了幾次
    專業版：O(n) 掃描，回傳 {(id_i, id_{i+1}): count} 的字典

    >>> get_stats([1, 2, 3, 1, 2])
    {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    把序列中所有的 pair 替換成新 token。

    高中生版：找到所有連續出現的 (a, b)，用新數字 c 取代
    專業版：線性掃描，遇到匹配的 pair 就替換為 idx

    >>> merge([1, 2, 3, 1, 2], (1, 2), 4)
    [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
```

---

## BasicTokenizer 完整實作

### 高中生版理解架構

```
BasicTokenizer
├── train()   → 學習：反覆合併最常見的 pair
├── encode()  → 編碼：用學到的規則把文字變數字
└── decode()  → 解碼：把數字變回文字
```

### 專業版完整程式碼

```python
class BasicTokenizer:
    """
    最基本的 BPE Tokenizer。
    直接在 UTF-8 bytes 上操作，不做 regex splitting。
    """

    def __init__(self):
        self.merges = {}   # (int, int) -> int : 合併規則
        self.vocab = {}    # int -> bytes : 詞彙表

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        從文本中訓練 tokenizer。

        高中生版：看一堆文字，找出最常一起出現的組合，記下來
        專業版：迭代 BPE，每次合併最頻繁的 byte pair

        Args:
            text: 訓練文本
            vocab_size: 目標詞彙表大小 (必須 >= 256)
            verbose: 是否印出每步的合併資訊
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (the number of byte values)"
        num_merges = vocab_size - 256

        # Step 1: 將文字轉為 UTF-8 bytes
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)  # [0, 1, ..., 255] 的整數列表

        # Step 2: 初始化詞彙表（256 個 byte tokens）
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        # Step 3: 迭代合併
        for i in range(num_merges):
            # 統計所有 pair
            stats = get_stats(ids)
            if not stats:
                break  # 沒有 pair 可合併了

            # 找最頻繁的 pair
            pair = max(stats, key=stats.get)

            # 分配新 token ID
            idx = 256 + i

            # 在序列中替換
            ids = merge(ids, pair, idx)

            # 記錄合併規則和詞彙
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} "
                    f"({vocab[idx]}) had {stats[pair]} occurrences"
                )

        # 儲存到 instance
        self.merges = merges
        self.vocab = vocab

    def decode(self, ids: list[int]) -> str:
        """
        將 token IDs 解碼為文字。

        高中生版：查表，把每個數字對應的字元片段接起來
        專業版：從 vocab 查找每個 ID 對應的 bytes，串接後 UTF-8 decode
        """
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def encode(self, text: str) -> list[int]:
        """
        將文字編碼為 token IDs。

        高中生版：先把文字拆成最小單位，再反覆合併（按照訓練時學到的順序）
        專業版：先轉 UTF-8 bytes，再按 merge 優先順序迭代合併
        """
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)

        while len(ids) >= 2:
            # 找出當前序列中，merge 優先順序最高（index 最小）的 pair
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            # 如果沒有任何可合併的 pair，結束
            if pair not in self.merges:
                break

            # 執行合併
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)

        return ids
```

---

## 實際使用範例

### 範例 1：經典 BPE 範例

```python
tokenizer = BasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3, verbose=True)

# 輸出：
# merge 1/3: (97, 97) -> 256 (b'aa') had 4 occurrences
# merge 2/3: (97, 98) -> 257 (b'ab') had 2 occurrences
# merge 3/3: (256, 257) -> 258 (b'aaab') had 2 occurrences

encoded = tokenizer.encode(text)
print(f"編碼結果: {encoded}")
# [258, 100, 258, 97, 99]

decoded = tokenizer.decode(encoded)
print(f"解碼結果: {decoded}")
# aaabdaaabac  ← 完全還原！

assert decoded == text, "解碼結果應該與原文完全相同"
```

### 範例 2：真實文本

```python
# 用一段真實文字訓練
training_text = """
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy cat.
"""

tokenizer = BasicTokenizer()
tokenizer.train(training_text, vocab_size=280, verbose=True)

# 測試
test = "The quick brown fox"
ids = tokenizer.encode(test)
print(f"'{test}' → {ids} ({len(ids)} tokens)")
print(f"解碼驗證: '{tokenizer.decode(ids)}'")

# 查看詞彙表中的新 tokens
print("\n新學到的 tokens:")
for idx in range(256, len(tokenizer.vocab)):
    print(f"  {idx}: {tokenizer.vocab[idx]}")
```

### 範例 3：中文支援

```python
tokenizer = BasicTokenizer()
text = "你好你好你好世界世界" * 10  # 重複多次讓 BPE 有足夠的統計
tokenizer.train(text, vocab_size=270)

encoded = tokenizer.encode("你好世界")
decoded = tokenizer.decode(encoded)
print(f"中文測試: '{decoded}' ({len(encoded)} tokens)")

# 注意：中文字元在 UTF-8 中是 3 bytes
# "你" = [228, 189, 160], "好" = [229, 165, 189]
# BPE 會學到合併這些 bytes
```

---

## 深入理解 encode 的巧妙之處

### 高中生版
encode 的關鍵在於：**合併的順序很重要**！

假設 train 時學到了兩個合併規則：
1. (a, b) → X （第一個學到的）
2. (X, c) → Y （第二個學到的）

encode 時，我們必須**先合併 (a, b) 再合併 (X, c)**，因為這是訓練時的順序。

### 專業版

encode 使用 `min(stats, key=lambda p: self.merges.get(p, float("inf")))` 來選擇合併順序：
- `self.merges` 的 value 是 token ID（256, 257, 258...）
- ID 越小 = 越早學到 = 優先合併
- 不在 merges 中的 pair 返回 `inf`，表示不合併

這確保了 encode 和 train 使用相同的合併順序，保證一致性。

---

## 關鍵概念對照表

| 概念 | 高中生版 | 專業版 |
|------|---------|--------|
| merges | 合併規則表 | `dict[(int,int), int]` 記錄 pair → new_id |
| vocab | 每個數字代表什麼 | `dict[int, bytes]` 記錄 id → byte sequence |
| train | 找出常見組合並記住 | BPE 迭代：統計 → 合併最頻繁 pair |
| encode | 按規則把文字變數字 | UTF-8 → bytes → 按 merge 順序迭代合併 |
| decode | 查表把數字變回文字 | 查 vocab → 串接 bytes → UTF-8 decode |

---

## 練習

1. **基礎**：用 `BasicTokenizer` 訓練一個 vocab_size=300 的 tokenizer，觀察它學到了什麼 tokens
2. **進階**：計算不同 vocab_size (260, 280, 300, 500) 對同一段文字的壓縮率
3. **挑戰**：試著在 Shakespeare 文本上訓練（見 `references/nanoGPT/data/shakespeare_char/`）

**下一篇**：我們會加入 regex pattern splitting，實作 `RegexTokenizer`——GPT-4 所使用的 tokenizer。
