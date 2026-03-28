# 01 - BPE 演算法基礎

> 基於 Andrej Karpathy 的 "Let's build the GPT Tokenizer" 影片
> 參考 repo: `references/minbpe`

---

## 什麼是 Tokenizer？

### 高中生版

想像你要教一個外國朋友學中文。你不會直接丟一整本書給他，而是先教他認字：

1. 先學基本的部首（像「口」「木」「水」）
2. 再學由部首組成的字（像「林」= 木+木）
3. 最後學由字組成的詞（像「森林」= 森+林）

**Tokenizer 就是在做同樣的事**，但對象是電腦。電腦不認識文字，只認識數字。所以我們需要把文字切成小塊（tokens），再把每個小塊對應到一個數字。

**舉例**：
```
"Hello world" → [15339, 1917]
```
電腦看到的不是英文字母，而是 `15339` 和 `1917` 這兩個數字。

### 專業版

**Tokenizer** 是 LLM（大型語言模型）的前處理元件，負責將原始文字（string）轉換為整數序列（token IDs），反之亦然。

它執行三個核心操作：
1. **train(text, vocab_size)** - 從訓練文本中學習詞彙表（vocabulary）
2. **encode(text) → list[int]** - 將文字編碼為 token ID 序列
3. **decode(ids) → str** - 將 token ID 序列解碼回文字

Tokenizer 在 LLM 架構中的位置：
```
Raw Text → [Tokenizer: encode] → Token IDs → [Embedding Layer] → Vectors → [Transformer] → Output
```

**為什麼不用 character-level？**
- 字元級別的詞彙表太小（ASCII 只有 128 個），序列會非常長
- 模型需要更多步驟才能學到有意義的語義
- BPE 在詞彙表大小和序列長度之間取得平衡

---

## 什麼是 BPE (Byte Pair Encoding)？

### 高中生版

想像你有一堆樂高積木：
1. 最小的積木是 256 個基本字元（a, b, c, ... 和各種符號）
2. 你觀察哪兩塊積木**最常黏在一起**
3. 把它們合併成一塊新積木，給它一個新編號
4. 重複這個過程，直到你有足夠多的「大積木」

**具體例子**：

假設你有文字 `"aaabdaaabac"`：

```
第 0 步：[a, a, a, b, d, a, a, a, b, a, c]
         出現最多的配對是 (a, a)，出現了 4 次

第 1 步：把 (a, a) 合併成 Z
         [Z, a, b, d, Z, a, b, a, c]
         出現最多的配對是 (a, b)，出現了 2 次

第 2 步：把 (a, b) 合併成 Y
         [Z, Y, d, Z, Y, a, c]
         出現最多的配對是 (Z, Y)，出現了 2 次

第 3 步：把 (Z, Y) 合併成 X
         [X, d, X, a, c]
```

原本 11 個 token，現在只要 5 個！

### 專業版

**Byte Pair Encoding (BPE)** 是一種迭代壓縮演算法，由 Sennrich et al. (2015) 引入 NLP 領域，後被 GPT-2 論文 (2019) 推廣為 LLM 的標準 tokenization 方法。

**演算法流程**：

1. 從 byte-level 開始：將輸入文字轉為 UTF-8 bytes，得到 0-255 的整數序列
2. 初始詞彙表包含 256 個 byte tokens
3. 迭代執行：
   - 統計所有相鄰 pair 的出現次數
   - 找出出現最頻繁的 pair
   - 將該 pair 合併為新 token，分配下一個可用 ID
   - 在序列中替換所有該 pair 的出現
4. 重複直到達到目標 vocab_size

**關鍵特性**：
- **Byte-level**：在 UTF-8 bytes 上操作，天然支援所有語言和特殊字元
- **貪心演算法**：每次選擇最頻繁的 pair，是局部最優策略
- **壓縮效果**：常見的子詞（subword）會被合併為單一 token，減少序列長度

**Python 實作**：

```python
def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """統計所有相鄰 pair 的出現次數"""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """將序列中所有 pair 替換為新 token idx"""
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

## 具體例子：手動執行 BPE

讓我們用 Wikipedia 上的經典例子，在 Python 中實際操作：

```python
text = "aaabdaaabac"

# Step 1: 轉為 bytes
tokens = list(text.encode("utf-8"))
print(f"原始 tokens: {tokens}")
# [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]
# (a=97, b=98, c=99, d=100)

# Step 2: 統計 pair
stats = get_stats(tokens)
print(f"Pair 統計: {stats}")
# {(97, 97): 4, (97, 98): 2, (98, 100): 1, (100, 97): 1, (97, 99): 1}

# Step 3: 最常出現的 pair
top_pair = max(stats, key=stats.get)
print(f"最頻繁 pair: {top_pair} (出現 {stats[top_pair]} 次)")
# (97, 97) 出現 4 次

# Step 4: 合併！新 token ID = 256
tokens = merge(tokens, (97, 97), 256)
print(f"合併後: {tokens}")
# [256, 97, 98, 100, 256, 97, 98, 97, 99]

# 繼續第二輪...
stats = get_stats(tokens)
top_pair = max(stats, key=stats.get)
print(f"第二輪最頻繁: {top_pair}")
# (97, 98) → 合併為 257

tokens = merge(tokens, (97, 98), 257)
print(f"第二輪後: {tokens}")
# [256, 257, 100, 256, 257, 97, 99]

# 繼續第三輪...
stats = get_stats(tokens)
top_pair = max(stats, key=stats.get)
tokens = merge(tokens, top_pair, 258)
print(f"第三輪後: {tokens}")
# [258, 100, 258, 97, 99]
```

**最終結果**：
- 原本 11 個 tokens → 壓縮到 5 個 tokens
- 詞彙表：256 個 bytes + 3 個新 token = 259 個 token
- 合併紀錄：`(97,97)→256`, `(97,98)→257`, `(256,257)→258`

---

## 為什麼 Tokenization 很重要？

### 高中生版

Tokenization 是很多 AI 「奇怪行為」的根本原因：

| 問題 | 原因 |
|------|------|
| AI 不會拼字 | 它看到的是 token，不是字母 |
| AI 數學差 | 數字被切成奇怪的 token（"380" 可能變成 "3" + "80"）|
| AI 不太會處理非英文 | 非英文字元需要更多 bytes，token 更碎片化 |
| AI 不太會反轉字串 | 它不是逐字元處理的 |

### 專業版

Karpathy 在影片中列出了多個 tokenization 造成的 LLM 問題：

1. **字元操作困難**：LLM 的基本單位是 token 而非字元，無法直接進行字元級操作
2. **算術不一致**：數字的 tokenization 不一致（"127" 是一個 token，"677" 卻是兩個）
3. **多語言效率差異**：同樣的語義，非英文可能需要更多 token
4. **特殊 token 漏洞**：`<|endoftext|>` 等特殊 token 可能被攻擊者利用
5. **SolidGoldMagikarp**：某些 token 在訓練時幾乎沒出現，導致 embedding 未被充分訓練

---

## 小結

| 概念 | 一句話總結 |
|------|-----------|
| Tokenizer | 把文字轉成數字的工具 |
| BPE | 反覆合併最常見的 pair，建立詞彙表 |
| Byte-level | 在 UTF-8 bytes 上操作，支援所有語言 |
| vocab_size | 詞彙表大小，典型值：32K-100K |

**下一篇**：我們會用 Python 從零實作一個完整的 BasicTokenizer。
