# 01 - Transformer 基礎

> 理解 GPT 的核心引擎：Transformer 架構
> 參考 Karpathy 的 "Let's build GPT from scratch" 和 nanoGPT repo

---

## 什麼是語言模型？

### 高中生版

語言模型就是一個**猜字遊戲高手**。

給它一串文字的開頭，它會猜下一個字（或 token）最可能是什麼：

```
輸入：「今天天氣真」
模型猜：「好」（機率 45%）、「差」（機率 20%）、「熱」（機率 15%）...
```

它不是真的「理解」文字，而是從大量文本中學到了**統計規律**。就像你讀了很多書之後，能猜到故事接下來會怎樣發展。

### 專業版

**語言模型（Language Model, LM）** 是一個機率分布 $P(x_{t+1} | x_1, x_2, ..., x_t)$，給定前面的 token 序列，預測下一個 token 的機率。

GPT 系列模型是 **autoregressive language model**：
- 訓練時：用 cross-entropy loss 最小化 next-token prediction error
- 推理時：逐步生成 token，每次把新生成的 token 加入 context

---

## Transformer 架構概覽

### 高中生版

Transformer 就像一個超級讀書機器，分三個步驟工作：

```
1. 📖 Token Embedding：把每個 token（數字）變成一個「個性向量」
   → 就像給每個字一張「身分證」，上面有很多特徵數值

2. 🔍 Self-Attention：讓每個字「看看」前面的字，決定哪些最重要
   → 就像讀一篇文章時，會回頭看前面的內容來理解現在的句子

3. 🧠 Feed-Forward：對每個位置做一個「思考」
   → 消化吸收了上下文之後，做深入思考
```

這三步重複好幾層（GPT-2 有 12 層），每層都在「加深理解」。

### 專業版

Transformer (decoder-only) 的架構：

```
Input Token IDs: [t1, t2, t3, ..., tT]
       ↓
Token Embedding (wte): [E1, E2, ..., ET]  (vocab_size × n_embd)
       +
Position Embedding (wpe): [P1, P2, ..., PT]  (block_size × n_embd)
       ↓
    X = E + P   (B, T, n_embd)
       ↓
┌──────────────────────────────┐
│  Transformer Block × N_layer │
│  ┌─────────────────────────┐ │
│  │ LayerNorm               │ │
│  │ Causal Self-Attention   │ │
│  │ Residual Connection (+) │ │
│  │ LayerNorm               │ │
│  │ MLP (Feed-Forward)      │ │
│  │ Residual Connection (+) │ │
│  └─────────────────────────┘ │
└──────────────────────────────┘
       ↓
    LayerNorm
       ↓
    Linear Head (n_embd → vocab_size)
       ↓
    Logits → Softmax → 下一個 token 的機率分布
```

---

## Self-Attention 詳解

### 高中生版

想像你在上課，老師問了一個問題。你要回答，但需要「參考」前面聽到的內容。

Self-Attention 就是這個「參考」的過程：

1. **Query (問題)**：「我現在需要什麼資訊？」
2. **Key (標籤)**：每個前面的位置都舉牌說「我有什麼資訊」
3. **Value (內容)**：每個位置真正的資訊內容

```
例子："The cat sat on the ___"

Query (for ___): 「我需要一個名詞，跟貓坐的地方有關」
Key ("cat"):     「我是動物名詞」     → 相關度：低
Key ("sat"):     「我是動作」         → 相關度：低
Key ("on"):      「我表示位置關係」   → 相關度：高！
Key ("the"):     「我是冠詞」         → 相關度：中

→ 注意力最集中在 "on" 和 "the"，所以模型會猜 "mat" 之類的地方名詞
```

### 專業版

**Scaled Dot-Product Attention**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Causal (因果) Attention**：使用下三角遮罩，確保每個位置只能看到自己和之前的位置。

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V 三個投影矩陣合併成一個（效率考量）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 輸出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # 一次算出 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 重塑為 multi-head: (B, T, C) → (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention: Q @ K^T / sqrt(d_k)，加上 causal mask
        # PyTorch 2.0+ 支援 Flash Attention
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # 重組 multi-head 輸出
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
```

**Multi-Head Attention 的直覺**：
- 每個 head 關注不同的「面向」
- Head 1 可能關注語法關係
- Head 2 可能關注語義關係
- Head 3 可能關注位置接近的 token
- 最後合併所有 head 的觀察結果

---

## MLP (Feed-Forward Network)

### 高中生版
Attention 讓模型「看到」上下文，MLP 讓模型「思考」看到的東西。

就像你讀完一段文字後，腦中會做一些「消化」和「推理」——MLP 就是在做這件事。

### 專業版

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)  # 擴展 4 倍
        self.gelu   = nn.GELU()                                      # 非線性激活
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)   # 投影回原維度

    def forward(self, x):
        x = self.c_fc(x)      # (B, T, C) → (B, T, 4C)
        x = self.gelu(x)      # 非線性
        x = self.c_proj(x)    # (B, T, 4C) → (B, T, C)
        return x
```

**為什麼先擴展再壓縮？**
- 擴展到 4 倍維度 → 更大的「思考空間」
- GELU 激活函數 → 引入非線性（否則多層 Linear 等於一層）
- 壓縮回原維度 → 保持 residual connection 的維度一致

---

## Residual Connection & LayerNorm

### 高中生版

**Residual Connection（殘差連接）**：
走捷徑！不只看最新的思考結果，也保留之前的資訊。
```
新結果 = 之前的資訊 + 新的思考
```

**LayerNorm（層正規化）**：
把數值調整到合理範圍，就像考試分數正規化，避免某些數值太大或太小。

### 專業版

```python
class Block(nn.Module):
    """一個 Transformer Block = Attention + MLP，加上 Residual + LayerNorm"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-norm: LayerNorm 放在 sublayer 之前（GPT-2 的做法）
        x = x + self.attn(self.ln_1(x))  # Residual + Attention
        x = x + self.mlp(self.ln_2(x))   # Residual + MLP
        return x
```

**Pre-norm vs Post-norm**：
- 原始 Transformer 論文用 post-norm（LN 在 sublayer 之後）
- GPT-2 改用 pre-norm（LN 在 sublayer 之前），訓練更穩定

---

## GPT-2 模型大小

| 模型 | n_layer | n_head | n_embd | 參數量 |
|------|---------|--------|--------|--------|
| GPT-2 | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 350M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

### 參數量計算（以 GPT-2 124M 為例）

```python
# Token Embedding: vocab_size × n_embd = 50257 × 768 ≈ 38.6M
# Position Embedding: block_size × n_embd = 1024 × 768 ≈ 0.8M
# 每個 Block:
#   Attention: 4 × n_embd² = 4 × 768² ≈ 2.4M
#   MLP: 8 × n_embd² = 8 × 768² ≈ 4.7M
#   → 每個 Block ≈ 7.1M
# 12 個 Blocks: 12 × 7.1M ≈ 85.2M
# LM Head: (weight tying with wte, 不額外增加參數)
# 總計: 38.6 + 0.8 + 85.2 ≈ 124.6M ✓
```

---

## 小結

| 元件 | 高中生版 | 功能 |
|------|---------|------|
| Token Embedding | 每個字的「身分證」 | 將 token ID 轉為向量 |
| Position Embedding | 記錄字的位置 | 讓模型知道字的順序 |
| Self-Attention | 「回頭看」前面的字 | 建立 token 間的關聯 |
| MLP | 「深入思考」 | 非線性轉換 |
| Residual | 「走捷徑」 | 保留原始資訊 |
| LayerNorm | 「正規化分數」 | 穩定訓練 |
| LM Head | 「猜下一個字」 | 輸出 token 機率 |

**下一篇**：我們會看完整的 GPT 模型定義，並學會如何載入 GPT-2 的預訓練權重。
