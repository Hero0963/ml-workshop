# 02 - GPT 模型架構完整解析

> 深入 nanoGPT 的 model.py — 只有 ~300 行的完整 GPT 實作
> 對應 `references/nanoGPT/model.py`

---

## GPTConfig：模型的設計圖

### 高中生版

蓋房子前要先畫設計圖。`GPTConfig` 就是 GPT 的設計圖，告訴我們：
- 房子有幾層（`n_layer`）
- 每層有幾個觀察窗（`n_head`）
- 每個觀察窗有多大（`n_embd`）
- 能看多遠（`block_size`）
- 認識多少字（`vocab_size`）

### 專業版

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024   # context length (最大序列長度)
    vocab_size: int = 50304  # 詞彙表大小 (GPT-2: 50257, padding 到 64 的倍數以提高效率)
    n_layer: int = 12        # Transformer Block 數量
    n_head: int = 12         # Attention head 數量
    n_embd: int = 768        # Embedding 維度
    dropout: float = 0.0     # Dropout 率
    bias: bool = True        # 是否使用 bias
```

**為什麼 vocab_size 是 50304 而不是 50257？**
- GPT-2 的實際詞彙量是 50,257
- padding 到 64 的倍數 (50304 = 786 × 64) 可以提高 GPU 運算效率
- 多出來的 47 個 token 不會被使用

---

## GPT 模型結構

### 高中生版

```
GPT 模型就像一棟大樓：

🏢 GPT 大樓
├── 1F: Token Embedding (wte) → 查字典，把每個字變成向量
├── 1F: Position Embedding (wpe) → 加上位置資訊
├── 1F: Dropout → 隨機「遮住」一些資訊（訓練用）
├── 2F-13F: Transformer Block × 12 → 12 層「思考」
│   每一層都在做：
│   ├── 看看前面的字（Attention）
│   └── 深入思考（MLP）
├── 頂樓: LayerNorm → 最後正規化
└── 天台: LM Head → 猜下一個字
```

### 專業版

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),    # token embedding
            wpe  = nn.Embedding(config.block_size, config.n_embd),    # position embedding
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: token embedding 和 LM head 共享權重
        self.transformer.wte.weight = self.lm_head.weight
```

**Weight Tying 的直覺**：
- Token Embedding 把 token → 向量
- LM Head 把 向量 → token 機率
- 這兩個操作是「互逆」的，用同一組權重很合理
- 節省 ~38M 參數（vocab_size × n_embd）

---

## Forward Pass 完整流程

### 高中生版

```
輸入：[15339, 1917]（"Hello world" 的 token IDs）

Step 1: 查字典 → 每個 token 變成 768 維向量
Step 2: 加位置 → 每個向量加上「我在第幾個位置」的資訊
Step 3: 丟進 12 層 Transformer
  Layer 1: 看看前面的字，思考一下
  Layer 2: 再看看前面的字，思考更深
  ...
  Layer 12: 最後一次看和思考
Step 4: 正規化
Step 5: 猜下一個 token → 得到 50304 個 token 的機率
```

### 專業版

```python
def forward(self, idx, targets=None):
    """
    Args:
        idx: (B, T) token indices
        targets: (B, T) target token indices (用於計算 loss)
    Returns:
        logits: (B, T, vocab_size) 或 (B, 1, vocab_size) (推理時)
        loss: scalar (訓練時) 或 None (推理時)
    """
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size

    pos = torch.arange(0, t, dtype=torch.long, device=device)

    # 1. Embedding
    tok_emb = self.transformer.wte(idx)     # (B, T, n_embd)
    pos_emb = self.transformer.wpe(pos)     # (T, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)

    # 2. Transformer Blocks
    for block in self.transformer.h:
        x = block(x)

    # 3. Final LayerNorm
    x = self.transformer.ln_f(x)

    # 4. LM Head
    if targets is not None:
        # 訓練：計算所有位置的 loss
        logits = self.lm_head(x)            # (B, T, vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )
    else:
        # 推理：只算最後一個位置（節省計算）
        logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
        loss = None

    return logits, loss
```

---

## 載入 GPT-2 預訓練權重

### 高中生版
OpenAI 已經花了大量資源訓練好 GPT-2。我們可以直接「下載」他們的成果來用，不需要自己從零開始訓練。就像買一個已經組好的模型，而不是自己慢慢拼。

### 專業版

```python
@classmethod
def from_pretrained(cls, model_type):
    """從 HuggingFace 載入 OpenAI 的 GPT-2 預訓練權重"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
    }[model_type]

    config_args['vocab_size'] = 50257  # GPT-2 的實際 vocab size
    config_args['block_size'] = 1024
    config_args['bias'] = True

    # 建立我們的模型結構
    config = GPTConfig(**config_args)
    model = GPT(config)

    # 從 HuggingFace 載入權重
    from transformers import GPT2LMHeadModel
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)

    # 複製權重（需要處理 Conv1D → Linear 的轉置）
    # ... (權重對齊邏輯)

    return model
```

**關鍵細節**：
- OpenAI 的 GPT-2 用 Conv1D，nanoGPT 用 Linear → 需要轉置某些權重
- 透過 HuggingFace `transformers` 庫下載權重
- 下載後的模型可以直接用來生成文字

---

## 文字生成（推理）

### 高中生版

生成文字就像寫接龍：
1. 給一個開頭
2. 猜下一個字
3. 把猜的字加到句子後面
4. 重複步驟 2-3

```
開頭：  "Once upon a"
第 1 步："Once upon a" → 猜 "time"
第 2 步："Once upon a time" → 猜 ","
第 3 步："Once upon a time," → 猜 "there"
...
```

### 專業版

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Autoregressive generation。

    Args:
        idx: (B, T) 初始 token 序列
        max_new_tokens: 要生成幾個新 token
        temperature: 控制隨機性 (0=確定性, 1=正常, >1=更隨機)
        top_k: 只從機率最高的 k 個 token 中選
    """
    for _ in range(max_new_tokens):
        # 裁剪到 block_size（context window 的限制）
        idx_cond = idx if idx.size(1) <= self.config.block_size \
                       else idx[:, -self.config.block_size:]

        # Forward pass
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature  # 取最後一個位置，除以 temperature

        # Top-k sampling
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Softmax → 機率 → 抽樣
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # 加入序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
```

**Temperature 的直覺**：
| Temperature | 效果 | 適用場景 |
|-------------|------|---------|
| 0.0 | 總是選最可能的 token | 事實性回答 |
| 0.7 | 保守但有變化 | 一般寫作 |
| 1.0 | 原始機率分布 | 平衡創意和一致性 |
| 1.5+ | 高度隨機 | 創意寫作/頭腦風暴 |

---

## 訓練優化器設定

### 專業版

nanoGPT 使用精心設計的 AdamW optimizer 設定：

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # 分兩組：需要 weight decay 的和不需要的
    # 2D 參數（矩陣權重）→ decay
    # 1D 參數（bias, LayerNorm）→ no decay
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer
```

**為什麼 bias 和 LayerNorm 不做 weight decay？**
- Weight decay 是正則化手段，防止權重過大
- Bias 和 LayerNorm 的參數很少，不太會 overfit
- 對它們做 decay 反而可能降低模型表達能力

---

## 重點整理

```
GPT = Token Embedding + Position Embedding
    + N × (LayerNorm → Attention → Residual
          → LayerNorm → MLP → Residual)
    + LayerNorm
    + Linear Head (weight tied with Token Embedding)
```

| 概念 | 一句話 |
|------|--------|
| Weight Tying | Embedding 和 LM Head 共享權重，省參數 |
| Pre-norm | LayerNorm 放在 sublayer 之前，更穩定 |
| Flash Attention | PyTorch 2.0 的高效 attention 實作 |
| Temperature | 控制生成的隨機性 |
| Top-k | 只從前 k 個最可能的 token 中選 |

**下一篇**：動手訓練 nanoGPT — 用莎士比亞文本訓練一個小型 GPT！
