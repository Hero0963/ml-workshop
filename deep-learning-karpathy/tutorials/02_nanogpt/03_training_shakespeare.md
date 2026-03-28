# 03 - 訓練 nanoGPT：莎士比亞文本生成

> 從資料準備到文字生成的完整流程
> 對應 `references/nanoGPT/` 的訓練流程

---

## 概覽

### 高中生版

我們要做的事情很酷：
1. 餵一堆莎士比亞的劇本給 AI
2. AI 學會莎士比亞的寫作風格
3. 然後 AI 自己寫出莎士比亞風格的劇本！

就像你讀了很多金庸小說後，也能模仿他的風格寫武俠故事。

### 專業版

完整的訓練 pipeline：
```
Raw Text → Tokenize → Train/Val Split → DataLoader → Training Loop → Generation
```

我們會先用 **character-level** tokenization（每個字元是一個 token），因為這是最簡單的方式，適合理解核心概念。之後再升級到 BPE tokenizer。

---

## Step 1: 資料準備

### 高中生版

首先要準備「課本」給 AI 讀。我們用莎士比亞的所有劇本（大約 100 萬個字元）。

### 具體程式碼

```python
# scripts/prepare_shakespeare.py

import os
import numpy as np

# 下載莎士比亞文本
input_file = "references/nanoGPT/data/shakespeare_char/input.txt"
if not os.path.exists(input_file):
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, input_file)

with open(input_file, 'r') as f:
    data = f.read()

print(f"文本長度: {len(data):,} 字元")
print(f"前 200 字元:\n{data[:200]}")

# Character-level tokenization
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"詞彙表大小: {vocab_size}")
print(f"字元: {''.join(chars)}")

# 建立 encoder/decoder
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 驗證
print(f"\n'hello' 編碼: {encode('hello')}")
print(f"解碼回來: {decode(encode('hello'))}")

# 切分 Train / Validation
data_ids = np.array(encode(data), dtype=np.uint16)
n = int(0.9 * len(data_ids))
train_data = data_ids[:n]
val_data = data_ids[n:]

print(f"\nTrain: {len(train_data):,} tokens")
print(f"Val:   {len(val_data):,} tokens")

# 儲存
train_data.tofile("train.bin")
val_data.tofile("val.bin")
print("✓ 已儲存 train.bin 和 val.bin")
```

---

## Step 2: DataLoader

### 高中生版

AI 不是一次讀完整本書，而是一次讀一小段（比如 256 個字元）。就像你做考卷時，一次看一道題目。

**訓練的方式**：
- 給 AI 看 256 個字元
- 問它：「每個位置的下一個字元是什麼？」
- 告訴它正確答案
- AI 從錯誤中學習

### 具體程式碼

```python
import torch

def get_batch(split, block_size=256, batch_size=64):
    """
    隨機取一個 batch 的訓練資料。

    Args:
        split: 'train' or 'val'
        block_size: context length (每次看多少 token)
        batch_size: 一次看幾段

    Returns:
        x: (B, T) 輸入 token
        y: (B, T) 目標 token (x 往右移一位)
    """
    data = train_data if split == 'train' else val_data
    # 隨機選起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x, y

# 示範
x, y = get_batch('train', block_size=8, batch_size=1)
print(f"輸入: {x[0].tolist()}")
print(f"目標: {y[0].tolist()}")
print()
print("訓練樣本解析：")
for t in range(8):
    context = x[0, :t+1].tolist()
    target = y[0, t].item()
    print(f"  context: {decode(context):20s} → target: '{itos[target]}'")
```

**輸出類似**：
```
訓練樣本解析：
  context: F                    → target: 'i'
  context: Fi                   → target: 'r'
  context: Fir                  → target: 's'
  context: Firs                 → target: 't'
  context: First                → target: ' '
  context: First                → target: 'C'
  context: First C              → target: 'i'
  context: First Ci             → target: 't'
```

---

## Step 3: 訓練配置

### 高中生版

我們要設定「學習計劃」：
- 模型大小（小到能在你的電腦上跑）
- 學多久（迭代次數）
- 學多快（學習率）

### 有 GPU vs 只有 CPU

```python
# === 有 GPU 的配置 ===
config_gpu = dict(
    block_size=256,     # context length
    batch_size=64,      # 一次看幾段
    n_layer=6,          # 6 層 Transformer
    n_head=6,           # 6 個 attention head
    n_embd=384,         # embedding 維度
    max_iters=5000,     # 訓練迭代次數
    learning_rate=1e-3, # 學習率
    dropout=0.2,        # dropout 率
    device='cuda',      # GPU
)

# === 只有 CPU 的配置 ===
config_cpu = dict(
    block_size=64,      # 較短的 context
    batch_size=12,      # 較小的 batch
    n_layer=4,          # 4 層
    n_head=4,           # 4 個 head
    n_embd=128,         # 較小的 embedding
    max_iters=2000,     # 較少迭代
    learning_rate=1e-3,
    dropout=0.0,        # 小模型不需要 dropout
    device='cpu',
)
```

---

## Step 4: 訓練循環

### 高中生版

訓練過程就像上學：
1. **出題**：從莎士比亞文本中取一段
2. **答題**：模型猜每個位置的下一個字
3. **批改**：比較答案，計算「錯了多少」（loss）
4. **改進**：調整模型參數，讓下次答得更好
5. 重複幾千次...

### 具體程式碼

```python
import torch
from model import GPT, GPTConfig

# 選擇裝置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用裝置: {device}")

# 建立模型
config = GPTConfig(
    block_size=256,
    vocab_size=vocab_size,  # 65 (character-level)
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
)
model = GPT(config)
model = model.to(device)
print(f"模型參數量: {model.get_num_params()/1e6:.2f}M")

# 優化器
optimizer = model.configure_optimizers(
    weight_decay=1e-1,
    learning_rate=1e-3,
    betas=(0.9, 0.99),
    device_type=device,
)

# 訓練！
max_iters = 5000
eval_interval = 500

for iter in range(max_iters):
    # 每隔一段評估
    if iter % eval_interval == 0:
        model.eval()
        losses = []
        for _ in range(20):
            X, Y = get_batch('val')
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses.append(loss.item())
        val_loss = sum(losses) / len(losses)
        print(f"Step {iter}: val loss = {val_loss:.4f}")
        model.train()

    # 訓練一步
    X, Y = get_batch('train')
    X, Y = X.to(device), Y.to(device)
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"訓練完成！最終 loss: {loss.item():.4f}")
```

**訓練過程的 loss 變化（大概）**：
```
Step 0:    val loss = 4.2000  ← 隨機猜（65 個字元，-ln(1/65) ≈ 4.17）
Step 500:  val loss = 2.1000  ← 開始學到一些模式
Step 1000: val loss = 1.8000  ← 學到常見字母組合
Step 2000: val loss = 1.6000  ← 學到單字和句子結構
Step 3000: val loss = 1.5200  ← 學到莎士比亞的風格
Step 5000: val loss = 1.4700  ← 接近收斂
```

---

## Step 5: 生成文字

### 高中生版

訓練完成！現在 AI 可以寫莎士比亞風格的劇本了。我們給它一個開頭，它會接著寫下去。

### 具體程式碼

```python
model.eval()

# 起始 token（換行符）
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# 生成 500 個字元
generated = model.generate(context, max_new_tokens=500, temperature=0.8)
print(decode(generated[0].tolist()))
```

**生成結果範例**（GPU 訓練 5000 步後）：
```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.
```

**不同 temperature 的效果**：

```python
# temperature = 0.5 (保守)
# → 語法正確，但可能重複
print("=== Temperature 0.5 ===")
print(decode(model.generate(context, 200, temperature=0.5)[0].tolist()))

# temperature = 1.0 (標準)
# → 平衡的創意和一致性
print("=== Temperature 1.0 ===")
print(decode(model.generate(context, 200, temperature=1.0)[0].tolist()))

# temperature = 1.5 (創意)
# → 更多意外的組合，但可能不太通順
print("=== Temperature 1.5 ===")
print(decode(model.generate(context, 200, temperature=1.5)[0].tolist()))
```

---

## 從 Character-Level 升級到 BPE

### 為什麼要升級？

| 特性 | Character-Level | BPE |
|------|----------------|-----|
| 詞彙表大小 | ~65 | ~50,000 |
| 序列長度 | 很長（每字元一個 token） | 較短（常見詞合併為一個 token） |
| 學習效率 | 低（要學很多字母組合） | 高（直接學詞彙級語義） |
| 適合 | 小規模實驗 | 實際應用 |

### 使用 GPT-2 BPE Tokenizer

```python
import tiktoken

# 載入 GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# 編碼
text = "First Citizen: Before we proceed any further, hear me speak."
ids = enc.encode(text)
print(f"Character tokens: {len(text)}")
print(f"BPE tokens: {len(ids)}")
# Character: 62 tokens → BPE: ~15 tokens，壓縮約 4 倍！
```

---

## 常見問題

### Q: 訓練要多久？
- **GPU (A100)**：~3 分鐘
- **CPU**：~30 分鐘（用小模型配置）
- **MacBook MPS**：~10 分鐘

### Q: Loss 多低才算好？
- Character-level Shakespeare：~1.47 是 nanoGPT 的 benchmark
- Loss 越低，生成的文字越像真的莎士比亞
- 但太低可能 overfit（背下來而不是學到規律）

### Q: 能用來寫中文嗎？
- Character-level tokenizer 可以（把每個中文字元當 token）
- 但中文字元太多，建議用 BPE tokenizer
- 需要中文訓練資料

---

## 小結

| 步驟 | 做什麼 | 為什麼 |
|------|--------|--------|
| 1. 資料準備 | 文字→數字 | 電腦只懂數字 |
| 2. DataLoader | 隨機取小段 | 記憶體和效率考量 |
| 3. 配置模型 | 決定大小 | 平衡能力和計算資源 |
| 4. 訓練循環 | 反覆學習 | 讓模型越來越好 |
| 5. 生成 | 逐字產出 | 享受成果！|

**下一篇**：練習題——自己動手訓練和改進 nanoGPT！
