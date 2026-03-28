# 04 - nanoGPT 練習題

> 動手做才學得會！

---

## 練習 1：在莎士比亞上訓練 nanoGPT（基礎）

### 任務
使用 `scripts/train_nanogpt.py` 在 Shakespeare 文本上訓練一個 character-level GPT。

### 步驟
```bash
# 1. 確保依賴已安裝
cd deep-learning-karpathy/
uv sync

# 2. 訓練
uv run python scripts/train_nanogpt.py

# 3. 生成文字
uv run python scripts/sample_text.py
```

### 觀察與思考
1. 觀察 loss 的下降曲線
2. 在不同訓練階段生成文字，比較品質差異：
   - 100 步：完全是亂碼
   - 500 步：開始出現英文單字
   - 2000 步：出現像樣的句子
   - 5000 步：有莎士比亞的味道了

---

## 練習 2：調整超參數（進階）

### 任務
嘗試修改以下超參數，觀察對訓練結果的影響：

```python
experiments = {
    "baseline": dict(n_layer=6, n_head=6, n_embd=384, block_size=256),
    "deeper":   dict(n_layer=12, n_head=6, n_embd=384, block_size=256),
    "wider":    dict(n_layer=6, n_head=6, n_embd=768, block_size=256),
    "longer":   dict(n_layer=6, n_head=6, n_embd=384, block_size=512),
    "tiny":     dict(n_layer=2, n_head=2, n_embd=64,  block_size=64),
}
```

### 記錄表格
| 實驗 | 參數量 | 最終 val loss | 訓練時間 | 生成品質 |
|------|--------|--------------|---------|---------|
| baseline | ? M | ? | ? min | |
| deeper | ? M | ? | ? min | |
| wider | ? M | ? | ? min | |
| longer | ? M | ? | ? min | |
| tiny | ? M | ? | ? min | |

### 思考問題
1. 加深（more layers）和加寬（more embedding dim）哪個效果更好？
2. 更長的 context（block_size）帶來什麼好處和代價？
3. 極小的模型（tiny）能學到什麼？

---

## 練習 3：在自己的資料上訓練（進階）

### 任務
準備你自己感興趣的文本資料，用 nanoGPT 訓練。

### 建議資料
1. **中文古詩**：收集唐詩三百首
2. **程式碼**：收集 Python 程式碼
3. **歌詞**：收集某歌手的所有歌詞
4. **小說**：某本公版小說

### 步驟

```python
# 1. 準備你的文本（存為 input.txt）
with open("my_data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Character-level tokenization
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"字元數: {vocab_size}")

# 3. 用同樣的方式訓練
# 修改 scripts/train_nanogpt.py 中的 data_path 和 vocab_size
```

### 注意事項
- 文本越大越好（至少幾 MB）
- 中文字元集較大，可能需要調整 vocab_size
- 如果用 BPE，壓縮率更高，效果更好

---

## 練習 4：理解 Attention（視覺化）

### 任務
視覺化 Attention weights，理解模型在「看」什麼。

```python
import matplotlib.pyplot as plt
import torch

def visualize_attention(model, text, layer=0, head=0):
    """視覺化某一層某個 head 的 attention weights"""
    model.eval()

    # encode text
    ids = encode(text)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    # hook to capture attention weights
    attention_weights = []
    def hook_fn(module, input, output):
        # output 是 (B, T, C)，但我們需要中間的 attention weights
        pass  # 需要修改 model 來返回 attention weights

    # 更簡單的方法：手動計算
    with torch.no_grad():
        tok_emb = model.transformer.wte(x)
        pos_emb = model.transformer.wpe(torch.arange(len(ids), device=device))
        x_in = tok_emb + pos_emb

        # 只過前幾層
        for i, block in enumerate(model.transformer.h):
            if i == layer:
                # 手動算 attention
                x_norm = block.ln_1(x_in)
                B, T, C = x_norm.size()
                q, k, v = block.attn.c_attn(x_norm).split(block.attn.n_embd, dim=2)

                nh = block.attn.n_head
                hs = C // nh
                q = q.view(B, T, nh, hs).transpose(1, 2)
                k = k.view(B, T, nh, hs).transpose(1, 2)

                # Attention weights
                att = (q @ k.transpose(-2, -1)) * (1.0 / (hs ** 0.5))
                att = att.masked_fill(
                    torch.triu(torch.ones(T, T, device=device), diagonal=1).bool(),
                    float('-inf')
                )
                att = torch.softmax(att, dim=-1)

                # 取特定 head
                att_head = att[0, head].cpu().numpy()
                break
            x_in = block(x_in)

    # 畫圖
    chars_list = [itos[i] for i in ids]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(att_head, cmap='viridis')
    ax.set_xticks(range(len(chars_list)))
    ax.set_yticks(range(len(chars_list)))
    ax.set_xticklabels(chars_list, rotation=90)
    ax.set_yticklabels(chars_list)
    ax.set_title(f"Layer {layer}, Head {head}")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"attention_L{layer}_H{head}.png")
    plt.show()

# 使用
# visualize_attention(model, "First Citizen", layer=0, head=0)
```

### 觀察
- 某些 head 會聚焦在前一個 token（像 bigram）
- 某些 head 會看更遠的 token
- 某些 head 會特別注意標點符號或換行

---

## 練習 5：載入 GPT-2 預訓練模型（挑戰）

### 任務
載入 OpenAI 的 GPT-2 預訓練模型，不做任何訓練就生成文字。

```python
from model import GPT

# 載入預訓練模型（會自動從 HuggingFace 下載）
model = GPT.from_pretrained('gpt2')
model.eval()
model.to(device)

# 用 tiktoken 做 tokenization
import tiktoken
enc = tiktoken.get_encoding("gpt2")

# 生成
prompt = "In the beginning, there was"
ids = enc.encode(prompt)
x = torch.tensor([ids], dtype=torch.long, device=device)

generated = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=40)
print(enc.decode(generated[0].tolist()))
```

### 挑戰
1. 比較 GPT-2 (124M) 和 GPT-2 Medium (350M) 的生成品質
2. 嘗試不同的 prompt
3. 嘗試 finetune GPT-2 在 Shakespeare 上

---

## 練習 6：Scaling Laws 觀察（數據分析）

### 任務
訓練不同大小的模型，觀察 loss vs 參數量的關係。

```python
import matplotlib.pyplot as plt

configs = [
    ("Tiny",   dict(n_layer=1, n_head=1, n_embd=32)),
    ("Small",  dict(n_layer=2, n_head=2, n_embd=64)),
    ("Medium", dict(n_layer=4, n_head=4, n_embd=128)),
    ("Large",  dict(n_layer=6, n_head=6, n_embd=384)),
    ("XL",     dict(n_layer=8, n_head=8, n_embd=512)),
]

results = []
for name, cfg in configs:
    model = GPT(GPTConfig(vocab_size=vocab_size, block_size=256, **cfg))
    n_params = model.get_num_params()
    # 訓練 1000 步...
    # val_loss = train_and_evaluate(model, 1000)
    # results.append((name, n_params, val_loss))

# 畫 log-log 圖
# params = [r[1] for r in results]
# losses = [r[2] for r in results]
# plt.loglog(params, losses, 'o-')
# plt.xlabel("Parameters")
# plt.ylabel("Validation Loss")
# plt.title("Scaling Laws: Loss vs Parameters")
# plt.savefig("scaling_laws.png")
```

### 思考
- 參數量翻倍，loss 下降多少？
- 這和 Kaplan et al. (2020) "Scaling Laws for Neural Language Models" 的發現一致嗎？

---

## 延伸閱讀

1. [Attention Is All You Need (Vaswani et al. 2017)](https://arxiv.org/abs/1706.03762) - Transformer 原始論文
2. [GPT-2 Paper (Radford et al. 2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [Karpathy's Zero to Hero Series](https://karpathy.ai/zero-to-hero.html)
4. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - 視覺化解說
5. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
6. [nanoGPT repo](https://github.com/karpathy/nanoGPT) - 本教材的參考
7. [nanochat](https://github.com/karpathy/nanochat) - nanoGPT 的進化版（2025）
