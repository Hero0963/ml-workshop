# 第 04 課：Attention 與 Transformer 區塊

> 前置：第 01–03 課 ｜ 實驗：[`notebooks/04_attention.ipynb`](../notebooks/04_attention.ipynb) ｜ 預估時間：2.5 小時

## 這一課要回答的問題

- attention 在算什麼？為什麼要除以 $\sqrt{d_k}$？causal mask 為什麼讓「一次 forward 訓練所有位置」成立？
- attention 本身不知道詞的順序——位置資訊怎麼加進去？RoPE 的「相對位置」性質從哪裡來？
- 一個 Transformer 區塊由哪些零件組成？LayerNorm vs RMSNorm、GELU vs SwiGLU vs ReLU²、MHA vs GQA 各在解決什麼？

---

## 1. 白話版

### 1.1 每個字去「查詢」前面的字

讀到「牠追著球跑，因為**牠**很興奮」的第二個「牠」，你會回頭找「牠」指的是誰。attention 就是這個動作：

- 每個位置發出一個**問題**（query）：「我在找什麼？」
- 每個位置掛著一個**標籤**（key）：「我是什麼？」
- 問題和標籤越吻合，就從那個位置拿越多**內容**（value）。

拿到的是前面各位置內容的**加權平均**，權重由吻合程度經 softmax 決定。

### 1.2 不能偷看後面

訓練時整段文字都在手上，但位置 $t$ 在預測第 $t+1$ 個字時，不能看到第 $t+1$ 個字本身。**causal mask** 把「看未來」的權重設成 0。這樣一次 forward 就能同時訓練所有位置：每個位置都只用了它該用的資訊。

### 1.3 attention 不知道順序

加權平均不在乎誰先誰後：把輸入打亂，輸出只是跟著打亂。但「狗咬人」和「人咬狗」意思不同，所以要另外把位置告訴模型。GPT-2 的做法是替每個位置學一個向量加上去；現代模型用 **RoPE**：把 query 和 key 依照位置**旋轉**一個角度，兩個位置的吻合程度就只和「相隔多遠」有關。

### 1.4 一個區塊 = 溝通 + 思考

Transformer 區塊有兩步：**attention 讓各位置互相交換資訊**，**MLP 在每個位置各自加工**。兩步都是「在原本的東西上加一點修正」（residual），像一本大家輪流在上面寫筆記的共用筆記本（residual stream）。每一步之前先做一次正規化，讓數字的大小維持穩定。

---

## 2. 正式版

### 2.1 Scaled dot-product attention

輸入 $X \in \mathbb{R}^{T \times d}$（$T$ 個位置、每個 $d$ 維）。投影出 $Q = XW_Q$、$K = XW_K$、$V = XW_V$（$d_k$ 維），

$$\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V \tag{4.1}$$

**為什麼除以 $\sqrt{d_k}$**：若 $q$、$k$ 的每個分量獨立、平均 0、變異數 1，則

$$\mathrm{Var}(q^\top k) = \sum_{i=1}^{d_k}\mathrm{Var}(q_i k_i) = d_k \tag{4.2}$$

不縮放的話，分數的尺度隨維度長大，softmax 會飽和成近乎 one-hot，梯度幾乎為 0。除以 $\sqrt{d_k}$ 讓分數的變異數回到 1。

### 2.2 Causal mask

$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

$e^{-\infty} = 0$，所以位置 $i$ 只看得到 $1..i$。模型在位置 $i$ 輸出「第 $i+1$ 個 token 的分布」，(1.2) 的 loss 對所有位置一起算——**一次 forward 就得到 $T$ 個訓練訊號**。這叫 teacher forcing：訓練時每一步都餵真實的前文，而不是模型自己生成的。

### 2.3 Multi-head attention

把 $d$ 維切成 $H$ 個 head，每個 $d_h = d / H$ 維，各自做 (4.1)，再串接、乘上輸出投影：

$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_H)\, W_O \tag{4.3}$$

不同 head 可以學不同的「查詢方式」（一個看前一個字、一個看句首的主詞…）。參數量：$W_Q, W_K, W_V, W_O$ 各 $d \times d$，共 $4d^2$。

### 2.4 attention 對排列是等變的

設 $P$ 是一個排列矩陣。沒有 mask、沒有位置資訊時，$\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)$：打亂輸入，輸出只是跟著打亂。所以位置資訊必須另外提供。
（有 causal mask 時，對稱性被打破：位置 $i$ 看得到 $i$ 個 token，模型其實可以從「看到幾個東西」推出位置。Haviv 等人 2022 發現完全不加位置編碼的 causal LM 也能學到位置資訊。）

### 2.5 位置資訊

| 方法 | 做法 | 用在 |
|---|---|---|
| 可學的絕對位置 | 每個位置 $m$ 一個向量 $p_m$，加到 token embedding | GPT-2（1024 個位置） |
| 正弦絕對位置 | 固定的 $\sin/\cos$ 函數 | 原始 Transformer |
| **RoPE** | 把 $q$、$k$ 依位置旋轉 | Llama、Qwen、nanochat… |
| ALiBi | 在分數上加一個隨距離線性遞減的偏差 | BLOOM 等 |

**RoPE**（Su 等人 2021）：把 $d_h$ 維切成 $d_h/2$ 對，第 $i$ 對在位置 $m$ 旋轉角度 $m\theta_i$，$\theta_i = b^{-2i/d_h}$（$b$ 通常 10,000）：

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} \tag{4.4}$$

旋轉矩陣滿足 $R_m^\top R_n = R_{n-m}$，所以

$$\langle R_m q,\ R_n k\rangle = q^\top R_m^\top R_n k = \langle q,\ R_{n-m}k\rangle \tag{4.5}$$

**分數只和相對距離 $n - m$ 有關**。另外，旋轉不改變長度，也不需要任何參數。
實作上有兩種配對法：原論文配相鄰的兩維 $(2i, 2i+1)$；GPT-NeoX、Llama（以及本課程）配前半與後半 $(i, i + d_h/2)$。數學性質相同，但兩者的權重不能混用。nanochat 用 $b = 100{,}000$（較大的 base 讓低頻維度轉得更慢，對長 context 有利）。

### 2.6 區塊結構：pre-norm residual

$$x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x)), \qquad x \leftarrow x + \mathrm{MLP}(\mathrm{Norm}(x)) \tag{4.6}$$

原始 Transformer 把 norm 放在相加之後（post-norm）；GPT-2 起改成相加之前（pre-norm），最後再加一個 final norm。Xiong 等人（2020）分析：post-norm 在輸出層附近的梯度很大，必須靠 learning-rate warmup 才訓練得起來；pre-norm 的梯度穩定得多。
residual 讓每一層的輸出是對「residual stream」的**增量**，梯度也能沿著恆等路徑直接傳到淺層。

### 2.7 正規化：LayerNorm 與 RMSNorm

$$\mathrm{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad \mu = \tfrac{1}{d}\textstyle\sum_i x_i,\ \sigma^2 = \tfrac{1}{d}\textstyle\sum_i (x_i - \mu)^2 \tag{4.7}$$

$$\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \tag{4.8}$$

RMSNorm（Zhang & Sennrich 2019）拿掉減平均與 $\beta$：少一次歸約運算，實驗上品質相當。nanochat 更進一步連 $\gamma$ 都不要（`norm_affine=False`）。

### 2.8 MLP 與激活函數

$$\text{GELU MLP: } W_2\,\mathrm{GELU}(W_1 x), \quad W_1 \in \mathbb{R}^{4d \times d} \tag{4.9}$$

$$\text{SwiGLU: } W_2\left(\mathrm{SiLU}(W_g x) \odot W_1 x\right), \quad W_1, W_g \in \mathbb{R}^{d_{ff} \times d},\ d_{ff} \approx \tfrac{8}{3}d \tag{4.10}$$

SwiGLU（Shazeer 2020）多一個「閘」矩陣；把寬度從 $4d$ 降到 $\tfrac{8}{3}d$，三個矩陣的參數量就和兩個 $4d$ 矩陣相同。**ReLU²**（$\max(0, x)^2$，So 等人 2021 的 Primer）是 nanochat 的選擇：比 GELU 便宜，實驗上不輸。GPT-2 用 GELU 的 tanh 近似。

### 2.9 attention 的變體

- **MQA／GQA**（Shazeer 2019；Ainslie 等人 2023）：$H$ 個 query head 共用 $H_{kv} < H$ 組 key／value。參數少一點，更重要的是推論時的 KV cache 小 $H / H_{kv}$ 倍（第 09 課）。
- **QK-norm**：對每個 head 的 $q$、$k$ 先做 RMSNorm 再算分數，避免分數在訓練中爆大（ViT-22B 用來穩定大模型訓練；nanochat 採用）。
- **滑動視窗**：部分層只看最近 $w$ 個 token（nanochat 預設「三短一長」的層級模式）。

### 2.10 參數量

一層（MHA、$4d$ 的 GELU MLP、不計 bias 與 norm）：

$$\underbrace{4d^2}_{\text{attention}} + \underbrace{8d^2}_{\text{MLP}} = 12d^2 \tag{4.11}$$

$L$ 層共 $12Ld^2$——Kaplan 等人（2020）的「非 embedding 參數量」近似。GPT-2 small：$12 \times 12 \times 768^2 \approx 85\text{M}$，加上 embedding（$50257 \times 768 \approx 39\text{M}$）與位置 embedding，就是 124M。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/model.py`） |
|---|---|
| (4.1) 寫開的公式（供測試對照） | `attention_reference` |
| (4.1)(4.3) 實際使用的 attention（PyTorch 的 fused `scaled_dot_product_attention`）、GQA、QK-norm | `Attention` |
| (4.4) RoPE | `rope_cache`、`apply_rope`（前半／後半配對） |
| (4.6) pre-norm 區塊 | `Block` |
| (4.7)(4.8) | `nn.LayerNorm`、`RMSNorm`；`make_norm` 依 config 選 |
| (4.9)(4.10) 與 ReLU² | `MLP`（`mlp="gelu" / "swiglu" / "relu2"`） |
| 整體設定 | `GPTConfig`；`gpt2_config`、`llama_style_config`、`nanochat_style_config` |

測試：`test_rope_scores_depend_only_on_relative_position`（式 4.5）、`test_attention_module_matches_the_written_out_formula`、`test_causal_logits_ignore_future_tokens`、`test_grouped_query_attention_shrinks_the_kv_projections`。

## 4. 常見誤解

- **「attention 權重就是模型的解釋」**：權重只說明從哪裡拿資訊，不代表那些資訊怎麼被使用；多層、多 head 的組合更難直接解讀。
- **「causal mask 是為了生成」**：它首先是為了**訓練**——讓一次 forward 的每個位置都不偷看答案。生成時只是順便一致。
- **「RoPE 是加在 embedding 上的」**：它作用在每一層的 $q$、$k$ 上，不碰 $v$，也不碰 residual stream。
- **「多一個 head 就多一份參數」**：head 數不改變參數量（$4d^2$ 固定），只改變如何切分 $d$。

## 5. 練習

**想一想**

1. 證明 (4.2)。如果 $q$、$k$ 的分量變異數是 $\sigma^2$，縮放因子應該是多少？
2. 證明 §2.4 的排列等變性。加了 causal mask 之後為什麼不成立？
3. 用 (4.4) 驗證 $R_m^\top R_n = R_{n-m}$（只需看一個 $2 \times 2$ 區塊）。
4. SwiGLU 的 $d_{ff}$ 取多少時，參數量剛好等於 $4d$ 的 GELU MLP？
5. GQA 把 $H_{kv}$ 從 12 降到 4，attention 的參數量少了幾成？KV cache 少了幾成？

**動手改**（在 `04_attention.ipynb`）

6. 把 RoPE 的 base 從 10,000 改成 100 與 1,000,000，「分數隨距離的變化」圖會怎麼變？
7. 拿掉 $\sqrt{d_k}$ 縮放，在 $d_k = 256$ 時 attention 權重的熵是多少？
8. 驗證 GQA 版本的 `Attention` 在 $H_{kv} = H$ 時與一般 MHA 完全相同。

## 6. 延伸閱讀

- Vaswani 等人（2017），〈Attention Is All You Need〉：<https://arxiv.org/abs/1706.03762>
- Bahdanau 等人（2014），attention 的起源（機器翻譯）：<https://arxiv.org/abs/1409.0473>
- Su 等人（2021），RoFormer／RoPE：<https://arxiv.org/abs/2104.09864>
- Xiong 等人（2020），pre-LN vs post-LN：<https://arxiv.org/abs/2002.04745>
- Zhang & Sennrich（2019），RMSNorm：<https://arxiv.org/abs/1910.07467>
- Shazeer（2020），GLU 變體：<https://arxiv.org/abs/2002.05202>
- Ainslie 等人（2023），GQA：<https://arxiv.org/abs/2305.13245>
- Karpathy，〈Let's build GPT: from scratch, in code, spelled out.〉（影片）：<https://www.youtube.com/watch?v=kCc8FmEb1nY>
- CS336 第 3 講（架構與超參數，Tatsunori Hashimoto）：<https://cs336.stanford.edu/>——各家開源模型的架構選擇比較表是這門課最實用的內容之一。
