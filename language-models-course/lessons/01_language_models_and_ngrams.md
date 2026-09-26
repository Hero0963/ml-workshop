# 第 01 課：語言模型是什麼——機率、交叉熵與 n-gram

> 前置：第 00 課 ｜ 實驗：[`notebooks/01_ngram.ipynb`](../notebooks/01_ngram.ipynb) ｜ 預估時間：2 小時

## 這一課要回答的問題

- 「語言模型」精確地說是什麼東西？它輸出的是什麼？
- 訓練時的 loss、論文裡的 perplexity、nanochat 報的 bits-per-byte，三者是什麼關係？為什麼換了 tokenizer 就不能直接比 loss？
- 在神經網路之前，人們怎麼做語言模型？它卡在哪裡，神經網路又解決了什麼？

---

## 1. 白話版

### 1.1 猜下一個字的遊戲

「今天天氣很 ___」，你大概會猜「好」「熱」「冷」。你心裡其實有一張表：每個可能的字、各有多大的機會。**語言模型就是把這張表做出來的機器**：給它前面的文字，它對「下一個 token」的每一個可能都給一個機率，全部加起來等於 1。

有了「猜下一個」，就能替整段文字打分數：一段話出現的機率 ＝ 第一個字的機率 × 在第一個字之後第二個字的機率 × ……。也能拿來寫字：照機率抽一個、接上去、再抽下一個。ChatGPT 做的事，本質上就是這個遊戲玩得非常好。

### 1.2 驚訝程度

怎麼判斷一個語言模型好不好？拿一段它沒看過的真實文字，看它**有多驚訝**。

- 真實的下一個字，模型給了 0.9 的機率 → 不驚訝，好。
- 只給了 0.001 → 很驚訝，不好。

把「驚訝」定義成 $-\log(\text{機率})$：機率 1 時驚訝 0，機率越小驚訝越大。整段文字的平均驚訝，就是訓練時看到的 **loss（交叉熵）**。

perplexity 是它的另一種說法：perplexity ＝ 10 表示「模型平均起來像是在 10 個同樣可能的選項裡猜」。

### 1.3 最早的做法：數數

最直接的語言模型是**數數**：在一大堆文字裡數「今天天氣很」後面接「好」出現幾次、接「熱」幾次，除一下就是機率。只看前面 $n-1$ 個字的模型叫 **n-gram**。

問題很快出現：前文取得越長，預測越準，但**越長的前文越可能從沒出現過**——沒出現過就數不到，機率變成 0，驚訝變成無限大。這叫**稀疏性**。神經網路的解法是：不要把每種前文當成獨立的格子，而是把相似的前文放在相近的位置，讓它們分享統計（第 02 課的詞向量就是這個想法的第一步）。

---

## 2. 正式版

### 2.1 定義

設詞彙表 $V$，文字是 token 序列 $x_{1:T} = (x_1, \dots, x_T)$。**語言模型**是序列上的機率分布，用連鎖律分解成一連串「下一個 token」的條件分布：

$$p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}) \tag{1.1}$$

這個分解沒有任何假設（連鎖律永遠成立）；模型的工作只是把每一個 $p_\theta(\cdot \mid x_{<t})$ 做好。GPT 類模型一次 forward 就同時輸出所有位置的條件分布（第 04 課的 causal mask 讓這件事成立）。

### 2.2 訓練目標：最大概似 ＝ 最小交叉熵

給定訓練文字，最大化 log-likelihood 等價於最小化每個 token 的平均負 log 機率：

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \tag{1.2}$$

把真實資料的分布記為 $p$，(1.2) 是交叉熵 $H(p, p_\theta)$ 的估計，而

$$H(p, p_\theta) = H(p) + \mathrm{KL}(p \,\|\, p_\theta) \ge H(p) \tag{1.3}$$

KL 散度非負，所以 **loss 的下限是語言本身的熵 $H(p)$**：語言有真正的不確定性（「今天天氣很 ___」本來就有好幾個合理答案），再好的模型也降不到 0。scaling laws（第 07 課）裡那個「不可約的 loss」就是它。

### 2.3 單位：nats、bits、perplexity、bits-per-byte

- 用自然對數，(1.2) 的單位是 **nats**；用 $\log_2$ 則是 **bits**，$1 \text{ nat} = 1/\ln 2 \approx 1.443 \text{ bits}$。
- **perplexity** $= \exp(\mathcal{L}_{\text{nats}}) = 2^{\mathcal{L}_{\text{bits}}}$：「等效的選項個數」。
- **bits-per-byte（bpb）**：把整段文字的總 nats 除以它在 UTF-8 下的**總位元組數**：

$$\text{bpb} = \frac{\sum_t -\ln p_\theta(x_t \mid x_{<t})}{\ln 2 \cdot \sum_t \text{bytes}(x_t)} = \frac{\mathcal{L}_{\text{nats/token}}}{\ln 2 \cdot \overline{\text{bytes/token}}} \tag{1.4}$$

為什麼要 bpb？**per-token 的 loss 和 tokenizer 綁在一起**：詞彙量越大，每個 token 裝的字越多，每個 token 的 loss 自然越高，但這不代表模型比較差。bpb 以「原始文字的位元組」為分母，所以不同 tokenizer 的模型可以公平比較。nanochat 的 leaderboard 用的 `val_bpb` 就是 (1.4)（特殊 token 不計入）。本課程的 `training.evaluate` 也照這個定義算。

> 例：本課 tokenizer 平均每個 token 約 4 個位元組。per-token loss 1.4 nats ≈ 2.0 bits/token ≈ 0.50 bits/byte。

### 2.4 n-gram：馬可夫假設

假設下一個 token 只和前 $n-1$ 個有關：

$$p(x_t \mid x_{<t}) \approx p(x_t \mid x_{t-n+1:t-1}) \tag{1.5}$$

最大概似估計就是數數（$c(\cdot)$ 表示在訓練文字中出現的次數）：

$$\hat p_{\text{ML}}(x_t \mid h) = \frac{c(h, x_t)}{c(h)}, \qquad h = x_{t-n+1:t-1} \tag{1.6}$$

**稀疏性**：前文 $h$ 的可能組合數是 $|V|^{n-1}$，指數成長，而資料只有線性成長。只要驗證資料裡出現一個訓練時沒看過的 $(h, x_t)$，(1.6) 給出 0，(1.2) 變成無限大。

兩種經典補救：

**加 k 平滑**（add-k / Laplace）：每個格子先塞 $k$ 次假計數，

$$\hat p_k(x_t \mid h) = \frac{c(h, x_t) + k}{c(h) + k|V|} \tag{1.7}$$

**插值**：把不同長度前文的估計混在一起，長前文有資料時用它，沒有時自動退回短前文。記 $h_j$ 為最後 $j - 1$ 個 token（$j$ 階模型的前文），從均勻分布 $\hat p_0 = 1/|V|$ 開始一階一階往上疊：

$$\hat p_j(x_t \mid h_j) = \lambda(h_j)\, \hat p_{\text{ML}}(x_t \mid h_j) + \left(1 - \lambda(h_j)\right) \hat p_{j-1}(x_t \mid h_{j-1}) \tag{1.8}$$

$\lambda$ 是「相信這個前文多少」。若它是固定常數，就是 Jelinek–Mercer 插值；**Witten–Bell** 讓它依前文而定：

$$\lambda(h) = \frac{c(h)}{c(h) + u(h)} \tag{1.9}$$

$u(h)$ 是在 $h$ 之後出現過幾種**不同**的 token。直觀：前文看過很多次、後面接的東西又很固定，就相信它；前文很少見、或後面什麼都可能接（$u$ 大），就多分一點給短前文。沒看過的前文 $c(h) = 0$，$\lambda = 0$，整個交給短前文。
更講究的 Kneser–Ney 平滑在神經網路之前是業界標準；CCNet 等資料管線到今天還用 n-gram 模型的 perplexity 當品質過濾器（第 10 課）。

### 2.5 從數數到梯度下降

把 bigram 模型寫成一張可學的表 $W \in \mathbb{R}^{|V| \times |V|}$：$p(x_t = j \mid x_{t-1} = i) = \mathrm{softmax}(W_{i,:})_j$。對 (1.2) 做梯度下降，最佳解正是 (1.6) 的計數比例（第 $i$ 列的 softmax 等於第 $i$ 列的經驗分布時梯度為 0）。**所以神經網路和數數在「表格模型」上是同一件事**，差別在於神經網路可以換成別的函數形式：

- Bengio 等人（2003）把前文的每個詞先查一個**向量**（embedding），再用 MLP 算出下一個詞的分布。相似的詞有相似的向量，於是「今天天氣很好」學到的東西會自動用到「明天天氣很好」——這就是對稀疏性的回答。
- 把「固定長度的前文」換成「整段前文」，就走到了 RNN，再走到 Transformer（第 04 課）。

### 2.6 用語言模型生成

按 (1.1) 一個一個抽：$x_t \sim p_\theta(\cdot \mid x_{<t})$。抽樣時常用溫度、top-k、top-p 修改分布（第 05 課）。n-gram 的生成很能說明問題：$n$ 小時字串很亂；$n$ 大時，最大概似的 n-gram 只會走訓練資料裡走過的路，每一段 $n$ 個 token 都原封不動來自訓練文字，生成的是**拼貼**——局部通順，整體跳來跳去；前文長到幾乎只出現過一次時，就會整段背出來。**「低 loss」和「會創造」是兩件事**，實驗裡會看到。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| (1.6)(1.7) 加 k 平滑的 n-gram | `ngram.AddKLM`（計數用 `ngram.NGramCounts`：把前文雜湊成一個整數，排序後二分搜尋） |
| (1.8)(1.9) Witten–Bell 插值 | `ngram.InterpolatedLM`（`up_to(n)` 只用前 $n$ 階，共用同一份計數） |
| 每位元組的 bits | `ngram.bits_per_byte` |
| (1.2)(1.4) 神經網路的 loss 與 bpb | `training.lm_loss`、`training.evaluate(..., token_bytes=...)` |
| 每個 token 的位元組數 | `tokenizer.BPETokenizer.token_byte_lengths`（特殊 token 記 0） |

`tests/test_ngram.py` 裡值得看的兩個：`test_unigram_bits_per_byte_is_the_byte_entropy`（unigram 的 MLE 在訓練資料上的 bpb 恰好等於位元組分布的熵）、`test_huge_smoothing_gives_eight_bits_per_byte`（平滑到極致就是均勻分布：每個位元組 8 bits）。

## 4. 常見誤解

- **「loss 越低越好，所以 n 越大越好」**：在訓練資料上是，在沒看過的資料上不是。Lab 01 會畫出驗證 bpb 對 $n$ 的 U 形曲線。
- **「兩個模型的 per-token loss 可以直接比」**：只有 tokenizer 相同才行；否則比 bpb。
- **「perplexity 10 表示模型在 10 個字裡猜」**：這是平均意義上的等效選項數，不是模型真的只考慮 10 個候選。
- **「語言模型只會預測下一個字，所以不會『理解』」**：預測下一個字的目標，會逼模型學到所有能幫助預測的東西（文法、事實、推理）。這是 GPT-2 論文標題「Language Models are Unsupervised Multitask Learners」的主張（第 05 課）。

## 5. 練習

**想一想**

1. 證明 (1.3)：$H(p, q) = H(p) + \mathrm{KL}(p \| q)$。
2. 一個模型對 50,257 個 token 給均勻分布。它的 per-token loss 是多少 nats？perplexity 是多少？
3. 同一段文字，tokenizer A 平均 3 bytes/token、loss 1.2 nats/token；tokenizer B 平均 5 bytes/token、loss 1.8 nats/token。哪個模型比較好？
4. 從 (1.7) 出發，說明 $k \to \infty$ 時模型變成什麼；$k \to 0$ 時又變成什麼。
5. 證明 §2.5 的說法：softmax 表格模型的交叉熵梯度為 0 時，第 $i$ 列的機率等於計數比例。

**動手改**（在 `01_ngram.ipynb`）

6. 把加 k 平滑的 $k$ 從 0.01 掃到 1，每個 $n$ 的最佳 $k$ 一樣嗎？
7. 把 (1.9) 換成固定的 $\lambda$（Jelinek–Mercer），試幾個值：驗證 bpb 最好能到多少？和 Witten–Bell 比呢？
8. 把驗證資料換成一段 Python 程式碼（或中文），bpb 會怎麼變？為什麼？

## 6. 延伸閱讀

- CS336 第 1 講（overview、tokenization）與整門課的課綱：<https://cs336.stanford.edu/>
- Jurafsky & Martin，《Speech and Language Processing》第 3 版草稿第 3 章〈N-gram Language Models〉：<https://web.stanford.edu/~jurafsky/slp3/>——perplexity、平滑、Kneser–Ney 最完整的教科書講法。
- Bengio 等人（2003），〈A Neural Probabilistic Language Model〉，*JMLR* 3：<https://www.jmlr.org/papers/v3/bengio03a.html>
- Shannon（1951），〈Prediction and Entropy of Printed English〉，*Bell System Technical Journal* 30(1)——「讓人猜下一個字母」估計英文熵的經典實驗。
