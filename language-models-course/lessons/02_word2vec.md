# 第 02 課：詞向量——word2vec 與分布假說

> 前置：第 01 課 ｜ 實驗：[`notebooks/02_word2vec.ipynb`](../notebooks/02_word2vec.ipynb) ｜ 預估時間：2.5 小時

## 這一課要回答的問題

- 怎麼把「詞」變成電腦能計算的向量，而且讓意思相近的詞向量也相近？
- word2vec 的 skip-gram 與 negative sampling 到底在最佳化什麼？
- 為什麼「king − man + woman ≈ queen」會成立？什麼時候不成立？
- word2vec 和今天的 GPT、embedding 模型有什麼關係？

---

## 1. 白話版

### 1.1 看一個字和誰在一起

語言學家 Firth（1957）有一句名言：「**看一個詞和誰作伴，就知道它是什麼意思**。」
「我養了一隻 ___，每天帶牠去散步」——空格可以填「狗」，也可以填「柯基」。兩個詞常出現在相同的上下文，意思就相近。這叫**分布假說**。

### 1.2 從編號到向量

電腦最直接的表示法是給每個詞一個編號（one-hot：一個很長、只有一格是 1 的向量）。但編號之間沒有遠近：「狗」和「貓」的距離，跟「狗」和「民主」一樣遠。

word2vec 的想法：給每個詞一個**短而密的向量**（例如 50 維），然後玩一個遊戲——**看到一個詞，猜它旁邊會出現哪些詞**。為了猜得準，常出現在相同鄰居旁的詞，向量會被推到相近的位置。遊戲本身不重要，重要的是玩完之後留下的向量。

### 1.3 一個省力的技巧：拿假的鄰居來比

「猜鄰居」若要對詞彙表裡每個詞都打分數，詞彙有幾十萬個就太慢了。negative sampling 改成一個是非題：「這一對（中心詞、鄰居）是真的從文章裡拿出來的，還是隨便配的？」
每個真實的配對，只要跟幾個隨機抽的假配對比較就好，速度快了幾千倍。

### 1.4 方向有意義

訓練完會發現，向量空間裡的「方向」也有意思：從 man 走到 woman 的那個位移，和從 king 走到 queen 的位移差不多。於是 king − man + woman 會落在 queen 附近。
這不是有人教它的，是「he／she 出現在旁邊的頻率」這種統計規律，自然在向量裡變成了一個方向。

---

## 2. 正式版

### 2.1 符號

詞彙表 $V$，每個詞 $w$ 有兩個向量：**中心向量** $v_w \in \mathbb{R}^d$（輸入表）與**上下文向量** $u_w \in \mathbb{R}^d$（輸出表）。
查表 $v_w = E^\top e_w$，$e_w$ 是 one-hot——**embedding 層就是一個乘 one-hot 的矩陣**，所以可以用梯度訓練。

### 2.2 Skip-gram（Mikolov 等人 2013a）

語料 $w_1, \dots, w_T$、視窗大小 $c$。最大化「中心詞預測鄰居」的平均 log 機率：

$$\frac{1}{T}\sum_{t=1}^{T}\ \sum_{-c \le j \le c,\ j \ne 0} \log p(w_{t+j} \mid w_t) \tag{2.1}$$

$$p(o \mid c) = \frac{\exp(u_o^\top v_c)}{\sum_{w \in V} \exp(u_w^\top v_c)} \tag{2.2}$$

分母要對整個詞彙表求和，每一對的計算量是 $O(|V|)$。同一篇論文的另一個模型 **CBOW** 反過來，用鄰居向量的平均去猜中心詞。

### 2.3 Negative sampling（Mikolov 等人 2013b）

把 (2.2) 的 log 機率換成一個二元分類目標：真實配對 $(c, o)$ 要被判成「真」，從雜訊分布抽的 $K$ 個詞要被判成「假」：

$$\log \sigma(u_o^\top v_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n}\left[\log \sigma(-u_{w_k}^\top v_c)\right] \tag{2.3}$$

$\sigma$ 是 sigmoid。雜訊分布用 unigram 的 3/4 次方：

$$P_n(w) = \frac{U(w)^{3/4}}{Z} \tag{2.4}$$

3/4 次方把高頻詞壓低、低頻詞抬高一點；論文說它明顯比 unigram 與均勻分布好。$K$ 的建議：小資料 5–20，大資料 2–5。每一對的計算量從 $O(|V|)$ 變成 $O(K)$。

### 2.4 兩個實務技巧

**高頻詞降採樣**（2013b）：每次出現的詞 $w$ 以機率

$$P_{\text{discard}}(w) = 1 - \sqrt{t / f(w)} \tag{2.5}$$

被丟掉，$f(w)$ 是相對頻率，$t$ 約 $10^{-5}$。「the」「a」這種詞提供的資訊少，丟掉大部分能加速訓練，也讓稀有詞的向量更好。

**動態視窗**：每個中心詞先從 $1..c$ 抽一個實際視窗大小，近的鄰居因此比遠的鄰居更常被配對。

### 2.5 類比為什麼會成立

類比題 $a : b :: c : ?$ 的標準解法（3CosAdd）：

$$\hat d = \arg\max_{x \notin \{a, b, c\}} \cos\left(v_x,\ v_b - v_a + v_c\right) \tag{2.6}$$

直觀的理由來自 GloVe（Pennington 等人 2014）的觀察：如果對所有上下文詞 $k$，

$$\frac{P(k \mid \text{king})}{P(k \mid \text{queen})} \approx \frac{P(k \mid \text{man})}{P(k \mid \text{woman})}$$

（例如 $k$ = he、she、his、her 時比例很極端，$k$ = crown 時兩邊都約等於 1），那麼在 log 空間裡「king − queen」和「man − woman」就是同一個位移。**向量的線性結構，來自共現機率比值的一致性**。

它也說明了類比什麼時候會失敗：語料裡如果 son 和 boy 的上下文完全相同，模型根本分不出兩者，「king : queen :: son : ?」答 niece 或 girl 也無可厚非（Lab 02 §2 會親眼看到）。另外，(2.6) 排除 $a, b, c$ 本身很關鍵：不排除的話，答案常常就是 $c$（Nissim 等人 2020 對此有詳細批評）。

### 2.6 SGNS 其實在分解一個矩陣（Levy & Goldberg 2014）

把 (2.3) 對整個語料加總，令 $\#(w, c)$ 為共現次數、$\#w$ 為中心詞 $w$ 的配對數。對單一格子 $x = v_w^\top u_c$，期望目標是

$$\ell(x) = \#(w,c)\log\sigma(x) + K\cdot\#w\cdot P_n(c)\log\sigma(-x) \tag{2.7}$$

對 $x$ 微分設為 0（用 $\sigma'(x) = \sigma(x)\sigma(-x)$）得到 $e^{x} = \frac{\#(w,c)}{K\,\#w\,P_n(c)}$。若 $P_n$ 取 unigram（$P_n(c) = \#c / |D|$）：

$$v_w^\top u_c = \log\frac{\#(w,c)\,|D|}{\#w\,\#c} - \log K = \mathrm{PMI}(w, c) - \log K \tag{2.8}$$

也就是說：**只要維度夠，SGNS 的最佳解就是 shifted PMI 矩陣**。這把「預測式」的 word2vec 和「計數式」的傳統方法接在一起：直接算正的 shifted PMI 矩陣再做 SVD，

$$M = \max(\mathrm{PMI} - \log K, 0), \qquad W = U_d \sqrt{\Sigma_d} \tag{2.9}$$

就能得到同一類向量。差別在於 SGNS 用低維度去近似（相當於加權的低秩分解，會多看高頻格子），而 SVD 對每一格一視同仁。Levy、Goldberg 與 Dagan（2015）進一步指出，兩派的表現差距大多來自超參數（視窗、降採樣、3/4 次方平滑…），而不是演算法本身。

### 2.7 兩張表，以及和 GPT 的關係

- word2vec 有**輸入表**（中心向量）和**輸出表**（上下文向量）。GPT 也有：token embedding 表是輸入表，LM head 的權重是輸出表（下一個 token 的 logit $= u_w^\top h$）。
- **Weight tying**（Press & Wolf 2016）讓兩張表共用同一個矩陣，GPT-2 就這樣做；nanochat 與多數新模型則不綁（第 05、06 課）。
- word2vec 是**靜態**向量：「bank」不管是河岸還是銀行都只有一個向量。Transformer 的隱藏狀態是**依上下文而變**的向量（第 04 課）；第 13 課再把整段文字壓成一個向量。

### 2.8 限制

- **一詞一向量**：多義詞被平均掉。
- **沒看過的詞沒有向量**：fastText（Bojanowski 等人 2016）用字元 n-gram 的向量相加來組出新詞——和 BPE（第 03 課）解決的是同一個問題。
- **會學到語料裡的偏見**：Bolukbasi 等人（2016）的經典例子「man : computer programmer :: woman : homemaker」。
- **相似 ≠ 同義**：反義詞常出現在相同上下文（hot／cold），所以向量也相近。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/word2vec.py`） |
|---|---|
| 詞彙表、編碼 | `build_vocab`、`encode_sentences` |
| (2.5) 降採樣、(2.4) 雜訊分布 | `keep_probabilities`、`noise_distribution` |
| 動態視窗的 skip-gram 配對 | `skipgram_pairs` |
| (2.3) SGNS 損失 | `SkipGram.loss`（兩張表：`center`、`context`，初始化照原始工具：中心向量小亂數、上下文向量 0） |
| 訓練 | `train_skipgram`（Adam、批次；原始工具是一次一對的 SGD） |
| (2.7) 全語料的封閉形式目標 | `sgns_objective` |
| (2.8)(2.9) PMI、shifted PPMI、SVD | `pmi_matrix`、`shifted_ppmi`、`svd_embeddings` |
| (2.6) 類比 | `solve_analogy`、`analogy_accuracy` |
| 有已知結構的玩具語料 | `data.toy_world_corpus`、`data.toy_world_analogies` |

最值得讀的測試：`tests/test_word2vec.py::test_sgns_optimum_is_shifted_pmi`——用滿秩的向量最佳化 (2.7)，檢查所有內積收斂到 $\mathrm{PMI} - \log K$，誤差在 0.05 以內。

## 4. 常見誤解

- **「word2vec 是深度學習」**：它只有兩張表、沒有隱藏層，是一個淺層的對數雙線性模型。它的影響力來自速度和規模。
- **「類比準確率高，表示向量理解了語意」**：3CosAdd 排除了輸入詞，而且類比資料集集中在少數幾種高度規律的關係（國家—首都、單複數）。
- **「餘弦相似度高 ＝ 同義」**：只代表上下文分布相似，反義詞、同一類別的並列詞（週一／週二）都會很像。
- **「用中心向量或上下文向量都一樣」**：兩者不同；常見做法是用中心向量，或兩者相加。

## 5. 練習

**想一想**

1. 從 (2.7) 推出 (2.8)。如果雜訊分布用 unigram 的 3/4 次方，(2.8) 要怎麼改？
2. 為什麼 (2.2) 的 softmax 每一對要花 $O(|V|)$？negative sampling 為什麼可以不算分母？
3. (2.5) 在 $t = 10^{-5}$ 時，一個相對頻率 5% 的詞每次出現被丟掉的機率是多少？
4. 用 §2.5 的比值論證，解釋為什麼「巴黎 : 法國 :: 東京 : 日本」需要語料裡有「首都」類的上下文。
5. GPT 的 LM head 為什麼可以看成 word2vec 的「上下文向量表」？weight tying 等於假設了什麼？

**動手改**（在 `02_word2vec.ipynb`）

6. 把玩具語料的 `ROLES` 拿掉（每個人物只剩性別與年齡的上下文），性別類比的準確率會掉到多少？錯的答案是什麼？
7. 改變視窗大小 1、3、5：哪一類類比（性別、首都、語言）最受影響？為什麼？
8. 在 TinyStories 上找幾個「反義詞很像」的例子。

## 6. 延伸閱讀

- Mikolov 等人（2013a），〈Efficient Estimation of Word Representations in Vector Space〉：<https://arxiv.org/abs/1301.3781>
- Mikolov 等人（2013b），〈Distributed Representations of Words and Phrases and their Compositionality〉：<https://arxiv.org/abs/1310.4546>——negative sampling、降採樣、片語。
- Goldberg & Levy（2014），〈word2vec Explained〉：<https://arxiv.org/abs/1402.3722>——兩頁推完 SGNS 的目標函數。
- Levy & Goldberg（2014），〈Neural Word Embedding as Implicit Matrix Factorization〉，NeurIPS：<https://proceedings.neurips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html>
- Levy、Goldberg、Dagan（2015），〈Improving Distributional Similarity with Lessons Learned from Word Embeddings〉，TACL：<https://aclanthology.org/Q15-1016/>
- Pennington 等人（2014），GloVe：<https://aclanthology.org/D14-1162/>
- Bojanowski 等人（2016），fastText：<https://arxiv.org/abs/1607.04606>
- Jurafsky & Martin，SLP3 第 6 章〈Vector Semantics and Embeddings〉：<https://web.stanford.edu/~jurafsky/slp3/>
