# 第 13 課：Embedding 模型——從詞向量到句向量

> 前置：第 02、06 課 ｜ 實驗：[`notebooks/13_embeddings.ipynb`](../notebooks/13_embeddings.ipynb)（需要 Lab 06 的模型） ｜ 預估時間：2.5 小時

## 這一課要回答的問題

- 怎麼把**一整段文字**變成一個向量，讓意思相近的文字向量也相近？
- 對比學習（InfoNCE）在最佳化什麼？為什麼「同一批裡的其他樣本」就能當負例？溫度參數在做什麼？
- 為什麼現在最強的 embedding 模型都拿大型語言模型當骨幹？要取哪個位置的向量？
- 怎麼評估 embedding 模型？為什麼 BM25 這種 1990 年代的方法還是很難打敗？

---

## 1. 白話版

### 1.1 從一個字到一段話

第 02 課的 word2vec 替每個**字**學一個向量。但我們常需要比較的是**一段話**：「搜尋時哪篇文章最能回答這個問題？」「這兩則客訴講的是不是同一件事？」
embedding 模型把任意長度的文字變成一個固定長度的向量，用兩個向量的夾角（餘弦相似度）表示意思有多接近。有了它，搜尋就變成「找最近的向量」。

### 1.2 用配對來教

怎麼教模型「意思相近」？給它很多**正確的配對**：問題與答案、標題與內文、同一篇文章的兩個片段。訓練的目標是：每一對的兩個向量要靠近，和**別人的**向量要遠離。
一批 64 對資料，對第 1 個問題來說，第 1 個答案是正解，其他 63 個答案都是「錯的」——不用另外找負例，同一批的其他資料就是。這叫 **in-batch negatives**。

### 1.3 拿語言模型當骨幹

預訓練過的語言模型已經很懂文字。把它每個位置的輸出向量**取平均**，或取**最後一個 token** 的向量（它看過整段文字），再用配對資料微調，就是一個很好的 embedding 模型。2024 年起排行榜上的強者大多這樣做。

### 1.4 老方法依然強

BM25 只看「查詢的字有沒有出現在文件裡、出現幾次、這個字罕不罕見」，完全不懂意思。但很多搜尋任務的關鍵字就是答案，所以它是一條很難超越的基線；實務上常把兩者合用（hybrid search）。

---

## 2. 正式版

### 2.1 設定

encoder $f_\theta$ 把文字映成單位向量，相似度 $s(a, b) = f_\theta(a)^\top f_\theta(b)$。兩種架構：

- **bi-encoder**：查詢與文件各自編碼，文件向量可以事先算好、建索引（第 2.6 節），適合大規模檢索。
- **cross-encoder**：把查詢與文件串在一起送進模型，直接輸出分數。更準，但每一對都要跑一次，只適合對少量候選重排（reranking）。

### 2.2 Pooling：一段文字取哪個向量

Transformer 輸出每個 token 一個隱藏狀態 $h_1, \dots, h_T$：

- **mean pooling**：$\frac{1}{T}\sum_t h_t$（Sentence-BERT 的預設）；
- **[CLS]**：雙向 encoder（BERT）在開頭放一個特殊 token，取它的輸出；
- **last token**：causal 的語言模型只有最後一個位置看過整段文字，所以取它（E5-Mistral、Qwen3-Embedding 的做法，通常在結尾加一個固定的 token）。

另一條路是把 causal 模型改成雙向 attention 再訓練（LLM2Vec、NV-Embed）。本課程的 `GPTConfig(causal=False)` 可以做到這件事。

### 2.3 InfoNCE 與 in-batch negatives

一批 $B$ 對 $(q_i, d_i)$，向量都已正規化，溫度 $\tau$：

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp\left(s(q_i, d_i)/\tau\right)}{\sum_{j=1}^{B}\exp\left(s(q_i, d_j)/\tau\right)} \tag{13.1}$$

就是「從 $B$ 個候選裡挑出正解」的 $B$ 類交叉熵（van den Oord 等人 2018 稱為 InfoNCE）；常再加上反方向（從 $d_i$ 找 $q_i$）取平均。

- **下限與上限**：完全隨機的向量給出 $\log B$；完美分開時趨近 0（`tests/test_embeddings.py` 驗證這兩個極端）。
- **batch 越大、負例越多**，任務越難、訊號越豐富——這是對比學習偏好大 batch 的原因。
- **溫度 $\tau$**：餘弦相似度在 $[-1, 1]$，除以 0.05 後差距被放大 20 倍，softmax 才夠「尖」。$\tau$ 太大學不動，太小則對少數困難負例過度敏感。
- **hard negatives**：刻意找「很像但不是正解」的負例（例如 BM25 搜出來排名很前面的錯誤文件），比隨機負例更有訓練價值（DPR、ANCE）。

### 2.4 對齊與均勻（Wang & Isola 2020）

(13.1) 可以拆成兩股力量：

$$\ell_{\text{align}} = \mathbb{E}_{(x, y)\ \text{正例}}\|f(x) - f(y)\|^2, \qquad \ell_{\text{uniform}} = \log \mathbb{E}_{x, y}\ e^{-t\|f(x) - f(y)\|^2} \tag{13.2}$$

好的 embedding 要**正例靠近**（alignment 小）且**所有向量在球面上散開**（uniformity 小）。只有前者會「塌縮」：所有文字都映到同一點，alignment 完美卻毫無用處。Lab 13 會量這兩個數字。

### 2.5 沒有標註時的配對

- **SimCSE**（Gao 等人 2021）：同一句話過兩次 dropout，當成正例。
- **Contriever**（Izacard 等人 2021）：從同一份文件**獨立裁出兩個片段**（independent cropping）當正例。Lab 13 用的就是這個方法，正例來自同一篇 TinyStories。
- **弱監督**：網路上的天然配對（標題—內文、問題—回答、引用），E5 用這類資料做大規模預訓練，再用標註資料微調。
- **LLM 生成**：E5-Mistral（Wang 等人 2023）用 GPT-4 生成各種任務的合成訓練資料。

### 2.6 Matryoshka 表示

希望同一個向量**截斷成前 $k$ 維**也能用（省儲存、加速搜尋）。Matryoshka representation learning（Kusupati 等人 2022）把多個前綴長度的 loss 加起來：

$$\mathcal{L}_{\text{MRL}} = \frac{1}{|\mathcal{K}|}\sum_{k \in \mathcal{K}} \mathcal{L}_{\text{InfoNCE}}\left(\frac{f(q)_{1:k}}{\|f(q)_{1:k}\|}, \frac{f(d)_{1:k}}{\|f(d)_{1:k}\|}\right) \tag{13.3}$$

模型因此把最重要的資訊放在前面的維度。jina-embeddings-v5-omni 的 model card 就列出可截斷的維度（32 到 1024）。

### 2.7 評估與檢索

- **檢索指標**：recall@k（正解在前 $k$ 名的比例）、MRR（正解排名倒數的平均）、nDCG@k。
- **MTEB**（Muennighoff 等人 2022）與多語言的 MMTEB（2025）：檢索、分類、分群、重排、STS 等數十個任務的綜合排行榜；BRIGHT（2024）專門測「需要推理才找得到」的檢索。
- **BM25**：

$$\text{score}(q, d) = \sum_{w \in q} \mathrm{idf}(w)\,\frac{\mathrm{tf}(w, d)\,(k_1 + 1)}{\mathrm{tf}(w, d) + k_1\left(1 - b + b\,\frac{|d|}{\overline{|d|}}\right)} \tag{13.4}$$

詞頻會飽和（$k_1$）、長文件會被懲罰（$b$）。
- **近似最近鄰搜尋**：上百萬個向量時不能逐一比較，要用 IVF、HNSW 等索引（FAISS 函式庫）。本 repo 的 [`notes/faiss.md`](../../notes/faiss.md)、[`notes/hnsw.md`](../../notes/hnsw.md) 有整理。

### 2.8 2026 年的 embedding 模型長什麼樣

查證方式：Hugging Face 上各組織依建立日期排序的模型清單與 model card（一手，查證日期 2026-09-25）。

| 趨勢 | 例子 |
|---|---|
| **LLM 當骨幹、多種尺寸** | Qwen3-Embedding（0.6B／4B／8B，2025-06，Apache-2.0）；NVIDIA Nemotron-3-Embed（1B／8B，2026-07） |
| **多模態（文字、圖片、影片、音訊共用一個向量空間）** | Qwen3-VL-Embedding（2026-01，Apache-2.0）；jina-embeddings-v5-omni（2026-05，CC-BY-NC-4.0，Matryoshka 32–1024 維） |
| **為「需要推理的檢索」訓練** | BGE-Reasoner-Embed（2025-09，以 BRIGHT 評估）；llama-nv-embed-reasoning-3b（2026-02，CC-BY-NC-4.0） |
| **小、快、可量化** | EmbeddingGemma（300M，2025）；Microsoft 的 bitnet-embedding（270M／0.6B，1.58 位元權重，2026-07，MIT） |
| **API 模型** | Gemini Embedding（2025-03 論文） |

注意授權差很多：同樣是「開放權重」，有 Apache-2.0、MIT，也有禁止商用的 CC-BY-NC。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/embeddings.py`） |
|---|---|
| §2.2 pooling | `mean_pool`、`last_token_pool` |
| bi-encoder（GPT 骨幹＋pooling＋正規化） | `TextEncoder`、`tokenize_batch`；雙向版本用 `GPTConfig(causal=False)`（`model.Attention` 的 `padding_mask`） |
| (13.1) | `info_nce_loss`（對稱版） |
| (13.2) | `alignment_and_uniformity` |
| (13.3) | `matryoshka_loss` |
| §2.5 independent cropping、切兩半 | `random_crops`、`split_halves` |
| 訓練 | `train_contrastive` |
| §2.7 指標與基線 | `retrieval_metrics`、`bm25_scores`、`average_word_vectors`（第 02 課的詞向量取平均） |

## 4. 常見誤解

- **「語言模型的 hidden state 直接拿來就是好的 embedding」**：沒有對比微調時，平均後的向量常常擠在一個很窄的錐體裡（各向異性），相似度幾乎都很高（Lab 13 §2）。
- **「餘弦相似度 0.8 代表很相似」**：絕對數值因模型而異，只有排序有意義。
- **「dense retrieval 一定比 BM25 好」**：在關鍵字導向、領域特殊或詞彙罕見的查詢上，BM25 常常更好；hybrid 通常最穩。
- **「embedding 維度越高越好」**：Matryoshka 模型在前 1/4 的維度就保留大部分效果；儲存與搜尋成本和維度成正比。

## 5. 練習

**想一想**

1. $B = 64$ 時，隨機初始化的 encoder 的 InfoNCE loss 約是多少？
2. 從 (13.1) 的梯度說明：為什麼困難負例（相似度高的錯誤文件）得到的梯度比較大？溫度怎麼影響這件事？
3. 為什麼 causal 的語言模型用 mean pooling 時，開頭的 token 會被「低估」？last-token pooling 有什麼缺點？
4. BM25 的 (13.4) 在 $b = 0$ 與 $b = 1$ 時各是什麼意思？

**動手改**（在 `13_embeddings.ipynb`）

5. 把溫度改成 0.02 與 0.2，檢索結果與 alignment／uniformity 怎麼變？
6. 把 encoder 改成雙向（`causal=False`）再做對比訓練，比 causal＋mean pooling 好嗎？
7. 做一個 hybrid 檢索：分數 ＝ BM25（正規化後）＋ embedding 相似度，比單獨用兩者好嗎？

## 6. 延伸閱讀

- Reimers & Gurevych（2019），Sentence-BERT：<https://arxiv.org/abs/1908.10084>
- van den Oord 等人（2018），CPC／InfoNCE：<https://arxiv.org/abs/1807.03748>；Chen 等人（2020），SimCLR：<https://arxiv.org/abs/2002.05709>
- Wang & Isola（2020），alignment 與 uniformity：<https://arxiv.org/abs/2005.10242>
- Gao 等人（2021），SimCSE：<https://arxiv.org/abs/2104.08821>；Izacard 等人（2021），Contriever：<https://arxiv.org/abs/2112.09118>
- Karpukhin 等人（2020），DPR：<https://arxiv.org/abs/2004.04906>；Xiong 等人（2020），ANCE：<https://arxiv.org/abs/2007.00808>；Khattab & Zaharia（2020），ColBERT：<https://arxiv.org/abs/2004.12832>
- Wang 等人（2022），E5：<https://arxiv.org/abs/2212.03533>；Wang 等人（2023），E5-Mistral：<https://arxiv.org/abs/2401.00368>
- BehnamGhader 等人（2024），LLM2Vec：<https://arxiv.org/abs/2404.05961>；Lee 等人（2024），NV-Embed：<https://arxiv.org/abs/2405.17428>
- Kusupati 等人（2022），Matryoshka：<https://arxiv.org/abs/2205.13147>
- Muennighoff 等人（2022），MTEB：<https://arxiv.org/abs/2210.07316>；Enevoldsen 等人（2025），MMTEB：<https://arxiv.org/abs/2502.13595>；排行榜：<https://huggingface.co/spaces/mteb/leaderboard>；BRIGHT：<https://arxiv.org/abs/2407.12883>
- Zhang 等人（2025），Qwen3 Embedding：<https://arxiv.org/abs/2506.05176>；Lee 等人（2025），Gemini Embedding：<https://arxiv.org/abs/2503.07891>；Vera 等人（2025），EmbeddingGemma：<https://arxiv.org/abs/2509.20354>；Li 等人（2026），BitNet Text Embeddings：<https://arxiv.org/abs/2606.25674>
- Malkov & Yashunin（2016），HNSW：<https://arxiv.org/abs/1603.09320>；Johnson 等人（2017），FAISS：<https://arxiv.org/abs/1702.08734>
