# 第 08 課：條件生成與 Classifier-Free Guidance

> 前置：第 05、06 課 ｜ 實驗：[`notebooks/08_classifier_free_guidance.ipynb`](../notebooks/08_classifier_free_guidance.ipynb) ｜ 預估時間：2 小時（實驗訓練在 CPU 上約 15–20 分鐘）

## 這一課要回答的問題

- 怎麼讓模型「畫一個 7」而不是隨便畫？
- 只把標籤餵給網路為什麼常常不夠？guidance 在「放大」什麼？
- 為什麼幾乎所有文生圖模型都在用 classifier-free guidance？它的代價是什麼？

---

## 1. 白話版

### 1.1 把「要畫什麼」告訴網路

最直接的做法：除了 $x_t$ 和 $t$，再把標籤 $y$（例如「7」）也當成輸入。網路學的就變成「在知道要畫 7 的前提下，猜雜訊」。這叫**條件擴散模型**。

### 1.2 為什麼還要 guidance

只餵標籤，模型畫出的 7 常常「不夠像 7」：有些歪七扭八、有些像 1。因為模型學的是**所有** 7 的分布，包含那些寫得很潦草的。

guidance 的想法：同一張圖，問網路兩次——

- 「如果要畫 7，雜訊是什麼？」（有條件）
- 「如果什麼都不指定，雜訊是什麼？」（無條件）

兩個答案的**差**，就是「7 之所以是 7」的方向。把這個差**放大**，往那個方向多走一點，畫出來的就更「典型」地像 7。

### 1.3 代價

放大得越多，每個 7 都越像「教科書上的 7」，但所有 7 也越來越像彼此——**多樣性下降**。放大過頭，影像還會出現過飽和、過度銳利的怪樣子。guidance scale 是品質與多樣性之間的一個旋鈕。

---

## 2. 正式版

### 2.1 條件模型

訓練 $\epsilon_\theta(x_t, t, y)$，損失和 (4.9) 相同，只是多了輸入 $y$。它估計的是條件分布的 score：$s_\theta(x_t, t, y) \approx \nabla_{x_t}\log p_t(x_t \mid y)$。
本課程把類別嵌入**加到時間嵌入上**（`LabelEmbedding`），再注入每個 ResBlock。文字條件則通常用 cross-attention（第 10 課）。

### 2.2 Classifier guidance

由貝氏定理，$p(x_t \mid y) \propto p(x_t)\,p(y \mid x_t)$，所以

$$\nabla_{x_t}\log p(x_t \mid y) = \nabla_{x_t}\log p(x_t) + \nabla_{x_t}\log p(y \mid x_t) \tag{8.1}$$

Dhariwal & Nichol（2021）另外訓練一個**認得帶雜訊影像**的分類器 $p_\phi(y \mid x_t, t)$，並把第二項乘上 $\gamma$：

$$\hat\epsilon = \epsilon_\theta(x_t, t) - \gamma\,\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(y \mid x_t) \tag{8.2}$$

（係數 $-\sqrt{1-\bar\alpha_t}$ 來自 score 與雜訊的換算，式 5.2。）$\gamma > 1$ 相當於從「銳化」過的分布取樣：

$$\tilde p(x \mid y) \propto p(x)\,p(y \mid x)^{\gamma} = p(x \mid y)\,p(y \mid x)^{\gamma - 1}$$

$p(y \mid x)$ 高的樣本（分類器很有把握的）被放大。缺點：要另外訓練一個在各種雜訊等級都能用的分類器；而且對分類器取梯度，容易得到「騙過分類器」的對抗式雜訊，而不是真正更好的圖。

### 2.3 Classifier-free guidance（CFG）

Ho & Salimans（2022）的觀察：(8.1) 裡的分類器梯度可以用兩個擴散模型的差來表示，

$$\nabla_{x_t}\log p(y \mid x_t) = \nabla_{x_t}\log p(x_t \mid y) - \nabla_{x_t}\log p(x_t)$$

所以不需要分類器，只要**同時有條件模型與無條件模型**。換成雜訊預測：

$$\tilde\epsilon = \epsilon_\theta(x_t, t, \varnothing) + s\left(\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing)\right) \tag{8.3}$$

- $s = 0$：無條件模型
- $s = 1$：普通的條件模型
- $s > 1$：往「條件方向」外插，也就是 guidance

**一個網路同時當兩個模型**：訓練時以機率 $p_{\text{uncond}}$（常用 0.1–0.2）把標籤換成特殊的「空標籤」$\varnothing$（程式碼：`drop_labels`），網路就同時學會有條件與無條件兩種預測。取樣時每步把 batch 複製兩份，一份給真標籤、一份給空標籤，一次前向算完（`ClassifierFreeGuidance`）。代價是**每步的計算量加倍**。

**符號注意**：Ho & Salimans 原文寫成 $\tilde\epsilon = (1+w)\,\epsilon_\theta(x_t, y) - w\,\epsilon_\theta(x_t, \varnothing)$，他們的 $w$ 等於這裡的 $s - 1$。Hugging Face `diffusers` 的 `guidance_scale` 和本課的 $s$ 相同（Stable Diffusion v1 系列的預設值是 7.5）。

(8.3) 對 velocity 預測（flow matching，第 09 課）一樣成立，因為 $\epsilon$、score、velocity 之間都是對 $x_t$ 的線性換算（在同一個 $x_t$、同一個 $t$ 下換算係數相同）。

### 2.4 guidance 不是「正確」的取樣

值得知道：(8.3) 對應的「銳化分布」$p(x_t \mid y)\,p(y \mid x_t)^{s-1}$ 在不同雜訊等級之間**並不一致**（它不是某一個固定分布加噪後的結果），所以 CFG 並不是精確地從任何分布取樣。它是一個在實務上極為有效的啟發式方法。這也催生了許多改良：

- **只在中間的雜訊區段開啟 guidance**（Kynkäänniemi 等人 2024）：高雜訊端的 guidance 會壓縮多樣性，低雜訊端幾乎沒用。
- **Autoguidance**（Karras 等人 2024）：不跟無條件模型比，而是跟「同一個模型的較差版本」比，減少多樣性損失。

### 2.5 從類別到文字

文生圖模型把 $y$ 換成文字編碼器（CLIP、T5 或 LLM）輸出的向量序列，$\varnothing$ 換成空字串的編碼。
**Negative prompt** 就是把 (8.3) 的 $\varnothing$ 換成「不想要的東西」的編碼，讓生成往遠離它的方向走。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| 訓練時隨機丟標籤 | `drop_labels`（`src/diffusion_course/guidance.py`） |
| (8.3) | `ClassifierFreeGuidance`——包成一個「模型」，任何取樣器都能直接用 |
| 類別嵌入與空標籤 | `LabelEmbedding`（`src/diffusion_course/models/embeddings.py`）——`y=None` 等同空標籤 |
| 測試：$s = 1$ 等於條件模型、$s = 0$ 等於無條件模型 | `tests/test_guidance.py` |

因為 `ClassifierFreeGuidance` 和其他模型一樣是 `model(x, t, y)` 介面，`ddpm_sample`、`ddim_sample`、`flow_sample` 都不用改一行就支援 guidance。

## 4. 常見誤解

- **「guidance scale 越大越好」**：準確度會先升後持平，多樣性一路下降，太大還會過飽和。實驗會畫出這條取捨曲線。
- **「CFG 需要訓練兩個模型」**：一個網路、隨機丟標籤就夠了。
- **「$s = 1$ 就是沒有條件」**：$s = 1$ 是**有條件但不加強**；$s = 0$ 才是無條件。
- **「guidance 只能用在文字」**：任何條件（類別、影像、深度圖、姿勢……）都能用。

## 5. 練習

**想一想**

1. 從 (8.1) 出發，推導 (8.2) 中的係數 $-\sqrt{1-\bar\alpha_t}$。
2. 證明 $p(x)\,p(y \mid x)^{\gamma} = p(x \mid y)\,p(y \mid x)^{\gamma-1}\,p(y)$，並解釋 $\gamma > 1$ 時哪些樣本被加權。
3. 把 Ho & Salimans 的 $(1+w)\epsilon_c - w\epsilon_u$ 改寫成 (8.3) 的形式，確認 $s = 1 + w$。
4. $p_{\text{uncond}} = 0$ 或 $p_{\text{uncond}} = 1$ 時，CFG 會怎樣？

**動手改**（在 `08_classifier_free_guidance.ipynb`）

5. 實作「guidance interval」：只在 $t \in [200, 800]$ 使用 $s = 4$，其他時候用 $s = 1$。和全程 $s = 4$ 比較準確度與多樣性。
6. 把 `P_UNCOND` 改成 0.3，重訓後掃一次 scale。
7. （進階）實作 classifier guidance：訓練一個輸入帶雜訊 $x_t$ 與 $t$ 的分類器，套用 (8.2)。

## 6. 延伸閱讀

- Dhariwal & Nichol（2021），classifier guidance：<https://arxiv.org/abs/2105.05233>
- Ho & Salimans（2022），classifier-free guidance：<https://arxiv.org/abs/2207.12598>
- Sander Dieleman，〈Guidance: a cheat code for diffusion models〉（2022）：<https://sander.ai/2022/05/26/guidance.html>
- Nichol 等人（2021），GLIDE（CFG 用於文生圖）：<https://arxiv.org/abs/2112.10741>
- Kynkäänniemi 等人（2024），guidance interval：<https://arxiv.org/abs/2404.07724>
- Karras 等人（2024），autoguidance：<https://arxiv.org/abs/2406.02507>
