# 第 05 課：Score 與 SDE——同一個模型的另一種說法

> 前置：第 02、04 課 ｜ 實驗：[`notebooks/05_score_field.ipynb`](../notebooks/05_score_field.ipynb) ｜ 預估時間：2.5 小時

## 這一課要回答的問題

- 第 04 課「猜雜訊」的網路，和第 02 課的 score 是什麼關係？
- 為什麼網路猜出的 $\hat x_0$ 在雜訊多時是「平均」？（Tweedie 公式）
- 把步數取到無限多，DDPM 會變成什麼？為什麼同一個模型可以「不加雜訊」地確定性取樣？

---

## 1. 白話版

### 1.1 雜訊的反方向，就是回家的方向

$x_t$ 是乾淨資料加上雜訊。網路猜出了雜訊 $\hat\epsilon$，那麼「把雜訊扣掉」的方向 $-\hat\epsilon$，就是從 $x_t$ 走回資料的方向。
第 02 課說 score 是「往資料多的地方走的箭頭」——**這兩件事是同一個箭頭**，只差一個縮放倍率。

所以：

- DDPM 的網路其實就是在學 score；
- DDPM 的「很多個 $t$」就是第 02 課 annealed Langevin 的「很多個雜訊等級」；
- DDPM 取樣和 annealed Langevin，是同一個想法的兩種寫法。

### 1.2 把步數切到無限細

DDPM 用 1000 個離散步驟。如果切成一萬、一百萬步……極限就是一條**連續時間**的隨機微分方程（SDE）：一個粒子一邊被推著走、一邊被隨機踢。

連續時間的好處是：這條「有隨機踢」的路，有一個**沒有隨機、完全確定的雙胞胎**（probability flow ODE）。兩者在每個時刻的**分布**都一樣，但 ODE 版本的每個粒子走的是一條平滑的確定路徑：同一個起點永遠走到同一張圖。

這個確定性雙胞胎是第 07 課 DDIM、第 09 課 flow matching 的起點。

---

## 2. 正式版

### 2.1 Score matching：怎麼學一個不知道答案的梯度

想學 $s_\theta(x) \approx \nabla_x \log p(x)$，最直接的目標是

$$J(\theta) = \tfrac{1}{2}\,\mathbb{E}_{p(x)}\left\|s_\theta(x) - \nabla_x \log p(x)\right\|^2$$

但 $\nabla_x \log p$ 正是我們不知道的東西。Hyvärinen（2005）用分部積分把它改寫成不含未知項的形式：

$$J(\theta) = \mathbb{E}_{p(x)}\left[\mathrm{tr}\left(\nabla_x s_\theta(x)\right) + \tfrac{1}{2}\left\|s_\theta(x)\right\|^2\right] + C$$

可惜 $\mathrm{tr}(\nabla_x s_\theta)$ 要算 $d$ 次反向傳播，影像的 $d$ 動輒上萬，太貴。

### 2.2 Denoising score matching

Vincent（2011）的做法：不學乾淨資料的 score，改學**加了雜訊之後**的分布 $q_\sigma(\tilde x) = \int p(x)\,\mathcal{N}(\tilde x; x, \sigma^2 I)\,dx$ 的 score。目標變成

$$J_{\text{DSM}}(\theta) = \tfrac{1}{2}\,\mathbb{E}_{x \sim p,\ \tilde x \sim \mathcal{N}(x, \sigma^2 I)}\left\|s_\theta(\tilde x) - \nabla_{\tilde x}\log q_\sigma(\tilde x \mid x)\right\|^2, \qquad \nabla_{\tilde x}\log q_\sigma(\tilde x \mid x) = -\frac{\tilde x - x}{\sigma^2} = -\frac{\epsilon}{\sigma} \tag{5.1}$$

目標項完全已知。為什麼這樣學得到 $\nabla \log q_\sigma(\tilde x)$？因為

$$\nabla_{\tilde x}\log q_\sigma(\tilde x) = \mathbb{E}\left[\nabla_{\tilde x}\log q_\sigma(\tilde x \mid x)\ \middle|\ \tilde x\right]$$

（對 $q_\sigma(\tilde x) = \int q_\sigma(\tilde x \mid x)\,p(x)\,dx$ 取 log 再微分即得），而 MSE 迴歸的最佳解正是條件期望。和第 04 課 §2.8 是同一個道理。

### 2.3 DDPM 就是 denoising score matching

DDPM 的 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，所以

$$\nabla_{x_t}\log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

定義

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar\alpha_t}} \tag{5.2}$$

就有 $\left\|\epsilon_\theta - \epsilon\right\|^2 = (1-\bar\alpha_t)\left\|s_\theta - \nabla_{x_t}\log q(x_t \mid x_0)\right\|^2$：**$L_{\text{simple}}$ 是對所有雜訊等級、以 $(1-\bar\alpha_t)$ 加權的 denoising score matching。** 程式碼：`eps_to_score`（`src/diffusion_course/score.py`）。

和 NCSN（第 02 課的 annealed Langevin）的關係：把 DDPM 的 $x_t$ 除以 $\sqrt{\bar\alpha_t}$，

$$\frac{x_t}{\sqrt{\bar\alpha_t}} = x_0 + \sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}\,\epsilon$$

就是「原資料加上標準差 $\sigma_t = 1/\sqrt{\mathrm{SNR}(t)}$ 的雜訊」，和 NCSN 完全同型，只差一個縮放。

### 2.4 Tweedie 公式：$\hat x_0$ 為什麼是平均

若 $x_t = a\,x_0 + \sigma\,\epsilon$，則（Robbins 1956；Efron 2011）

$$\mathbb{E}[x_0 \mid x_t] = \frac{x_t + \sigma^2\,\nabla_{x_t}\log p_t(x_t)}{a} \tag{5.3}$$

代入 $a = \sqrt{\bar\alpha_t}$、$\sigma^2 = 1-\bar\alpha_t$ 與 (5.2)，得到

$$\mathbb{E}[x_0 \mid x_t] \approx \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t, t)}{\sqrt{\bar\alpha_t}}$$

這正是第 04 課的 `predict_x0`。所以 $\hat x_0$ 是**所有可能原圖的條件平均**（最小均方誤差估計）：雜訊多時可能性很多、平均起來很糊（第 04 課實驗 §4 的「擠在中間」），雜訊少時才銳利。這也解釋了為什麼一步去噪做不出好圖，必須多步。

### 2.5 連續時間：SDE

Song 等人（2021）把前向過程寫成隨機微分方程（Itô SDE）：

$$dx = f(x, t)\,dt + g(t)\,dw \tag{5.4}$$

$f$ 是漂移（確定性的推動），$g$ 控制隨機踢的強度，$w$ 是布朗運動。兩個主要例子：

| 名稱 | SDE | 對應的離散模型 |
|---|---|---|
| VP-SDE | $dx = -\tfrac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw$ | DDPM（$\beta_t$ 的連續極限） |
| VE-SDE | $dx = \sqrt{\dfrac{d[\sigma^2(t)]}{dt}}\,dw$ | NCSN（只加雜訊、不縮小） |

VP-SDE 的邊際分布是 $x_t \mid x_0 \sim \mathcal{N}\left(e^{-\frac{1}{2}\int_0^t \beta}\,x_0,\ (1 - e^{-\int_0^t \beta})\,I\right)$——對照第 03 課練習 3 的 $\bar\alpha_t \approx \exp(-\sum_s \beta_s)$，就是它的離散版。

### 2.6 反向時間 SDE

Anderson（1982）證明：(5.4) 在時間上倒著走，也是一條 SDE：

$$dx = \left[f(x, t) - g(t)^2\,\nabla_x \log p_t(x)\right]dt + g(t)\,d\bar w \tag{5.5}$$

其中 $dt$ 是負的（時間從 $T$ 走回 0），$\bar w$ 是反向的布朗運動。**唯一未知的就是每個時刻的 score $\nabla_x \log p_t(x)$**——用 (5.2) 的網路代入，就能從雜訊解回資料。DDPM 的 Algorithm 2 可以看成 (5.5) 的一種離散化。

### 2.7 Probability flow ODE

Song 等人（2021）的關鍵觀察：下面這條**沒有隨機項**的 ODE

$$\frac{dx}{dt} = f(x, t) - \tfrac{1}{2}\,g(t)^2\,\nabla_x \log p_t(x) \tag{5.6}$$

在每個時刻 $t$ 的分布，和 SDE (5.4) 完全相同。

**為什麼？** 看分布怎麼隨時間演化。SDE 的密度滿足 Fokker–Planck 方程：

$$\partial_t p = -\nabla\cdot(f p) + \tfrac{1}{2}g^2\,\Delta p$$

利用 $\Delta p = \nabla\cdot(\nabla p) = \nabla\cdot(p\,\nabla\log p)$，改寫成

$$\partial_t p = -\nabla\cdot\left(\left[f - \tfrac{1}{2}g^2\,\nabla\log p\right] p\right)$$

這正是「粒子以速度 $f - \tfrac{1}{2}g^2\nabla\log p$ 確定性地流動」時的連續方程（continuity equation）。同一條密度演化、兩種粒子層級的實現：一種帶隨機踢，一種不帶。

VP-SDE 的 probability flow ODE 是 $\frac{dx}{dt} = -\tfrac{1}{2}\beta(t)\left[x + \nabla_x\log p_t(x)\right]$。

### 2.8 ODE 帶來的好處

- **確定性的對應**：每個雜訊 $x_T$ 對應唯一一張圖，可以做插值、反推（把真實圖片編碼回雜訊再編輯）。
- **可以用任何 ODE 解法**：步數可以大幅減少（第 07 課）。
- **精確的 likelihood**：用 instantaneous change of variables 公式沿 ODE 積分 $\nabla\cdot$ 速度場即可。
- **通往 flow matching**：既然生成只是「解一條 ODE」，何不直接學速度場？（第 09 課）

取捨：SDE 取樣的隨機性可以「修正」前面步驟累積的誤差，ODE 不行；實務上兩者各有擅長（EDM 論文有詳細比較）。

### 2.9 三種取樣方式對照

| 方法 | 做法 | 在本課程 |
|---|---|---|
| 反向 SDE | 離散化 (5.5)，每步加雜訊 | `ddpm_sample`（Algorithm 2） |
| Probability flow ODE | 離散化 (5.6)，不加雜訊 | `ddim_sample(eta=0)`（第 07 課解釋為什麼它是 (5.6) 的離散化） |
| Predictor–corrector | 每個預測步後，再做幾步 Langevin「校正」 | 練習 7；Song 等人 2021 §4.2 |

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| (5.2) 雜訊 ↔ score | `eps_to_score`（`src/diffusion_course/score.py`） |
| 常態混合加噪後的精確 score | `GaussianMixture.diffused(alpha_bar).score`（`src/diffusion_course/data.py`） |
| (5.3) Tweedie | `predict_x0`（`src/diffusion_course/ddpm.py`） |
| 反向 SDE 取樣 | `ddpm_sample` |
| Probability flow ODE 取樣 | `ddim_sample(..., eta=0.0)`（`src/diffusion_course/ddim.py`） |

## 4. 常見誤解

- **「score-based 模型和 DDPM 是兩種模型」**：訓練目標只差一個逐時間的權重，網路輸出只差一個縮放，可以互換。
- **「ODE 取樣和 SDE 取樣出來的分布不同」**：若 score 完全準確，每個時刻的分布完全相同；實務上的差異來自 score 誤差與離散化誤差。
- **「$\hat x_0$ 糊掉代表模型沒學好」**：高雜訊時 $\hat x_0$ 本來就該是糊的平均，這是最佳解（Tweedie）。
- **「學到的 score 到處都準」**：只有在該雜訊等級的資料常出現的地方才準。實驗會畫出「小 $t$ 時，遠離資料處誤差很大」。

## 5. 練習

**想一想**

1. 證明 $\nabla_{\tilde x}\log q_\sigma(\tilde x) = \mathbb{E}[\nabla_{\tilde x}\log q_\sigma(\tilde x \mid x) \mid \tilde x]$。
2. 從 (5.1) 的最佳解出發，推導 Tweedie 公式 (5.3)（先做 $a = 1$ 的情況）。
3. 驗證 $\left\|\epsilon_\theta - \epsilon\right\|^2 = (1-\bar\alpha_t)\left\|s_\theta - \nabla\log q(x_t \mid x_0)\right\|^2$。
4. 寫出 VE-SDE 的 probability flow ODE。
5. 若 $p_0 = \mathcal{N}(0, I)$，VP-SDE 的 $p_t$ 是什麼？此時 probability flow ODE 的速度是多少？這說明了什麼？

**動手改**（在 `05_score_field.ipynb`）

6. 把訓練步數減半，score 誤差地圖怎麼變？哪裡先變差？
7. 實作 predictor–corrector：每個 DDPM 步驟之後，用學到的 score 做 1 步 Langevin（步長自己試），比較樣本品質。

## 6. 延伸閱讀

- Song 等人（2021），〈Score-Based Generative Modeling through Stochastic Differential Equations〉：<https://arxiv.org/abs/2011.13456>——本課 §2.5–2.9 的出處。
- Yang Song 的部落格（2021）：<https://yang-song.net/blog/2021/score/>——有很好的動畫與直覺。
- Hyvärinen（2005），〈Estimation of Non-Normalized Statistical Models by Score Matching〉，*JMLR* 6。
- Vincent（2011），〈A Connection Between Score Matching and Denoising Autoencoders〉，*Neural Computation* 23(7)。
- Efron（2011），〈Tweedie's Formula and Selection Bias〉，*JASA* 106。
- Karras 等人（2022），EDM：<https://arxiv.org/abs/2206.00364>——把 SDE／ODE 取樣的設計選擇拆開比較。
- MIT 6.S184〈Introduction to Flow Matching and Diffusion Models〉講義（2025）：<https://arxiv.org/abs/2506.02070>，課程網站：<https://diffusion.csail.mit.edu/>——用 SDE／ODE 語言統一整個領域。
