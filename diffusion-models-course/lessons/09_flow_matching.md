# 第 09 課：Flow Matching 與 Rectified Flow

> 前置：第 05、07 課 ｜ 實驗：[`notebooks/09_flow_matching.ipynb`](../notebooks/09_flow_matching.ipynb) ｜ 預估時間：2.5 小時

## 這一課要回答的問題

- 既然生成就是解一條 ODE（第 05、07 課），能不能**直接學那條 ODE 的速度場**？
- 最簡單的「從雜訊到資料」的路徑是什麼？為什麼訓練時的直線，生成時卻是曲線？
- Rectified flow 怎麼把路徑「拉直」，讓一兩步就能生成？
- flow matching 和擴散模型到底是不是同一件事？（是，差在哪裡？）

> **時間方向提醒**：本課依照 Lipman 等人與 Meta〈Flow Matching Guide and Code〉的慣例，**$t = 0$ 是雜訊、$t = 1$ 是資料**，和 DDPM 相反。程式碼直接把變數叫做 `noise` 與 `data`。

---

## 1. 白話版

### 1.1 換一個問題問

擴散模型問的是「這張圖的雜訊是什麼？」。flow matching 問的是「**站在這裡，該往哪個方向、用多快的速度走？**」
生成時，從一個雜訊點出發，照著網路給的速度一小步一小步走，走到 $t = 1$ 就是一張圖。

### 1.2 最簡單的路：直線

訓練時怎麼知道「正確的速度」？我們自己決定路線：隨便抽一個雜訊點 $n$、一張真圖 $d$，**用直線把它們連起來、等速走**。
在時間 $t$，你站在 $x_t = t\,d + (1-t)\,n$，速度永遠是 $d - n$。網路的工作就是：看到 $x_t$ 和 $t$，猜出 $d - n$。

和 DDPM 一樣，這還是一個普通的 MSE 回歸問題。

### 1.3 可是生成時的路是彎的

問題在於：很多條直線會交叉。站在交叉點上，網路不知道你是哪一條線上的旅人，只能回答「經過這裡的所有旅人，平均往哪走」。
平均過後的速度場，走出來的路徑是**彎的**（而且彼此不會交叉）。彎的路要走很多小步才準。

### 1.4 Rectified flow：把路拉直

解法很妙：用訓練好的模型，把每個雜訊點**實際送到**它會生成的那張圖，得到新的一組「雜訊–圖」配對。
這些配對來自不交叉的彎曲路徑，所以用直線把它們連起來時，**交叉少很多**。拿這組配對重新訓練，新的速度場就更直——直到一步就能從雜訊走到圖。

---

## 2. 正式版

### 2.1 連續正規化流（CNF）

一個時間相依的速度場 $v_t(x)$ 定義了 ODE

$$\frac{dx}{dt} = v_t(x), \qquad x(0) \sim p_0 = \mathcal{N}(0, I) \tag{9.1}$$

它把 $p_0$ 推送成一族分布 $p_t$，兩者滿足**連續方程**：

$$\partial_t p_t + \nabla\cdot(p_t\,v_t) = 0 \tag{9.2}$$

（第 05 課 §2.7 推 probability flow ODE 時用過同一個方程。）Chen 等人（2018）的 Neural ODE 用最大概似訓練 CNF，每步都要解 ODE，很貴。Flow matching 的目標是**不解 ODE 就能訓練**。

### 2.2 Flow matching 的目標

選一條從 $p_0$ 到 $p_1 \approx p_{\text{data}}$ 的機率路徑 $p_t$，以及生成它的速度場 $u_t$，然後回歸：

$$L_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0,1],\ x \sim p_t}\left\|v_\theta(x, t) - u_t(x)\right\|^2$$

問題：$p_t$ 與 $u_t$ 都是對整個資料集的積分，算不出來。

### 2.3 Conditional flow matching

Lipman 等人（2022）的做法（和第 05 課的 denoising score matching 是同一招）：**先對單一資料點 $x_1$ 定義路徑**，再平均。
取直線（最佳傳輸）路徑：

$$x_t = t\,x_1 + (1-t)\,x_0, \qquad x_0 \sim \mathcal{N}(0, I) \tag{9.3}$$

對固定的 $x_1$，這條路徑的速度是常數 $u_t(x_t \mid x_1) = x_1 - x_0$。條件版損失：

$$L_{\text{CFM}}(\theta) = \mathbb{E}_{t,\ x_1 \sim p_{\text{data}},\ x_0 \sim \mathcal{N}(0, I)}\left\|v_\theta(x_t, t) - (x_1 - x_0)\right\|^2 \tag{9.4}$$

**定理**（Lipman 等人 2022）：$\nabla_\theta L_{\text{CFM}} = \nabla_\theta L_{\text{FM}}$。直覺：MSE 回歸的最佳解是條件期望，

$$v^*(x, t) = \mathbb{E}\left[x_1 - x_0 \mid x_t = x\right] \tag{9.5}$$

而這正好就是「所有經過 $x$ 的條件路徑速度的平均」——也就是能生成邊際路徑 $p_t$ 的那個速度場。

程式碼：`flow_matching_loss`，三行。取樣就是用 Euler 或 Heun 法解 (9.1)：`flow_sample`。

### 2.4 為什麼生成的路徑是彎的

(9.5) 是一個平均。在 $x_t$ 處可能有很多組 $(x_0, x_1)$ 經過，它們的方向不同，平均起來的速度場會隨位置和時間變化，所以 ODE 的解是曲線。
另一個看法：ODE 的解不能交叉（唯一性），但隨機配對的直線會大量交叉，所以真正的流必須彎曲來「避開」彼此。
路徑越彎，Euler 法每步的誤差越大，需要的步數越多。

### 2.5 Rectified flow 與 reflow

Liu、Gong、Liu（2022）提出同樣的目標（他們稱為 rectified flow），並加上 **reflow**：

1. 用獨立配對 $(x_0, x_1)$ 訓練第一個模型 $v^{(1)}$（1-rectified flow）。
2. 對新抽的雜訊 $z_0$，解 ODE 得到 $z_1 = \mathrm{ODE}_{v^{(1)}}(z_0)$，得到**有關聯的**配對 $(z_0, z_1)$。
3. 用這組配對、同樣的直線損失 (9.4) 訓練 $v^{(2)}$。

他們證明 reflow 不會增加傳輸成本，而且配對的直線交叉變少，新的流更直。若流完全是直的，**一步 Euler 就是精確的**。實務上 reflow 之後再做蒸餾，就能得到一到兩步的生成器（InstaFlow 等後續工作）。

### 2.6 和擴散模型的關係：同一家人

考慮一般的高斯路徑 $x_t = \alpha_t\,x_1 + \sigma_t\,x_0$（$x_0 \sim \mathcal{N}(0, I)$）：

| 路徑 | $\alpha_t$ | $\sigma_t$ |
|---|---|---|
| flow matching／rectified flow | $t$ | $1 - t$ |
| DDPM（VP，時間反過來看） | $\sqrt{\bar\alpha}$ | $\sqrt{1 - \bar\alpha}$ |

給定 $x_t$，四種預測彼此都是線性換算。以直線路徑為例（$v = x_1 - x_0$）：

$$\hat x_1 = x_t + (1-t)\,\hat v, \qquad \hat\epsilon = x_t - t\,\hat v, \qquad \nabla_x\log p_t(x_t) = -\frac{\hat\epsilon}{1-t} \tag{9.6}$$

所以**一個 flow matching 模型就是一個擴散模型**，只是用了不同的雜訊排程（直線路徑）和不同的損失權重（velocity 的 MSE 等於某種加權的 $\epsilon$ 或 $x_1$ MSE）。Kingma & Gao（2023）證明這些目標都可以寫成加權的 ELBO；Gao 等人（2024）的部落格〈Diffusion Meets Flow Matching〉說明了 DDIM 取樣器和 flow matching 的 Euler 取樣器在重新參數化後是同一個東西。

那為什麼業界轉向 flow matching？

- **簡單**：沒有 $\beta_t$、$\bar\alpha_t$，直線路徑、velocity 目標，實作最少。
- **路徑較直**：在相同步數下，離散化誤差通常較小。
- **大規模實驗的結果**：Esser 等人（2024，Stable Diffusion 3）比較了多種擴散與 flow 的損失與時間取樣方式，rectified flow 搭配 **logit-normal 時間取樣**（多訓練中間的 $t$）表現最好，並對高解析度做 timestep shift（第 03 課 §2.5 的同一個道理）。

### 2.7 時間方向的混亂，一次講清楚

| 文獻 | $t = 0$ | $t = 1$ | 速度目標 |
|---|---|---|---|
| Lipman 等人 2022、Meta Guide、本課程 | 雜訊 | 資料 | $x_1 - x_0$（資料 − 雜訊） |
| Liu 等人 2022（rectified flow） | 雜訊（$\pi_0$） | 資料（$\pi_1$） | 同上 |
| Esser 等人 2024（SD3） | 資料 | 雜訊 | 雜訊 − 資料 |
| DDPM | 資料（$x_0$） | 雜訊（$x_T$） | （預測 $\epsilon$） |

讀論文或程式碼時，第一件事就是確認它用哪一套。

---

## 3. 對照程式碼

| 概念 | 位置（`src/diffusion_course/flow_matching.py`） |
|---|---|
| (9.3) 直線插值 | `interpolate` |
| (9.4) CFM 損失；傳入 `noise` 固定配對（reflow 用） | `flow_matching_loss` |
| Euler／Heun 解 (9.1) | `flow_sample` |
| 測試：單點資料的精確速度場，損失為 0、Euler 精確到達 | `tests/test_flow_matching.py` |

模型介面完全不變（`model(x, t, y)`），所以 U-Net、DiT、`ClassifierFreeGuidance` 都能直接用在 flow matching 上（第 10 課）。

## 4. 常見誤解

- **「flow matching 的生成路徑是直線」**：**訓練用的條件路徑**是直線；**生成時的邊際路徑**一般是彎的。只有 reflow（或特殊配對，如 minibatch OT）才讓它變直。
- **「flow matching 和擴散是競爭的兩種方法」**：對高斯路徑而言，兩者是同一個模型家族的不同參數化與權重。
- **「一步生成只要用 flow matching 就好」**：未經 reflow／蒸餾的 flow matching 模型，一步生成的品質很差（實驗會看到）。
- **「velocity 預測在 $t$ 接近 1 時很穩」**：由 (9.6)，從 $\hat v$ 換算 score 要除以 $1-t$，在 $t \to 1$ 時會爆掉，和 $\epsilon$-prediction 在另一端的問題對稱。

## 5. 練習

**想一想**

1. 證明對固定的 $x_0$、$x_1$，(9.3) 的速度是 $x_1 - x_0$，並可以寫成 $(x_1 - x_t)/(1-t)$。
2. 推導 (9.6) 的三個換算式。
3. 說明為什麼 ODE 的解不能交叉，而隨機配對的直線會交叉。這對 (9.5) 有什麼含意？
4. 若資料分布本身就是 $\mathcal{N}(\mu, I)$，最佳速度場 $v^*(x, t)$ 是什麼？生成路徑直嗎？

**動手改**（在 `09_flow_matching.ipynb`）

5. 把時間取樣從均勻改成 logit-normal（$t = \mathrm{sigmoid}(n)$，$n \sim \mathcal{N}(0, 1)$），少步數時的品質有變化嗎？
6. 再做一次 reflow（3-rectified flow），1 步生成還能更好嗎？
7. 實作 minibatch OT 配對：在每個 batch 內，用匈牙利演算法（`scipy.optimize.linear_sum_assignment`）把雜訊與資料配對，再訓練。路徑變直了嗎？（需要自行 `uv add scipy`。）

## 6. 延伸閱讀

- Lipman 等人（2022），〈Flow Matching for Generative Modeling〉：<https://arxiv.org/abs/2210.02747>
- Liu、Gong、Liu（2022），〈Flow Straight and Fast〉（rectified flow）：<https://arxiv.org/abs/2209.03003>
- Albergo & Vanden-Eijnden（2022），〈Building Normalizing Flows with Stochastic Interpolants〉：<https://arxiv.org/abs/2209.15571>
- Lipman 等人（2024），〈Flow Matching Guide and Code〉：<https://arxiv.org/abs/2412.06264>——最完整的教材；官方程式庫 <https://github.com/facebookresearch/flow_matching>（**CC BY-NC 授權，非商用**）。
- Gao 等人（2024），〈Diffusion Meets Flow Matching: Two Sides of the Same Coin〉：<https://diffusionflow.github.io/>
- Kingma & Gao（2023），〈Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation〉：<https://arxiv.org/abs/2303.00848>
- Esser 等人（2024），Stable Diffusion 3：<https://arxiv.org/abs/2403.03206>
- Ma 等人（2024），SiT（同一個 Transformer 比較擴散與 flow 的各種設計）：<https://arxiv.org/abs/2401.08740>
- MIT 6.S184 課程講義（2025）：<https://arxiv.org/abs/2506.02070>
