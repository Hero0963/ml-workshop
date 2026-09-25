# 附錄 A：符號表與公式小抄

> 整門課的公式集中在這裡，方便複習與查找。式號對應各課講義。

## 1. 符號

| 符號 | 意義 | 首次出現 |
|---|---|---|
| $x_0$ | 乾淨資料（DDPM 慣例） | 03 |
| $x_t$ | 第 $t$ 步的帶雜訊樣本 | 03 |
| $\epsilon \sim \mathcal{N}(0, I)$ | 標準常態雜訊 | 02 |
| $T$ | 總步數（常用 1000） | 03 |
| $\beta_t$ | 第 $t$ 步加入的雜訊**變異數** | 03 |
| $\alpha_t = 1 - \beta_t$ | 第 $t$ 步保留的訊號變異數比例 | 03 |
| $\bar\alpha_t = \prod_{s \le t}\alpha_s$ | 到第 $t$ 步累積保留的訊號變異數比例 | 03 |
| $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ | 訊雜比 | 03 |
| $\tilde\mu_t, \tilde\beta_t$ | 後驗 $q(x_{t-1} \mid x_t, x_0)$ 的均值與變異數 | 04 |
| $\sigma_t^2$ | 反向一步的取樣變異數（$\beta_t$ 或 $\tilde\beta_t$） | 04 |
| $\epsilon_\theta(x_t, t)$ | 雜訊預測網路 | 04 |
| $s(x) = \nabla_x \log p(x)$ | score | 02 |
| $s_\theta(x_t, t)$ | 學到的 score | 05 |
| $\eta$ | DDIM 的隨機性參數（0 = 確定） | 07 |
| $s$（guidance 語境） | classifier-free guidance scale | 08 |
| $y$, $\varnothing$ | 條件（類別或文字）、空條件 | 08 |
| $v_\theta(x, t)$ | 速度場（flow matching） | 09 |
| $x_0, x_1$（flow 語境） | **雜訊、資料**（和 DDPM 相反！） | 09 |

**程式碼慣例**：步數索引 $i = 0, \dots, T-1$ 對應論文的 $t = i + 1$；所有模型收到的時間是 $[0, 1)$ 的浮點數；flow matching 的變數直接叫 `noise`、`data`。

## 2. 常態分布的工具（第 02 課）

| | 公式 |
|---|---|
| 重參數化 | $x = \mu + \sigma\epsilon$ |
| 線性組合 | $aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2,\ a^2\sigma_1^2 + b^2\sigma_2^2)$ |
| 乘積 | $\mathcal{N}(x; a, A)\,\mathcal{N}(x; b, B) \propto \mathcal{N}\left(x; \frac{Ba + Ab}{A+B}, \frac{AB}{A+B}\right)$ |
| 同變異數的 KL | $\mathrm{KL} = \frac{\lVert \mu_1 - \mu_2\rVert ^2}{2\sigma^2}$ |
| 常態的 score | $\nabla_x \log \mathcal{N}(x; \mu, \sigma^2 I) = -(x - \mu)/\sigma^2$ |
| Langevin | $x_{k+1} = x_k + \eta\,\nabla\log p(x_k) + \sqrt{2\eta}\,z_k$ |

## 3. DDPM（第 03、04 課）

| | 公式 |
|---|---|
| 一步前向 (3.1) | $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I)$ |
| 封閉解 (3.2) | $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ |
| 後驗均值 (4.4) | $\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$ |
| 後驗變異數 (4.4) | $\tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$ |
| 訓練損失 (4.9) | $L_{\text{simple}} = \mathbb{E}\left\lVert \epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\right\rVert^2$ |
| 取樣一步 (4.10) | $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta\right) + \sigma_t z$ |
| linear schedule | $\beta_t$：$10^{-4} \to 0.02$，$T = 1000$ |
| cosine schedule (3.4) | $\bar\alpha_t = f(t)/f(0)$，$f(t) = \cos^2\left(\frac{t/T + 0.008}{1.008}\cdot\frac{\pi}{2}\right)$ |

## 4. 各種預測之間的換算

給定 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$（DDPM）：

| 從 | 到 | 公式 |
|---|---|---|
| $\hat\epsilon$ | $\hat x_0$ | $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ |
| $\hat x_0$ | $\hat\epsilon$ | $\hat\epsilon = (x_t - \sqrt{\bar\alpha_t}\,\hat x_0)/\sqrt{1-\bar\alpha_t}$ |
| $\hat\epsilon$ | score | $s = -\hat\epsilon/\sqrt{1-\bar\alpha_t}$ (5.2) |
| score | $\hat x_0$（Tweedie） | $\hat x_0 = (x_t + (1-\bar\alpha_t)\,s)/\sqrt{\bar\alpha_t}$ (5.3) |
| $\hat v$ | $\hat x_0$ | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$；$\hat x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,\hat v$ |

給定 $x_t = t\,x_1 + (1-t)\,x_0$（flow matching，$x_0$ 是雜訊）：

| 從 $\hat v$ 到 | 公式 (9.6) |
|---|---|
| 乾淨資料 | $\hat x_1 = x_t + (1-t)\,\hat v$ |
| 雜訊 | $\hat x_0 = x_t - t\,\hat v$ |
| score | $s = -\hat x_0/(1-t)$ |

## 5. Score 與 SDE（第 05 課）

| | 公式 |
|---|---|
| Denoising score matching 目標 | $\nabla_{\tilde x}\log q(\tilde x \mid x) = -\epsilon/\sigma$ |
| 前向 SDE (5.4) | $dx = f(x, t)\,dt + g(t)\,dw$ |
| VP-SDE | $dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dw$ |
| 反向 SDE (5.5) | $dx = [f - g^2\nabla_x\log p_t(x)]\,dt + g\,d\bar w$ |
| Probability flow ODE (5.6) | $dx/dt = f - \frac{1}{2}g^2\nabla_x\log p_t(x)$ |

## 6. DDIM（第 07 課）

| | 公式 |
|---|---|
| 一步 (7.2) | $x_{t-1} = \sqrt{\bar\alpha_{t-1}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\hat\epsilon + \sigma_t z$ |
| 隨機性 (7.3) | $\sigma_t = \eta\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}$ |
| ODE 形式 (7.4) | $d\bar x/d\bar\sigma = \epsilon_\theta$，$\bar x = x/\sqrt{\bar\alpha}$，$\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$ |
| slerp | $\frac{\sin((1-\lambda)\Omega)}{\sin\Omega}z_1 + \frac{\sin(\lambda\Omega)}{\sin\Omega}z_2$ |

## 7. Guidance（第 08 課）

| | 公式 |
|---|---|
| 貝氏分解 (8.1) | $\nabla\log p(x_t \mid y) = \nabla\log p(x_t) + \nabla\log p(y \mid x_t)$ |
| Classifier guidance (8.2) | $\hat\epsilon = \epsilon_\theta - \gamma\sqrt{1-\bar\alpha_t}\,\nabla\log p_\phi(y \mid x_t)$ |
| Classifier-free guidance (8.3) | $\tilde\epsilon = \epsilon_\theta(x_t, \varnothing) + s\,(\epsilon_\theta(x_t, y) - \epsilon_\theta(x_t, \varnothing))$ |
| 與 Ho & Salimans 的換算 | $s = 1 + w$ |

## 8. Flow matching（第 09 課）

| | 公式 |
|---|---|
| ODE (9.1) | $dx/dt = v_t(x)$，$x(0) \sim \mathcal{N}(0, I)$ |
| 連續方程 (9.2) | $\partial_t p_t + \nabla\cdot(p_t v_t) = 0$ |
| 直線路徑 (9.3) | $x_t = t\,x_1 + (1-t)\,x_0$ |
| CFM 損失 (9.4) | $\mathbb{E}\lVert v_\theta(x_t, t) - (x_1 - x_0)\rVert ^2$ |
| 最佳速度 (9.5) | $v^*(x, t) = \mathbb{E}[x_1 - x_0 \mid x_t = x]$ |
