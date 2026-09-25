# 第 04 課：DDPM——反向過程、訓練與取樣

> 前置：第 02、03 課 ｜ 實驗：[`notebooks/04_ddpm_2d.ipynb`](../notebooks/04_ddpm_2d.ipynb) ｜ 預估時間：3 小時（建議拿紙筆跟著推）

## 這一課要回答的問題

- 生成時要「倒帶」，每倒一步的分布長什麼樣子？
- 訓練目標從哪裡來？為什麼一個看起來很複雜的機率模型，最後的損失函數只是「猜雜訊的 MSE」？
- 訓練與取樣的完整演算法是什麼？損失為什麼降不到 0？

---

## 1. 白話版

### 1.1 知道答案時，倒帶一步很簡單

第 03 課：從 $x_0$ 加雜訊到 $x_t$ 有公式。反過來，若**同時知道**乾淨的 $x_0$ 和髒的 $x_t$，「上一步 $x_{t-1}$ 長什麼樣」也有公式——它大約在 $x_0$ 和 $x_t$ 之間，靠 $x_t$ 近一點，再帶一點點不確定性。

問題是：**生成時我們不知道 $x_0$**。這正是要生成的東西。

### 1.2 讓網路猜——猜什麼最方便？

既然不知道 $x_0$，就訓練一個網路從 $x_t$ 去猜。有幾種等價的猜法：

- 猜乾淨的圖 $x_0$；
- 猜**當初加進去的雜訊** $\epsilon$（知道 $\epsilon$，由 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ 就能反推 $x_0$）。

DDPM 選了猜雜訊，實驗上效果最好。訓練過程就是：

1. 拿一張真圖 $x_0$；
2. 隨機挑一個時間 $t$、隨機抽一份雜訊 $\epsilon$，算出 $x_t$；
3. 讓網路看 $x_t$ 和 $t$，猜 $\epsilon$；
4. 用 MSE 懲罰猜錯的量。

**就這樣。** 整個 DDPM 的訓練就是這四行。

### 1.3 為什麼損失降不到 0

看著一張雜訊很多的圖，你沒辦法確定「雜訊是哪些、原圖是哪些」——很多張不同的原圖加上不同的雜訊，都可能得到同一張 $x_t$。
網路最好的策略是猜「所有可能性的平均」，而平均一定會和真正的答案有差距。所以損失會停在某個正值，這是正常的。

---

## 2. 正式版

### 2.1 反向過程

定義生成模型為一條從雜訊出發的馬可夫鏈：

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1} \mid x_t), \qquad p(x_T) = \mathcal{N}(0, I), \qquad p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(\mu_\theta(x_t, t),\ \sigma_t^2 I\right) \tag{4.1}$$

**為什麼反向一步可以假設成常態？** 當每步雜訊 $\beta_t$ 夠小時，真正的反向轉移 $q(x_{t-1} \mid x_t)$ 近似於常態（Feller 1949；Sohl-Dickstein 等人 2015 在擴散模型中使用這個事實）。這也是 $T$ 必須很大的原因：步子一大，反向分布就不再像常態，一個常態分布描述不了它（第 07、09、11 課的「少步取樣」都在跟這件事搏鬥）。

### 2.2 ELBO：把訓練目標拆成一項一項

和 VAE 一樣，最大化 $\log p_\theta(x_0)$ 太難，改為最大化它的下界（Jensen 不等式）：

$$\log p_\theta(x_0) \ge \mathbb{E}_{q(x_{1:T} \mid x_0)}\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)}\right]$$

關鍵技巧：前向過程是馬可夫的，所以 $q(x_t \mid x_{t-1}) = q(x_t \mid x_{t-1}, x_0)$，再用貝氏定理把它改寫成「反向、且條件在 $x_0$ 上」的形式：

$$q(x_t \mid x_{t-1}, x_0) = \frac{q(x_{t-1} \mid x_t, x_0)\,q(x_t \mid x_0)}{q(x_{t-1} \mid x_0)}$$

代回去，分母分子的 $q(x_t \mid x_0)$ 會一路相消，得到（Ho 等人 2020，式 5）：

$$-\text{ELBO} = \underbrace{\mathrm{KL}\left(q(x_T \mid x_0)\,\|\,p(x_T)\right)}_{L_T} + \sum_{t=2}^{T}\underbrace{\mathbb{E}_q\,\mathrm{KL}\left(q(x_{t-1} \mid x_t, x_0)\,\|\,p_\theta(x_{t-1} \mid x_t)\right)}_{L_{t-1}}\ \underbrace{-\ \mathbb{E}_q \log p_\theta(x_0 \mid x_1)}_{L_0} \tag{4.2}$$

三種項的意義：

- $L_T$：終點和先驗 $\mathcal{N}(0, I)$ 差多少。沒有可學的參數（schedule 設計好就接近 0）。
- $L_{t-1}$：**模型的反向一步**要去模仿「**知道答案時的反向一步**」。這是主角。
- $L_0$：最後一步的重建項；實作上通常併入下面的簡化損失處理。

### 2.3 知道答案時的反向一步（後驗）

$q(x_{t-1} \mid x_t, x_0) \propto q(x_t \mid x_{t-1})\,q(x_{t-1} \mid x_0)$，兩個都是常態：

- 把 $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t}\,x_{t-1}, \beta_t)$ 看成 $x_{t-1}$ 的函數，它 $\propto \mathcal{N}(x_{t-1}; x_t/\sqrt{\alpha_t},\ \beta_t/\alpha_t)$；
- $q(x_{t-1} \mid x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar\alpha_{t-1}}\,x_0,\ 1-\bar\alpha_{t-1})$。

套第 02 課的 P3（常態乘常態）並化簡（用到 $\alpha_t + \beta_t = 1$、$\alpha_t\bar\alpha_{t-1} = \bar\alpha_t$）：

$$q(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(\tilde\mu_t(x_t, x_0),\ \tilde\beta_t I\right) \tag{4.3}$$

$$\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\,x_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\,\beta_t \tag{4.4}$$

$\tilde\mu_t$ 是 $x_0$ 和 $x_t$ 的加權平均；$\tilde\beta_t \le \beta_t$，因為多知道了 $x_0$，不確定性變小。

### 2.4 KL 變成均值的 MSE

$p_\theta$ 與 $q$ 都是常態、變異數固定（$\sigma_t^2$ 不學），由第 02 課 P4：

$$L_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\left\|\tilde\mu_t(x_t, x_0) - \mu_\theta(x_t, t)\right\|^2\right] + C \tag{4.5}$$

模型只需要學「均值」。

### 2.5 改成預測雜訊

把 $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}\left(x_t - \sqrt{1-\bar\alpha_t}\,\epsilon\right)$ 代入 (4.4)，係數化簡後得到

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon\right) \tag{4.6}$$

$x_t$ 在取樣時是已知的，唯一未知的是 $\epsilon$。所以把模型也寫成同樣的形式：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) \tag{4.7}$$

代入 (4.5)：

$$L_{t-1} = \mathbb{E}_{x_0, \epsilon}\left[\frac{\beta_t^2}{2\sigma_t^2\,\alpha_t\,(1-\bar\alpha_t)}\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\right)\right\|^2\right] + C \tag{4.8}$$

### 2.6 丟掉權重：$L_{\text{simple}}$

Ho 等人發現，**把 (4.8) 前面的權重整個拿掉**，樣本品質反而更好：

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\{1, \dots, T\},\ x_0,\ \epsilon \sim \mathcal{N}(0, I)}\left[\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t\right)\right\|^2\right] \tag{4.9}$$

為什麼？取 $\sigma_t^2 = \beta_t$ 時，(4.8) 的權重正比於 $\beta_t / (\alpha_t(1-\bar\alpha_t))$：$t = 1$ 時約 1，$t = T$ 時約 0.02。真正的 ELBO 非常偏重雜訊很少的步驟；$L_{\text{simple}}$ 相對地**加重了雜訊多、比較難的步驟**，那些步驟決定了整體結構，對視覺品質影響大。代價是 log-likelihood 變差（Nichol & Dhariwal 2021 用「學習 $\sigma_t$」的混合損失把它補回來）。

### 2.7 演算法

**訓練（Algorithm 1）**：重複直到收斂

1. $x_0 \sim$ 資料，$t \sim \mathcal{U}\{1, \dots, T\}$，$\epsilon \sim \mathcal{N}(0, I)$
2. 對 $\left\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\ t)\right\|^2$ 做一步梯度下降

**取樣（Algorithm 2）**：

1. $x_T \sim \mathcal{N}(0, I)$
2. 對 $t = T, \dots, 1$：$z \sim \mathcal{N}(0, I)$（$t = 1$ 時 $z = 0$），

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z \tag{4.10}$$

3. 回傳 $x_0$

$\sigma_t^2$ 取 $\beta_t$ 或 $\tilde\beta_t$，Ho 等人回報兩者結果相近（前者對應 $x_0 \sim \mathcal{N}(0, I)$ 時的最佳值，後者對應 $x_0$ 是單一點時的最佳值）。

### 2.8 損失的下限：最佳去噪器是條件期望

MSE 的最佳解是條件期望：

$$\epsilon_\theta^*(x_t, t) = \mathbb{E}[\epsilon \mid x_t]$$

此時的損失是 $\mathbb{E}\,\mathrm{Var}(\epsilon \mid x_t) > 0$，也就是 §1.3 說的「猜平均」。這個下限隨 $t$ 變化：

- $t$ 很大：$x_t$ 幾乎就是 $\epsilon$，很好猜，損失接近 0；
- $t$ 很小：雜訊比資料本身的細節還小，網路分不出哪些是雜訊，損失反而高。

實驗 §5 會把這條曲線畫出來。

### 2.9 三種等價的參數化

因為 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$，給定 $x_t$ 後，預測 $\epsilon$、預測 $x_0$、預測 $v$ 可以互相換算：

| 預測 | 定義 | 換算 |
|---|---|---|
| $\epsilon$-prediction（DDPM） | $\hat\epsilon$ | $\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$ |
| $x_0$-prediction | $\hat x_0$ | $\hat\epsilon = (x_t - \sqrt{\bar\alpha_t}\,\hat x_0)/\sqrt{1-\bar\alpha_t}$ |
| $v$-prediction（Salimans & Ho 2022） | $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1-\bar\alpha_t}\,x_0$ | $\hat x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,\hat v$ |

它們的最佳解彼此對應，差別在**隱含的損失權重**與**數值穩定性**。例如 $t$ 接近 $T$ 時，由 $\hat\epsilon$ 反推 $\hat x_0$ 要除以很小的 $\sqrt{\bar\alpha_t}$，誤差會被放大；$v$-prediction 沒有這個問題。2025 年的 JiT（第 11 課）更主張高維度時直接預測 $x_0$ 最好。

---

## 3. 對照程式碼

| 公式 | 位置（`src/diffusion_course/ddpm.py`） |
|---|---|
| (3.2) 前向跳躍 | `q_sample` |
| (4.9) $L_{\text{simple}}$ | `ddpm_loss`——四行，和 Algorithm 1 一一對應 |
| $\hat x_0$ 反推 | `predict_x0` |
| (4.4) 後驗均值 | `posterior_mean` |
| (4.7) 用 $\epsilon$ 寫的均值 | `mean_from_eps` |
| Algorithm 2 | `ddpm_sample` |

`ddpm_sample` 實際上走的是「先算 $\hat x_0$，再代入 (4.4)」這條路，而不是直接用 (4.7)。兩者在數學上相同（`tests/test_ddpm.py::test_x0_route_and_eps_route_give_the_same_mean` 驗證了這點），但先算 $\hat x_0$ 有個好處：影像可以先把 $\hat x_0$ 截在 $[-1, 1]$（`clip_x0=True`），避免取樣早期的離譜預測把整條鏈帶歪。

另外兩個「拿答案對答案」的測試值得一讀：

- `test_posterior_mean_matches_regression_on_samples`：真的抽 20 萬組 $(x_{t-1}, x_t)$，用線性迴歸估 $\mathbb{E}[x_{t-1} \mid x_t, x_0]$，和 (4.4) 比對。
- `test_sampling_with_the_oracle_recovers_the_data_point`：資料只有一個點 $c$ 時，$\mathbb{E}[\epsilon \mid x_t] = (x_t - \sqrt{\bar\alpha_t}\,c)/\sqrt{1-\bar\alpha_t}$ 可以精確寫出來，用它跑 1000 步取樣，必須剛好回到 $c$。

## 4. 常見誤解

- **「網路一步一步把 $x_t$ 變成 $x_{t-1}$，所以要訓練 $T$ 個網路」**：只有一個網路，$t$ 是它的輸入。
- **「訓練時也要跑 $T$ 步」**：訓練每次只抽一個 $t$；只有取樣要跑 $T$ 步。
- **「損失越低，生成品質越好」**：損失有下限（§2.8），而且 $L_{\text{simple}}$ 本來就不是 likelihood。判斷品質要看樣本本身（或 FID 等指標）。
- **「$\epsilon_\theta$ 預測出的就是當初那份雜訊」**：它預測的是 $\mathbb{E}[\epsilon \mid x_t]$，所有可能雜訊的平均。
- **「取樣的最後一步也要加雜訊」**：$t = 1$ 時 $z = 0$，最後直接輸出均值。

## 5. 練習

**想一想**

1. 從 P3 出發，完整推導 (4.4) 的 $\tilde\mu_t$ 與 $\tilde\beta_t$。
2. 把 $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$ 代入 (4.4)，驗證 (4.6)。提示：$\sqrt{\bar\alpha_{t-1}}/\sqrt{\bar\alpha_t} = 1/\sqrt{\alpha_t}$。
3. 資料只有一個點 $c$ 時，最佳 $\epsilon_\theta^*(x_t, t)$ 是什麼？此時 $L_{\text{simple}}$ 的最小值是多少？
4. 證明 $v$-prediction 的換算式 $\hat x_0 = \sqrt{\bar\alpha_t}\,x_t - \sqrt{1-\bar\alpha_t}\,\hat v$。
5. (4.8) 的權重在 $\sigma_t^2 = \beta_t$ 時等於 $\beta_t / (2\alpha_t(1-\bar\alpha_t))$。用實驗裡的 linear schedule 算出 $t = 1$ 與 $t = 1000$ 的值，比較差幾倍。

**動手改**（在 `04_ddpm_2d.ipynb`）

6. 把資料換成 `"checkerboard"`，同樣的訓練步數夠嗎？
7. 把 schedule 換成 cosine，損失對 $t$ 的曲線怎麼變？
8. 改成 $x_0$-prediction：寫一個新的損失函數讓網路輸出 $\hat x_0$，取樣時再換算成 $\hat\epsilon$。結果和 $\epsilon$-prediction 比較如何？

## 6. 延伸閱讀

- Ho 等人（2020），DDPM：<https://arxiv.org/abs/2006.11239>——§3 與附錄 A 是本課的推導來源。
- Calvin Luo（2022），〈Understanding Diffusion Models: A Unified Perspective〉：<https://arxiv.org/abs/2208.11970>——把 ELBO 推導寫得最細的教材，式子一行一行展開。
- Nakkiran 等人（2024），〈Step-by-Step Diffusion: An Elementary Tutorial〉：<https://arxiv.org/abs/2406.08929>——不用變分推論，用最少的機率知識推出 DDPM 與 DDIM。
- Nichol & Dhariwal（2021），學習 $\sigma_t$ 與混合損失：<https://arxiv.org/abs/2102.09672>
- Salimans & Ho（2022），$v$-prediction 在這篇提出：<https://arxiv.org/abs/2202.00512>
