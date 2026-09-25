# 第 07 課：加速取樣——DDIM 與 ODE 求解器

> 前置：第 05、06 課 ｜ 實驗：[`notebooks/07_ddim_sampling.ipynb`](../notebooks/07_ddim_sampling.ipynb)（需要 Lab 06 存的模型） ｜ 預估時間：2 小時

## 這一課要回答的問題

- DDPM 為什麼要跑 1000 步？能不能**不重新訓練**就少跑幾步？
- DDIM 是怎麼做到的？為什麼 $\eta = 0$ 時完全沒有隨機性？
- DDIM 和第 05 課的 probability flow ODE 是什麼關係？還有哪些更快的求解器？

---

## 1. 白話版

### 1.1 訓練時，模型從沒看過「一步一步」

回想訓練：每次隨機挑一個 $t$，**直接**用公式跳到 $x_t$，請網路猜雜訊。
網路學到的只是「看到雜訊程度為 $t$ 的圖，猜出雜訊」；它根本不知道、也不在乎我們取樣時是一步一步走的。

所以取樣時，我們有很大的自由：只要每一步「站在網路認得的雜訊程度上」，**要走幾步、怎麼走都可以**。

### 1.2 DDIM 的一步

每一步做兩件事：

1. 用網路猜出雜訊，算出「網路心中的乾淨圖」$\hat x_0$；
2. 把 $\hat x_0$ **重新加上雜訊**，但只加到**下一個（比較小的）雜訊等級**——而且這次加的雜訊，就用剛剛猜出來的那一份，不再抽新的。

因為第二步不抽新雜訊，整個過程是確定的：同一個起點永遠得到同一張圖。
而且下一個雜訊等級可以跳很遠：從 $t = 999$ 直接跳到 $t = 949$，50 步就走完全程。

### 1.3 確定性帶來的好處

起點雜訊和生成的圖是**一一對應**的，就像一本對照表。這讓我們可以在兩個雜訊之間平滑地內插，得到兩張圖之間平滑過渡的一系列圖；也可以把一張真實圖反推回它的雜訊，再做編輯。

---

## 2. 正式版

### 2.1 關鍵觀察：訓練只用到邊際分布

$L_{\text{simple}}$（式 4.9）只依賴 $q(x_t \mid x_0)$，不依賴 $q(x_t \mid x_{t-1})$ 的鏈結構。
所以任何**邊際分布相同**的前向過程，都對應同一個訓練目標、同一個最佳網路。Song、Meng、Ermon（2020）構造了一整族這樣的過程。

### 2.2 非馬可夫的前向過程族

對任意 $\sigma_t \ge 0$，定義

$$q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left(\sqrt{\bar\alpha_{t-1}}\,x_0 + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}},\ \sigma_t^2 I\right) \tag{7.1}$$

可以驗證：若 $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$，則由 (7.1) 得到的 $x_{t-1}$ 也滿足 $q(x_{t-1} \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\,x_0, (1-\bar\alpha_{t-1})I)$（練習 1）。邊際相同，所以 DDPM 訓練出的網路可以直接用。

注意 $\frac{x_t - \sqrt{\bar\alpha_t}\,x_0}{\sqrt{1-\bar\alpha_t}}$ 正是 $\epsilon$：(7.1) 就是「用 $x_0$ 和**同一份** $\epsilon$，重新組合出下一個雜訊等級，再額外加 $\sigma_t$ 的新雜訊」。

### 2.3 DDIM 的更新式

取樣時把 $x_0$、$\epsilon$ 換成網路的估計：

$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}}, \qquad \hat\epsilon = \epsilon_\theta(x_t, t)$$

$$x_{t-1} = \underbrace{\sqrt{\bar\alpha_{t-1}}\,\hat x_0}_{\text{預測的乾淨圖}} + \underbrace{\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\hat\epsilon}_{\text{指向 } x_t \text{ 的方向}} + \underbrace{\sigma_t z}_{\text{新雜訊}} \tag{7.2}$$

用一個參數 $\eta$ 控制隨機性：

$$\sigma_t = \eta\,\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\,\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}} \tag{7.3}$$

- $\eta = 1$：$\sigma_t^2 = \tilde\beta_t$，(7.2) 就是 DDPM（式 4.10，取後驗變異數）。
- $\eta = 0$：完全確定，這就是 **DDIM**。

### 2.4 跳步

(7.2) 中的 $t-1$ 可以換成任何更小的 $s$：只要把 $\bar\alpha_{t-1}$ 換成 $\bar\alpha_s$。取一個遞減的子序列 $\tau_1 > \tau_2 > \dots > \tau_S$（例如從 999 到 0 均勻取 50 個），就能用 $S$ 步取樣。程式碼：`ddim_timesteps`。

### 2.5 DDIM 就是 probability flow ODE 的離散化

令 $\bar x = x/\sqrt{\bar\alpha}$、$\bar\sigma = \sqrt{(1-\bar\alpha)/\bar\alpha}$（也就是第 05 課 §2.3 的「VE 座標」，$\bar\sigma = 1/\sqrt{\mathrm{SNR}}$）。把 $\eta = 0$ 的 (7.2) 兩邊除以 $\sqrt{\bar\alpha_{t-1}}$：

$$\bar x_{t-1} = \bar x_t + \left(\bar\sigma_{t-1} - \bar\sigma_t\right)\hat\epsilon$$

這正是 ODE

$$\frac{d\bar x}{d\bar\sigma} = \epsilon_\theta(x, t) \tag{7.4}$$

的 **Euler 法**（Song 等人 2020 §4.3）。而 (7.4) 就是 VP-SDE 的 probability flow ODE 換了座標的樣子。
所以：**DDIM ＝ 在一個聰明的座標系裡，用最簡單的 Euler 法解 probability flow ODE**。這個座標系讓軌跡接近直線，因此 Euler 法大步走也不會偏太多。

### 2.6 更好的求解器

既然是解 ODE，數值分析的工具都能用：

| 求解器 | 階數 | 說明 |
|---|---|---|
| DDIM | 1 | Euler 法；20–50 步通常夠用 |
| Heun（EDM） | 2 | 每步多評估一次網路做修正；Karras 等人 2022 用約 35 次評估達到當時最佳 FID |
| DPM-Solver／DPM-Solver++ | 2–3 | 利用 ODE「線性部分可以精確積分」的半線性結構；10–20 步 |
| UniPC | 可變 | predictor–corrector 架構，少步數時表現好 |

每一步的成本是「網路評估次數」（NFE），比較時應該用 NFE 而不是步數。

### 2.7 確定性 vs. 隨機性

| | $\eta = 0$（ODE） | $\eta > 0$（SDE） |
|---|---|---|
| 少步數時 | 通常較好 | 雜訊會被放大，常較差 |
| 誤差修正 | 無，誤差會累積 | 新雜訊能「洗掉」部分誤差 |
| 對應關係 | 雜訊 ↔ 影像一一對應，可插值、可反推 | 無 |

EDM 發現在步數夠多時，適量的隨機性（他們稱為 churn）能提升品質。

### 2.8 插值要用球面插值

兩個高維常態樣本 $z_1, z_2$ 的**直線中點** $(z_1 + z_2)/2$ 長度只有原本的約 $1/\sqrt{2}$——高維常態樣本幾乎都落在半徑 $\sqrt{d}$ 的球殼上，中點掉進了殼內部，是模型從沒見過的「太乾淨的雜訊」。
所以在雜訊空間要用**球面線性插值**（slerp）：

$$\mathrm{slerp}(z_1, z_2; \lambda) = \frac{\sin((1-\lambda)\Omega)}{\sin\Omega}\,z_1 + \frac{\sin(\lambda\Omega)}{\sin\Omega}\,z_2, \qquad \cos\Omega = \frac{z_1\cdot z_2}{\|z_1\|\|z_2\|}$$

### 2.9 少步數的極限

就算用最好的 ODE 求解器，網路本身學的是「沿著彎曲的路徑走一小段」的方向；步數降到個位數時，離散化誤差仍然很大。要做到 1–4 步，需要改變訓練本身：讓路徑變直（第 09 課 rectified flow），或把多步蒸餾成少步（第 11 課：progressive distillation、consistency models 等）。

---

## 3. 對照程式碼

| 概念 | 位置（`src/diffusion_course/ddim.py`） |
|---|---|
| 跳步子序列 | `ddim_timesteps` |
| (7.2)、(7.3) | `ddim_sample`——`eta` 參數控制隨機性 |
| $\hat x_0$ 截斷後重算 $\hat\epsilon$ | `ddim_sample` 裡 `clip_x0` 的分支 |
| 測試：10 步就能精確回到單點資料 | `tests/test_ddim.py::test_ten_step_ddim_recovers_the_data_point` |

截斷 $\hat x_0$ 後要**重算** $\hat\epsilon$，讓 (7.2) 的兩項彼此一致；否則「預測的乾淨圖」與「指向 $x_t$ 的方向」會互相矛盾。

## 4. 常見誤解

- **「DDIM 需要另外訓練」**：不需要，同一個 DDPM 模型直接用。
- **「DDIM 是一種新的模型」**：DDIM 是一種**取樣方法**（加上一族前向過程的理論），不是新架構。
- **「步數越少越糟，所以 DDIM 只是犧牲品質換速度」**：DDIM 從 100 步降到 50 步，品質通常幾乎不變；真正的品質下降出現在 10 步以下。（實驗裡的小模型，50 步 DDIM 比 1000 步 DDPM 略差，但差距來自隨機性而不是步數：同樣 50 步，$\eta = 1$ 反而最好。模型越不準，隨機性「洗掉誤差」的好處越明顯，見 §2.7。）
- **「在雜訊空間直接線性插值就好」**：高維下會掉到球殼內部，要用 slerp。

## 5. 練習

**想一想**

1. 驗證 (7.1)：若 $x_t \sim \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t)I)$，則 $x_{t-1}$ 的邊際是 $\mathcal{N}(\sqrt{\bar\alpha_{t-1}}\,x_0, (1-\bar\alpha_{t-1})I)$。
2. 證明 $\eta = 1$ 時 (7.3) 的 $\sigma_t^2$ 等於 $\tilde\beta_t$（式 4.4）。
3. 把 (7.2)（$\eta = 0$）兩邊除以 $\sqrt{\bar\alpha_{t-1}}$，推出 §2.5 的 Euler 形式。
4. $d$ 維標準常態向量長度的平方期望是多少？兩個獨立樣本的中點長度約是多少？

**動手改**（在 `07_ddim_sampling.ipynb`）

5. 實作 Heun 法（2 階）版本的 DDIM：在 $\bar x$、$\bar\sigma$ 座標下做預測–修正。相同 NFE 下與 DDIM 比較。
6. 實作 DDIM inversion：把一張真實的測試數字，用 $\eta = 0$ 從 $t = 0$ 往 $t = 999$ 反向走，得到它的雜訊，再生成回來，看能不能重建。

## 6. 延伸閱讀

- Song、Meng、Ermon（2020），DDIM：<https://arxiv.org/abs/2010.02502>
- Karras 等人（2022），EDM（Heun 求解器、隨機性的取捨）：<https://arxiv.org/abs/2206.00364>
- Lu 等人（2022），DPM-Solver：<https://arxiv.org/abs/2206.00927>；DPM-Solver++：<https://arxiv.org/abs/2211.01095>
- Zhao 等人（2023），UniPC：<https://arxiv.org/abs/2302.04867>
- Sander Dieleman，〈Perspectives on diffusion〉（2023）：<https://sander.ai/2023/07/20/perspectives.html>——把擴散看成 ODE、自編碼器、自迴歸等多種觀點。
