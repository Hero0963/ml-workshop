# 第 03 課：前向過程——把資料變成雜訊

> 前置：第 02 課 ｜ 實驗：[`notebooks/03_forward_process.ipynb`](../notebooks/03_forward_process.ipynb) ｜ 預估時間：1.5 小時

## 這一課要回答的問題

- DDPM 怎麼「一步一步」加雜訊？為什麼每一步還要把圖片縮小一點？
- 為什麼可以**一步跳到第 $t$ 步**，不用真的加 $t$ 次？
- noise schedule（$\beta_t$ 怎麼排）有什麼講究？linear 和 cosine 差在哪？

---

## 1. 白話版

### 1.1 每一步：縮一點、加一點

想像一杯果汁（資料）。每一步：

1. 倒掉一點點果汁（把圖片**乘上一個略小於 1 的數**）；
2. 補上等量的水（**加上一點雜訊**）。

重複一千次，杯子裡幾乎全是水——純雜訊。

為什麼要先倒掉？如果只加水不倒，杯子會一直變滿（數值的變異數一直變大，最後爆掉）。「倒多少、補多少」配好，杯子永遠是滿的：**總變異數維持在 1 左右**。這叫 variance preserving（VP）。

### 1.2 不用真的加一千次

每一步加的雜訊都是常態分布，而常態加常態還是常態（第 02 課 P2）。所以「加了 $t$ 次雜訊」的結果，可以直接寫成：

> 第 $t$ 步的圖 ＝（剩下的原圖比例）× 原圖 ＋（累積的雜訊比例）× 一份新的雜訊

一行公式就跳到任意一步。這讓訓練非常有效率：每次隨機挑一個 $t$，直接算出那一步的樣子，不用從頭模擬。

### 1.3 schedule：雜訊要怎麼排

每一步加多少雜訊，由 $\beta_t$ 決定。排得不好會有兩種浪費：

- 太早變成純雜訊 → 後面好幾百步都在「雜訊加雜訊」，網路學不到東西。
- 太晚才有雜訊 → 前面好幾百步幾乎沒變化。

cosine schedule 讓「原圖還剩多少」下降得比較均勻，避免 linear schedule 在小圖上「太早毀掉資訊」。

---

## 2. 正式版

### 2.1 一步的轉移

選一組 $0 < \beta_1 < \beta_2 < \dots < \beta_T < 1$（DDPM 用 $T = 1000$）。定義

$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right) \tag{3.1}$$

等價地（重參數化，第 02 課 P1）：$x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon_t$，$\epsilon_t \sim \mathcal{N}(0, I)$。

**變異數守恆**：若 $\mathrm{Var}(x_{t-1}) = 1$，由 P2，$\mathrm{Var}(x_t) = (1-\beta_t)\cdot 1 + \beta_t = 1$。
這就是係數要取 $\sqrt{1-\beta_t}$ 的原因。資料先正規化到大約單位尺度（影像 $[-1, 1]$、2D 資料標準差約 1），雜訊的「大小」才有一致的意義。

### 2.2 封閉解：一步跳到第 $t$ 步

令 $\alpha_t = 1 - \beta_t$、$\bar\alpha_t = \prod_{s=1}^{t} \alpha_s$。則

$$q(x_t \mid x_0) = \mathcal{N}\left(x_t;\ \sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t) I\right), \qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon \tag{3.2}$$

**推導（兩步看懂 $t$ 步）**：

$$x_t = \sqrt{\alpha_t}\,x_{t-1} + \sqrt{1-\alpha_t}\,\epsilon_t = \sqrt{\alpha_t}\left(\sqrt{\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_{t-1}}\,\epsilon_{t-1}\right) + \sqrt{1-\alpha_t}\,\epsilon_t$$

兩個獨立雜訊項合併（P2）：變異數是 $\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = 1 - \alpha_t\alpha_{t-1}$。所以

$$x_t = \sqrt{\alpha_t\alpha_{t-1}}\,x_{t-2} + \sqrt{1-\alpha_t\alpha_{t-1}}\,\bar\epsilon$$

歸納下去就是 (3.2)。

(3.2) 有兩個讀法：

- **訊號比例** $\sqrt{\bar\alpha_t}$：原圖還剩多少；
- **雜訊比例** $\sqrt{1-\bar\alpha_t}$：雜訊佔多少。兩者平方和為 1。

### 2.3 訊雜比（SNR）

$$\mathrm{SNR}(t) = \frac{\bar\alpha_t}{1-\bar\alpha_t} \tag{3.3}$$

它從很大（幾乎沒雜訊）單調下降到接近 0（幾乎純雜訊）。**schedule 的本質就是「SNR 隨 $t$ 怎麼下降」**，通常畫 $\log \mathrm{SNR}$ 來比較。
Kingma 等人（Variational Diffusion Models, 2021）證明，在連續時間的極限下，ELBO 只取決於 SNR 的兩個端點，而不取決於中間怎麼排——但**訓練效率與取樣品質**仍然很依賴 schedule。

### 2.4 兩種常見的 schedule

**Linear**（Ho 等人 2020）：$\beta_t$ 從 $10^{-4}$ 線性增加到 $0.02$，$T = 1000$。$\bar\alpha_T \approx 4 \times 10^{-5}$，終點非常接近純雜訊。

**Cosine**（Nichol & Dhariwal 2021）：直接設計 $\bar\alpha_t$，

$$\bar\alpha_t = \frac{f(t)}{f(0)}, \qquad f(t) = \cos^2\left(\frac{t/T + s}{1 + s}\cdot\frac{\pi}{2}\right), \quad s = 0.008 \tag{3.4}$$

再由 $\beta_t = 1 - \bar\alpha_t / \bar\alpha_{t-1}$ 反推，並把 $\beta_t$ 截在 0.999 以下（避免最後一步 $\beta_T = 1$ 造成除以零）。
Nichol & Dhariwal 觀察到：linear schedule 在 $64 \times 64$ 以下的影像上，後段大約兩成的步數幾乎已是純雜訊，對學習貢獻很小；cosine 讓 $\bar\alpha_t$ 在中段下降得比較線性。

### 2.5 兩個進階提醒

- **終點要真的是雜訊**：取樣時我們從 $\mathcal{N}(0, I)$ 出發，若 $\bar\alpha_T$ 不夠接近 0，訓練時看到的 $x_T$ 仍帶有原圖的一點點訊號，和取樣起點對不上。Lin 等人（2023）〈Common Diffusion Noise Schedules and Sample Steps are Flawed〉指出 Stable Diffusion 早期的 schedule 就有這個問題（生成的圖亮度偏向中間值），建議把終點 SNR 調成 0。
- **解析度會改變「同一個 $t$」的意義**：高解析度影像的相鄰像素高度相關，同樣的雜訊比例下，平均掉雜訊後原圖還「看得出來」，資訊被破壞得比較慢。所以高解析度需要把 schedule 往更多雜訊的方向平移（Hoogeboom 等人 2023〈simple diffusion〉；Stable Diffusion 3 的 timestep shift）。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| linear／cosine 的 $\beta_t$ | `linear_betas`、`cosine_betas`（`src/diffusion_course/schedules.py`） |
| $\alpha_t$、$\bar\alpha_t$、SNR 等所有常數 | `NoiseSchedule`（同上） |
| (3.2) 一步跳到第 $t$ 步 | `q_sample`（`src/diffusion_course/ddpm.py`） |
| 驗證「一千小步 = 一大步」 | `tests/test_ddpm.py::test_many_small_steps_equal_one_big_jump` |

**索引慣例**：程式裡的步數 $i = 0, \dots, T-1$ 對應論文的 $t = i + 1$。所以 `alpha_bars[0]` 已經帶一點點雜訊，`alpha_bars[T-1]` 幾乎是純雜訊。所有模型收到的時間是 $i / T \in [0, 1)$（`NoiseSchedule.timestep_input`）。

## 4. 常見誤解

- **「$\beta_t$ 是第 $t$ 步的雜訊標準差」**：$\beta_t$ 是**變異數**，標準差是 $\sqrt{\beta_t}$。
- **「$x_t$ 是 $x_{t-1}$ 加上雜訊」**：還要先乘 $\sqrt{1-\beta_t}$ 縮小。少了這一步就變成 variance exploding 的另一種過程（NCSN／VE-SDE，第 05 課）。
- **「前向過程需要一步一步模擬」**：訓練時永遠用 (3.2) 直接跳；逐步模擬只在驗證時用。
- **「資料尺度無所謂」**：若影像沒正規化到 $[-1, 1]$ 而是 $[0, 255]$，雜訊相對於訊號小了幾百倍，整個 schedule 就失效了。

## 5. 練習

**想一想**

1. 用歸納法完整證明 (3.2)。
2. 若 $\mathrm{Var}(x_0) = v \ne 1$，$\mathrm{Var}(x_t)$ 等於多少？$t$ 很大時趨向多少？
3. linear schedule 下，$\bar\alpha_t \approx \exp\left(-\sum_s \beta_s\right)$ 為什麼是好的近似？用它估計 $\bar\alpha_{500}$，再和實驗裡印出來的值比較。
4. 為什麼 cosine schedule 要加偏移 $s = 0.008$？$s = 0$ 時 $t$ 很小的 $\beta_t$ 會怎樣？

**動手改**（在 `03_forward_process.ipynb`）

5. 把 $T$ 改成 100（`NoiseSchedule.create("linear", 100)`，程式會自動放大 $\beta$），畫出 $\bar\alpha_t$ 與原本 $T = 1000$ 的比較（橫軸用 $t/T$）。
6. 換成 Fashion-MNIST（`image_dataset("fashion_mnist")`），同一個 $t$ 下資訊被破壞的程度一樣嗎？

## 6. 延伸閱讀

- Ho 等人（2020），DDPM §2 與 §4：<https://arxiv.org/abs/2006.11239>
- Nichol & Dhariwal（2021），〈Improved Denoising Diffusion Probabilistic Models〉§3.2（cosine schedule）：<https://arxiv.org/abs/2102.09672>
- Kingma 等人（2021），〈Variational Diffusion Models〉（SNR 觀點）：<https://arxiv.org/abs/2107.00630>
- Lin 等人（2023），終點 SNR 問題：<https://arxiv.org/abs/2305.08891>
- Hoogeboom 等人（2023），高解析度的 schedule 平移：<https://arxiv.org/abs/2301.11093>
