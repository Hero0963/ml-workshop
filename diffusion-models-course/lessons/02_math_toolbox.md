# 第 02 課：數學工具箱——高斯、score 與 Langevin 動力學

> 前置：第 01 課 ｜ 實驗：[`notebooks/02_langevin.ipynb`](../notebooks/02_langevin.ipynb) ｜ 預估時間：2 小時

## 這一課要回答的問題

- 擴散模型的推導裡，常態分布有哪幾個性質會一直被用到？
- 什麼是 **score**？為什麼「知道 score」就足以生成樣本？
- 只靠 score 取樣會在哪裡失敗？為什麼「加很多種程度的雜訊」剛好能解決這些失敗？（這就是擴散模型的出發點。）

---

## 1. 白話版

### 1.1 score 是一張「往人多的地方走」的地圖

想像一張地形圖，高度代表「這種圖出現的機率」：山頂是最常見的手寫數字，山谷是亂七八糟的雜訊。
**score** 就是在地圖上每個點畫一個箭頭，指向「往上爬最快的方向」，箭頭越長代表坡越陡。

有了這張箭頭地圖，就算不知道山有多高，也能往山頂走。

### 1.2 Langevin 動力學：一邊爬山一邊抖

如果只沿著箭頭爬，所有人最後都擠在山頂上——你只會得到「最典型」的那一張圖，沒有多樣性。
Langevin 動力學的做法是：**每走一步就隨機抖一下**。
抖動讓人不會全擠在山頂，而是按照高度比例散開：高的地方人多，矮的地方人少。走夠久以後，人群的分布**恰好就是**我們要的機率分布。

### 1.3 可是地圖只在「有人去過的地方」畫得準

真實情況下，箭頭地圖是神經網路從資料學出來的。資料只出現在山上，所以：

- **荒野裡的箭頭是亂畫的**：從隨機雜訊出發的人，一開始就站在荒野，會被亂指路。
- **兩座山之間走不過去**：如果有兩座山（例如「1」和「7」），中間的山谷很深，人一旦爬上其中一座就下不來，結果兩座山上的人數比例是錯的。

解法：**先把地形「模糊化」**。把資料加上很多雜訊，山會變矮變寬、連成一片，荒野也被覆蓋到，箭頭到處都畫得準。
先在最模糊的地形上走，再一點一點換成比較清楚的地形——這就是 annealed Langevin，也是擴散模型的核心精神。

---

## 2. 正式版

### 2.1 常態分布的四個性質

以下 $\mathcal{N}(\mu, \sigma^2 I)$ 表示 $d$ 維等向常態分布，密度為

$$p(x) = (2\pi\sigma^2)^{-d/2} \exp\left(-\frac{\|x-\mu\|^2}{2\sigma^2}\right)$$

**(P1) 重參數化**：要從 $\mathcal{N}(\mu, \sigma^2 I)$ 取樣，只要取 $\epsilon \sim \mathcal{N}(0, I)$，令

$$x = \mu + \sigma\epsilon$$

隨機性全部集中在 $\epsilon$，$\mu$、$\sigma$ 是確定的，因此可以對它們微分。擴散模型的 $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ 就是這個形式。

**(P2) 獨立常態的線性組合還是常態**：若 $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$、$Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ 且獨立，則

$$aX + bY \sim \mathcal{N}(a\mu_1 + b\mu_2,\ a^2\sigma_1^2 + b^2\sigma_2^2)$$

均值線性相加，**變異數按平方係數相加**。第 03 課用它把一千步加噪壓成一個公式。

**(P3) 常態乘常態還是常態**：兩個關於同一個變數 $x$ 的常態密度相乘，結果（正規化後）仍是常態，新的均值是「以精確度（變異數的倒數）加權的平均」：

$$\mathcal{N}(x; a, A)\,\mathcal{N}(x; b, B) \propto \mathcal{N}\left(x;\ \frac{B a + A b}{A + B},\ \frac{AB}{A+B}\right)$$

第 04 課用它（配合貝氏定理）求出反向一步的後驗 $q(x_{t-1} \mid x_t, x_0)$。

**(P4) 兩個常態之間的 KL 散度有封閉解**。一維時

$$\mathrm{KL}\left(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)\right) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

當兩邊變異數相同（都是 $\sigma^2 I$）時，只剩下 $\frac{\|\mu_1-\mu_2\|^2}{2\sigma^2}$——**KL 變成均值的平方誤差**。這就是擴散模型的損失函數最後會變成 MSE 的原因。

### 2.2 Score function

**定義**：機率密度 $p(x)$ 的 score 是

$$s(x) = \nabla_x \log p(x)$$

注意是對**資料 $x$** 微分，不是對參數。

**例子**：

- 常態 $\mathcal{N}(\mu, \sigma^2 I)$：$s(x) = -\frac{x - \mu}{\sigma^2}$，箭頭永遠指向均值，離越遠越長。
- 常態混合 $p(x) = \sum_k w_k\,\mathcal{N}(x; \mu_k, \sigma^2 I)$：

$$s(x) = \sum_k r_k(x)\,\frac{\mu_k - x}{\sigma^2}, \qquad r_k(x) = \frac{w_k\,\mathcal{N}(x;\mu_k,\sigma^2 I)}{\sum_j w_j\,\mathcal{N}(x;\mu_j,\sigma^2 I)}$$

  每個成分往自己的中心拉，拉力用「$x$ 屬於第 $k$ 群的後驗機率 $r_k$」加權。程式碼：`GaussianMixture.score`（`src/diffusion_course/data.py`）。

**score 最大的好處：不需要正規化常數。** 若 $p(x) = \tilde p(x) / Z$，其中 $Z = \int \tilde p$ 通常算不出來，則

$$\nabla_x \log p(x) = \nabla_x \log \tilde p(x) - \underbrace{\nabla_x \log Z}_{=0}$$

學 $p$ 本身要保證積分為 1，很難；學 score 沒有這個限制，任何輸出向量的網路都可以。

### 2.3 Langevin 動力學

給定 score，從任意起點 $x_0$ 重複

$$x_{k+1} = x_k + \eta\,\nabla_x \log p(x_k) + \sqrt{2\eta}\,z_k, \qquad z_k \sim \mathcal{N}(0, I)$$

當步長 $\eta \to 0$、步數 $K \to \infty$，$x_K$ 的分布收斂到 $p$。這是連續時間隨機微分方程 $dx = \nabla_x \log p(x)\,dt + \sqrt{2}\,dW_t$ 的 Euler–Maruyama 離散化，它的穩態分布正是 $p$（可由 Fokker–Planck 方程驗證）。

兩項各司其職：

- **漂移項** $\eta\,\nabla \log p$：往高機率處爬。只有這項就是梯度上升，會收斂到眾數（mode）。
- **擴散項** $\sqrt{2\eta}\,z$：隨機抖動。係數 $\sqrt{2\eta}$ 不是隨便選的——它恰好讓漂移與擴散平衡在 $p$ 上。

**有限步長會有偏差**。以標準常態為例（$\nabla \log p = -x$），穩態變異數 $v$ 滿足 $v = (1-\eta)^2 v + 2\eta$，解得

$$v = \frac{1}{1 - \eta/2}$$

$\eta = 0.1$ 時 $v \approx 1.05$。實驗裡會實際量到這個偏差。（MALA 用 Metropolis 修正來消除它，擴散模型則靠「最後幾步雜訊很小」讓偏差可以忽略。）

### 2.4 用學來的 score 做 Langevin 會遇到的三個問題

Song & Ermon（2019）指出，若 score 是從資料學來的，直接用 Langevin 取樣會失敗：

1. **流形假設**：真實影像集中在高維空間中的低維流形上，流形外 $p = 0$，$\log p$ 沒有定義，score 也就沒有意義。
2. **低密度區估不準**：訓練資料幾乎不會出現在低密度區，網路在那裡的 score 是外插的。偏偏 Langevin 從隨機起點出發時，正好都在低密度區。
3. **模式之間混合很慢**：兩個分得很開的模式之間，Langevin 幾乎跨不過去。更糟的是，score 對「各模式佔多少比例」幾乎不敏感——遠離邊界的地方，$\nabla \log p$ 只看得到最近那一群。結果各模式的樣本比例由「起點落在哪一邊」決定，而不是由真正的權重 $w_k$ 決定。

### 2.5 解法：加雜訊，而且加很多種程度

把資料加上高斯雜訊 $\tilde x = x + \sigma\epsilon$，得到「模糊化」的分布

$$p_\sigma(\tilde x) = \int p(x)\,\mathcal{N}(\tilde x;\, x, \sigma^2 I)\,dx$$

也就是 $p$ 與高斯做卷積。$\sigma$ 大時：

- 分布填滿整個空間 → 問題 1、2 消失（到處都有資料，score 到處都學得到）。
- 各模式變寬、連成一片 → 問題 3 消失（在模糊的分布上，各模式的權重會影響 score）。

但 $\sigma$ 大時的 $p_\sigma$ 離真正的 $p$ 很遠。解法是**一次學一整排 $\sigma_1 > \sigma_2 > \dots > \sigma_L$ 的 score**（用同一個網路，把 $\sigma$ 當輸入），取樣時從最大的 $\sigma$ 開始跑 Langevin，逐步換到小的：**annealed Langevin dynamics**（Song & Ermon 2019）。步長隨 $\sigma_i^2$ 縮放，$\eta_i = \eta_{\min}\,\sigma_i^2/\sigma_L^2$，讓每一層的步伐相對於該層的尺度一樣大。

常態混合加雜訊後仍是常態混合，只是每個成分的變異數變成 $\sigma_{\text{data}}^2 + \sigma^2$。所以在實驗裡，我們可以**精確算出**每個雜訊程度的 score，不必訓練網路，先把 annealed Langevin 的效果看清楚。

> **這就是擴散模型的雛形。** 「一排由大到小的雜訊程度」會在第 03 課變成 noise schedule；「用一個網路學所有雜訊程度的 score」會在第 04、05 課變成 $\epsilon_\theta(x_t, t)$。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| 常態混合的密度、score、加噪後的分布 | `GaussianMixture.log_prob` / `.score` / `.diffused`（`src/diffusion_course/data.py`） |
| Langevin 動力學 | `langevin_dynamics`（`src/diffusion_course/score.py`） |
| Annealed Langevin | `annealed_langevin_dynamics`（同上） |
| score 與 autograd 的一致性檢查 | `tests/test_score.py::test_closed_form_score_matches_autograd` |

## 4. 常見誤解

- **「score 是對參數的梯度」**：統計學裡的 score function 有時指 $\nabla_\theta \log p_\theta(x)$，但擴散模型文獻裡的 score 一律是 $\nabla_x \log p(x)$。
- **「Langevin 的雜訊越小越好」**：雜訊項係數是 $\sqrt{2\eta}$，不能自己調。去掉雜訊就變成梯度上升，只會收斂到眾數。
- **「score 準就能取樣」**：score 在高密度區準還不夠，Langevin 起點在低密度區，而且模式權重需要混合才能校正——這正是要加雜訊的原因。

## 5. 練習

**想一想**

1. 證明 $\mathcal{N}(\mu, \sigma^2 I)$ 的 score 是 $-(x-\mu)/\sigma^2$。
2. 推導 §2.3 的穩態變異數 $v = 1/(1-\eta/2)$。提示：穩態時 $x_{k+1}$ 與 $x_k$ 同分布。
3. 兩個常態混合成分相距很遠時，在其中一個中心附近，score 幾乎只由該成分決定。用 $r_k(x)$ 的公式解釋，並說明這為什麼讓 score 對權重 $w_k$ 「看不見」。
4. 證明常態混合加上 $\mathcal{N}(0, \sigma^2 I)$ 的獨立雜訊後，仍是常態混合，變異數變成 $\sigma_{\text{data}}^2 + \sigma^2$（用 P2）。

**動手改**（在 `02_langevin.ipynb`）

5. 把 Langevin 的步長從 0.01 改成 0.3，觀察樣本分布怎麼變形。
6. 把雙模式實驗的權重改成 0.95／0.05，annealed Langevin 還能抓對比例嗎？若改成只有 2 個雜訊等級呢？

## 6. 延伸閱讀

- Yang Song，〈Generative Modeling by Estimating Gradients of the Data Distribution〉（部落格，2021-05）：<https://yang-song.net/blog/2021/score/> ——score 觀點最好的入門，本課 §2.4 的三個問題就出自這裡。
- Song & Ermon（2019），NCSN 原始論文：<https://arxiv.org/abs/1907.05600>
- Song & Ermon（2020），〈Improved Techniques for Training Score-Based Generative Models〉：<https://arxiv.org/abs/2006.09011> ——怎麼選 $\sigma$ 序列與步長。
