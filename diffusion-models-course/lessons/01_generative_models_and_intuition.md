# 第 01 課：生成模型全景與擴散的直覺

> 前置：第 00 課 ｜ 本課沒有實驗 ｜ 預估時間：1 小時

## 這一課要回答的問題

- 「生成模型」到底在學什麼？為什麼它比分類難？
- 擴散模型的核心想法是什麼？為什麼這個看似繞遠路的做法，最後打敗了 GAN？
- 擴散模型這十年怎麼演變過來的？

---

## 1. 白話版

### 1.1 生成模型在學什麼

想像你看過一萬張手寫數字。分類模型要學的是「看到一張圖，說出它是幾」——答案只有 10 種。
生成模型要學的是「**手寫數字長什麼樣子**」，然後**自己寫出一張從沒看過、但一看就是手寫數字的圖**。

難在哪？一張 28×28 的灰階圖有 784 個像素。如果每個像素隨機亂填，你幾乎百分之百得到雜訊畫面。
「看起來像數字」的圖，在所有可能的圖裡只佔極小極小的一塊。生成模型要找到那一小塊，還要知道那一塊裡哪裡比較「常見」（例如 1 通常是直的，偶爾斜一點）。

### 1.2 擴散的核心想法：破壞很容易，一次修一點點也不難

把一張照片變成雜訊很簡單：每次撒一點胡椒粉（雜訊），撒一千次，照片就變成一片灰色雜點。這一步**完全不用學**。

反過來，從一片雜點變回照片很難——但如果只要求「**把撒了 501 次的圖，修成撒了 500 次的樣子**」，就簡單多了：兩張圖幾乎一樣，只差一點點雜訊。

擴散模型就是：

1. **訓練時**：拿真的照片，隨機撒一些雜訊，請神經網路猜「剛剛撒了什麼雜訊」。這是一個普通的監督式學習問題，答案（撒下去的雜訊）我們自己知道。
2. **生成時**：從一片純雜訊開始，請網路猜雜訊、扣掉一點點，重複一千次，雜訊就慢慢「顯影」成一張照片。

名字來自物理：一滴墨水滴進水裡會**擴散**開來，最後均勻分布。擴散模型學的是「把影片倒著播」——讓均勻的墨水重新聚回一滴。

### 1.3 為什麼這招好

| 別的做法 | 它的麻煩 | 擴散怎麼避開 |
|---|---|---|
| GAN（生成器和鑑別器對打） | 訓練不穩定；容易只學會幾種樣子（mode collapse） | 損失函數只是 MSE，穩定；而且每張訓練圖都會被用來監督，不會漏掉某類樣子 |
| VAE（壓縮再解壓） | 生成的圖常常糊糊的 | 分一千小步修，每步都只要修一點，細節保得住 |
| 自迴歸（像 GPT 逐個像素生成） | 影像像素太多，一個一個生很慢，而且要決定像素順序 | 每一步同時更新所有像素 |

代價是**生成很慢**：要跑很多步。第 07、09、11 課都在處理這件事。

---

## 2. 正式版

### 2.1 問題設定

資料 $x \in \mathbb{R}^d$ 來自一個未知分布 $p_{\text{data}}(x)$，我們只有它的樣本 $\{x^{(1)}, \dots, x^{(N)}\}$。
生成模型要學一個分布 $p_\theta(x)$，使得：

1. $p_\theta \approx p_{\text{data}}$（在某種距離或散度的意義下）；
2. 能**有效率地從 $p_\theta$ 取樣**。

有些模型還能算 $p_\theta(x)$ 的值（likelihood），有些不能——這是各家族的主要差異之一。

### 2.2 生成模型家族

| 家族 | 代表作 | 怎麼取樣 | 能算 likelihood？ | 主要取捨 |
|---|---|---|---|---|
| 自迴歸 | PixelCNN、GPT | $p(x) = \prod_i p(x_i \mid x_{<i})$，逐維生成 | 精確 | 取樣是序列的，高維很慢 |
| VAE | Kingma & Welling 2013 | 取潛變數 $z \sim p(z)$ 再解碼 | 下界（ELBO） | 樣本偏模糊 |
| GAN | Goodfellow 等人 2014 | 一次前向傳遞 | 不能 | 快、銳利，但訓練不穩、mode collapse |
| Normalizing flow | RealNVP、Glow | 可逆網路一次變換 | 精確 | 架構必須可逆，表達力受限 |
| **擴散／score-based** | DDPM、NCSN、Score SDE | 迭代去噪（或解 ODE／SDE） | 下界；ODE 形式可精確計算 | 品質高、訓練穩；取樣需多步 |
| **Flow matching** | Lipman 等人 2022、Rectified Flow | 解 ODE | 精確（透過 ODE） | 與擴散同源（第 09 課），路徑可更直 |

擴散模型可以看成**一個有 $T$ 層潛變數的 VAE**，而且編碼器（加噪）是固定的、不用學。第 04 課會從這個角度推出它的損失函數。

### 2.3 同一個模型的三種說法

這門課會反覆看到同一個東西被寫成三種語言。先把對照表放在這裡，之後每課都會回來填細節：

| 觀點 | 網路在學什麼 | 生成時做什麼 | 在哪一課 |
|---|---|---|---|
| 變分（DDPM） | 預測加進去的雜訊 $\epsilon_\theta(x_t, t)$ | 從 $x_T$ 一步步取樣 $x_{t-1} \sim p_\theta(x_{t-1} \mid x_t)$ | 03、04 |
| Score／SDE | 預測 $\nabla_x \log p_t(x)$（score） | 解反向時間 SDE，或 Langevin 動力學 | 02、05 |
| Flow／ODE | 預測速度場 $v_\theta(x, t)$ | 解一條常微分方程 | 07、09 |

三者之間可以互相換算（都是 $x_t$ 與網路輸出的**線性組合**），所以同一個訓練好的網路可以用三種方式取樣。這是整門課最重要的一個「頓悟點」。

### 2.4 擴散模型簡史

以下日期取自 arXiv 首次提交日（2026-09-25 查證）。

| 時間 | 事件 | 為什麼重要 |
|---|---|---|
| 2015-03 | Sohl-Dickstein 等人〈Deep Unsupervised Learning using Nonequilibrium Thermodynamics〉 | 第一次提出「逐步加噪、學習反轉」的框架 |
| 2019-07 | Song & Ermon〈Generative Modeling by Estimating Gradients of the Data Distribution〉（NCSN） | score 觀點：多尺度雜訊＋annealed Langevin |
| 2020-06 | Ho 等人〈Denoising Diffusion Probabilistic Models〉（DDPM） | 預測雜訊的簡單損失，影像品質首次追上 GAN |
| 2020-10 | Song 等人〈Denoising Diffusion Implicit Models〉（DDIM） | 同一個模型，少很多步取樣 |
| 2020-11 | Song 等人〈Score-Based Generative Modeling through SDEs〉 | 用 SDE 統一 DDPM 與 NCSN，引出 probability flow ODE |
| 2021-05 | Dhariwal & Nichol〈Diffusion Models Beat GANs on Image Synthesis〉 | classifier guidance；ImageNet 上正式超越 GAN |
| 2021-12 | Rombach 等人〈High-Resolution Image Synthesis with Latent Diffusion Models〉 | 在潛空間做擴散，成為 Stable Diffusion 的基礎 |
| 2022-06 | Karras 等人〈Elucidating the Design Space of Diffusion-Based Generative Models〉（EDM） | 把各種設計選擇拆開重新整理 |
| 2022-07 | Ho & Salimans〈Classifier-Free Diffusion Guidance〉 | 不用分類器的 guidance，至今幾乎所有文生圖模型都在用 |
| 2022-09／10 | Rectified Flow、Stochastic Interpolants、Flow Matching 三篇獨立提出 | flow 觀點，路徑更直、取樣更快 |
| 2022-12 | Peebles & Xie〈Scalable Diffusion Models with Transformers〉（DiT） | 用 Transformer 取代 U-Net |
| 2023-03 | Song 等人〈Consistency Models〉 | 一步生成的蒸餾路線 |
| 2024-03 | Esser 等人〈Scaling Rectified Flow Transformers〉（Stable Diffusion 3） | rectified flow ＋ MMDiT 成為文生圖主流配方 |
| 2025-02 | Nie 等人〈Large Language Diffusion Models〉（LLaDA） | 擴散式語言模型做到 8B 規模 |
| 2025-11 | Li & He〈Back to Basics: Let Denoising Generative Models Denoise〉（JiT） | 直接在像素上用純 ViT、預測乾淨影像 |
| 2026-02 | Deng 等人〈Generative Modeling via Drifting〉 | 把迭代過程搬到訓練階段，一步生成 |

---

## 3. 對照程式碼

這一課還沒有實作，但可以先看整體結構：

- 「撒雜訊」→ `diffusion_course.ddpm.q_sample`（第 03 課）
- 「請網路猜雜訊」→ `diffusion_course.ddpm.ddpm_loss`（第 04 課）
- 「從雜訊一步步修回來」→ `diffusion_course.ddpm.ddpm_sample`（第 04 課）

整個 DDPM 的核心程式碼不到 100 行，這也是它迷人的地方。

## 4. 常見誤解

- **「擴散模型是在把一張特定的圖還原」**：不是。生成時沒有「原圖」，網路每一步猜的是「在這種雜訊程度下，**可能的**乾淨圖平均起來長怎樣」。最後得到的是一張新圖。
- **「加噪過程也要訓練」**：DDPM 的加噪過程是固定公式，沒有參數。只有去噪網路要學。
- **「擴散模型只能做影像」**：同樣的數學用在音訊、影片、3D、分子結構、機器人動作，甚至文字（第 11 課）。
- **「擴散模型和 VAE、flow 是完全不同的東西」**：它們在數學上緊密相連——擴散是特殊的階層式 VAE，也可以寫成連續時間的 normalizing flow。

## 5. 練習

**想一想**

1. 一張 64×64 的 RGB 圖有幾個數字？如果每個數字只能是 0 或 255，總共有幾種可能的圖？這告訴你「隨機亂猜」有多沒希望？
2. GAN 的 mode collapse 是什麼意思？為什麼「每張訓練資料都拿來算 MSE」的擴散模型比較不會發生？
3. 如果加噪只加 10 步而不是 1000 步，每一步要修的量會變大。你覺得這會讓網路的工作變難還是變簡單？為什麼？（第 04 課會用「反向過程近似高斯」來回答。）

**動手改**

4. 還沒有 notebook。先確認環境：`cd diffusion-models-course && uv run pytest`，應該全部通過。

## 6. 延伸閱讀

- Lilian Weng，〈What are Diffusion Models?〉（2021-07，之後持續更新）：<https://lilianweng.github.io/posts/2021-07-11-diffusion-models/> ——最常被引用的入門長文，數學寫得很完整。
- 李宏毅，〈Diffusion Model 原理剖析〉（2023，中文影片）：<https://www.youtube.com/watch?v=ifCDXFdeaaM> ——課程頁：<https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php>
- Sohl-Dickstein 等人（2015）：<https://arxiv.org/abs/1503.03585>
- 從 U-Net 到 DiT 的架構演進（ICLR 2026 Blogposts）：<https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/>
