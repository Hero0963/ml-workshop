# 第 06 課：影像擴散——U-Net 與 MNIST

> 前置：第 04 課 ｜ 實驗：[`notebooks/06_mnist_ddpm.ipynb`](../notebooks/06_mnist_ddpm.ipynb) ｜ 預估時間：2 小時（實驗訓練在 CPU 上約 15–20 分鐘）

## 這一課要回答的問題

- 從 2D 點換到影像，演算法完全不變，那什麼要變？——**網路架構**。
- 擴散模型的 U-Net 由哪些零件組成？每個零件為什麼在那裡？
- 訓練影像擴散模型有哪些實務細節（EMA、截斷 $\hat x_0$、評估、記憶化）？

---

## 1. 白話版

### 1.1 去雜訊需要「看局部」也需要「看全局」

要判斷一個像素是雜訊還是筆畫，你需要看它**周圍**（局部：這裡是不是一條線的延伸？）；
要判斷整張圖大概是什麼數字，你需要看**整體**（全局：這是 3 還是 8？）。

**U-Net** 同時做到這兩件事：

1. **往下走（編碼）**：一路把圖縮小（28 → 14 → 7），每個位置看到的範圍越來越大，理解整體；
2. **往上走（解碼）**：一路放大回原尺寸，把細節補回來；
3. **跳接（skip connection）**：縮小時會丟失細節，所以把每個尺寸的特徵「直接抄一份」送到解碼那邊同尺寸的地方。

形狀像字母 U，因此得名。

### 1.2 告訴網路「現在的雜訊有多重」

同一張 $x_t$，雜訊多和雜訊少時該做的事完全不同。網路要知道 $t$。
做法和 Transformer 的位置編碼一樣：把 $t$ 變成一串不同頻率的 sin、cos 值，再經過一個小 MLP，加進 U-Net 的每一層。

### 1.3 EMA：看「最近一段時間的平均」

訓練時權重會隨每個 batch 抖動。取樣時改用「權重的移動平均」（Exponential Moving Average），圖會乾淨很多。這幾乎是擴散模型的標準配備。

---

## 2. 正式版

### 2.1 為什麼是 U-Net

$\epsilon_\theta(x_t, t)$ 的輸出和輸入同尺寸，是逐像素的「稠密預測」問題，和影像分割一樣。Ho 等人（2020）採用的 U-Net 源自醫學影像分割（Ronneberger 等人 2015），並加入 GroupNorm、時間嵌入與自注意力。之後的影像擴散模型在 2022 年 DiT 出現前幾乎都用它的變體（第 10 課）。

### 2.2 本課程的 U-Net（`src/diffusion_course/models/unet.py`）

以實驗用的設定（`base_channels=16`、`channel_mults=(1, 2, 2)`）為例：

| 階段 | 解析度 | 通道 | 內容 |
|---|---|---|---|
| stem | 28×28 | 1 → 16 | 3×3 卷積 |
| encoder level 0 | 28×28 | 16 | 2 個 ResBlock → 存 skip → 下採樣（stride-2 卷積） |
| encoder level 1 | 14×14 | 16 → 32 | 2 個 ResBlock → 存 skip → 下採樣 |
| encoder level 2 | 7×7 | 32 | ResBlock → **自注意力** → ResBlock → 存 skip |
| middle | 7×7 | 32 | ResBlock → 自注意力 → ResBlock |
| decoder level 2 | 7×7 | 32 + 32 → 32 | 接上 skip → ResBlock、注意力、ResBlock → 上採樣 |
| decoder level 1 | 14×14 | 32 + 32 → 32 | 接上 skip → 2 個 ResBlock → 上採樣 |
| decoder level 0 | 28×28 | 32 + 16 → 16 | 接上 skip → 2 個 ResBlock |
| head | 28×28 | 16 → 1 | GroupNorm → SiLU → 3×3 卷積（**權重初始化為 0**） |

參數約 31 萬。`base_channels=32` 時約 121 萬（有 GPU 時建議）。

### 2.3 零件逐一說明

**ResBlock**：

$$h = \mathrm{Conv}(\mathrm{SiLU}(\mathrm{GN}(x))) + W_c\,c, \qquad \mathrm{out} = x' + \mathrm{Conv}(\mathrm{Dropout}(\mathrm{SiLU}(\mathrm{GN}(h))))$$

$c$ 是條件向量（時間嵌入，第 08 課起再加上類別嵌入），經線性層投影後**加到每個通道**上；$x'$ 是輸入（通道數不同時先過 1×1 卷積）。
其他常見的注入方式：把 $c$ 變成 GroupNorm 的 scale 與 shift（Dhariwal & Nichol 2021 的 AdaGN），或 DiT 的 adaLN（第 10 課）。

**時間嵌入**：

$$\mathrm{emb}(t) = \left[\sin(\omega_1 t'), \dots, \sin(\omega_{d/2} t'),\ \cos(\omega_1 t'), \dots, \cos(\omega_{d/2} t')\right], \qquad \omega_i = 10000^{-(i-1)/(d/2)},\quad t' = 1000\,t$$

低頻的分量描述「大概在哪個階段」，高頻的分量分辨相鄰的步數。再接兩層 MLP（`TimeEmbedding`）。

**GroupNorm 而不是 BatchNorm**：同一個 batch 裡混著各種 $t$，雜訊程度差很多；BatchNorm 會把它們的統計量混在一起。GroupNorm 只在單一樣本內正規化，不受 batch 組成影響。

**自注意力只放在 7×7**：注意力的計算量與像素數的平方成正比，28×28 是 784 個位置，太貴；7×7 只有 49 個。低解析度正好也是需要「全局資訊」的地方。

**輸出層初始化為 0**：一開始網路輸出恆為 0，損失從 $\mathbb{E}\|\epsilon\|^2 = 1$ 開始，是個合理而穩定的起點（「預測雜訊為零」等於「猜 $x_t$ 就是乾淨的」）。DiT 的 adaLN-Zero 是同樣的想法。

### 2.4 訓練實務

- **資料尺度**：像素正規化到 $[-1, 1]$（`image_dataset` 裡的 `Normalize((0.5,), (0.5,))`），與雜訊的單位變異數相配。
- **截斷 $\hat x_0$**：取樣時把 $\hat x_0$ 截在 $[-1, 1]$（`clip_x0=True`），避免早期步驟的離譜預測把整條鏈帶歪。高解析度模型常改用 dynamic thresholding（Imagen，Saharia 等人 2022）。
- **EMA**：$\theta_{\text{EMA}} \leftarrow \lambda\,\theta_{\text{EMA}} + (1-\lambda)\,\theta$。平均的「記憶長度」約為 $1/(1-\lambda)$ 步。$\lambda$ 要配合訓練長度：實驗只訓練 2000 步，用 $\lambda = 0.995$（約 200 步）；論文等級的長訓練常用 0.9999。若 $\lambda$ 太大而訓練太短，EMA 會一直被初始的隨機權重拖住。Karras 等人（2023，EDM2）提出訓練後再決定 EMA 長度的方法。
- **訓練量**：論文等級的模型訓練數十萬步以上；本課程的實驗為了在 CPU 上跑完，只訓練 2000 步，樣本會有瑕疵但數字可辨認。

### 2.5 怎麼評估生成品質

- **FID**（Fréchet Inception Distance，Heusel 等人 2017）：把真實圖與生成圖都丟進 ImageNet 訓練的 Inception 網路，取特徵，各自擬合一個多維常態分布，計算兩個常態之間的 Fréchet 距離。越低越好。通常要 5 萬張樣本，對小數字圖意義有限，也有已知的偏差（Stein 等人 2023）。
- **本課程的替代方案**：訓練一個 1 分鐘就好的數字分類器（`diffusion_course.evaluation`），量兩件事：
  - **confidence**：分類器對樣本的最大機率平均值（看起來像不像**某個**數字）；
  - **class entropy**：十個類別的分布有多平均（1.0 表示十個數字出現頻率相同；太低代表模式崩塌）。

### 2.6 記憶化：模型會不會只是背下訓練資料？

生成模型可能「背」下訓練圖並原樣吐出。Carlini 等人（2023）從 Stable Diffusion 等當時最先進的模型中抽出了上千張幾乎與訓練圖一模一樣的樣本（從個人照片到商標都有）；Somepalli 等人（2022）也發現了明顯的資料複製現象。
這不只是技術問題，也牽涉**著作權與隱私**：若訓練資料有版權，模型吐出近乎原圖的內容就可能構成侵權。
實驗 §6 會做最簡單的檢查：對每張生成的數字，找訓練集中 L2 距離最近的圖並排比較。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| U-Net 與其零件 | `UNet`、`ResBlock`、`SelfAttention2d`、`Level`（`src/diffusion_course/models/unet.py`） |
| 時間嵌入、類別嵌入 | `SinusoidalEmbedding`、`TimeEmbedding`、`LabelEmbedding`（`models/embeddings.py`） |
| EMA 與訓練迴圈 | `EMA`、`train`（`src/diffusion_course/training.py`） |
| MNIST 載入 | `image_dataset`、`image_loader`（`src/diffusion_course/data.py`） |
| 評估用分類器 | `get_digit_classifier`、`judge_samples`（`src/diffusion_course/evaluation.py`） |
| 長時間訓練 | `scripts/train_mnist.py`、`scripts/sample_mnist.py` |

## 4. 常見誤解

- **「影像擴散需要不同的演算法」**：`ddpm_loss` 與 `ddpm_sample` 和 2D 實驗是**同一個函式**，只換了網路和資料。
- **「EMA decay 越接近 1 越好」**：要和訓練長度相配，否則 EMA 權重還停在初始化附近。
- **「loss 還在降，所以樣本一定還在變好」**：loss 變化常常很小，樣本品質要用看的或用指標量。
- **「生成的圖都是新的」**：不一定；小資料集、長時間訓練、大模型都會提高記憶化的風險。

## 5. 練習

**想一想**

1. 若 U-Net 的 `channel_mults` 改成 `(1, 2, 2, 2)`，28×28 的輸入會發生什麼事？（提示：7 不能被 2 整除。）
2. 把時間嵌入從 U-Net 拿掉，網路還能學會嗎？它會學到什麼樣的「平均策略」？
3. EMA decay 0.995 的記憶長度約 200 步，0.9999 約 1 萬步。若只訓練 2000 步卻用 0.9999，EMA 權重大約還保留多少比例的初始權重？（$0.9999^{2000} \approx ?$）

**動手改**（在 `06_mnist_ddpm.ipynb`；有 GPU 的話效果更明顯）

4. 把 `BASE_CHANNELS` 改成 32、`TRAIN_STEPS` 改成 10000，比較樣本與分類器指標。
5. 關掉 `clip_x0`，樣本有什麼變化？
6. 換成 Fashion-MNIST：`image_loader("fashion_mnist")`。注意評估用的分類器是 MNIST 的，需要另外訓練一個。

## 6. 延伸閱讀

- Ronneberger 等人（2015），U-Net：<https://arxiv.org/abs/1505.04597>
- Ho 等人（2020），DDPM 附錄 B（架構與訓練細節）：<https://arxiv.org/abs/2006.11239>
- Dhariwal & Nichol（2021），架構消融（注意力、AdaGN、BigGAN 式 ResBlock）：<https://arxiv.org/abs/2105.05233>
- Karras 等人（2023），EDM2（訓練動態、post-hoc EMA）：<https://arxiv.org/abs/2312.02696>
- Hugging Face〈The Annotated Diffusion Model〉：<https://huggingface.co/blog/annotated-diffusion>——另一份逐行註解的 PyTorch 實作。
- Carlini 等人（2023），〈Extracting Training Data from Diffusion Models〉：<https://arxiv.org/abs/2301.13188>
- Somepalli 等人（2022），〈Diffusion Art or Digital Forgery?〉：<https://arxiv.org/abs/2212.03860>
