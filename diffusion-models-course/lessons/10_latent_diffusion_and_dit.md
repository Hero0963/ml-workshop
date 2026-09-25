# 第 10 課：走向大模型——Latent Diffusion、DiT 與文字條件

> 前置：第 08、09 課 ｜ 實驗：[`notebooks/10_tiny_dit.ipynb`](../notebooks/10_tiny_dit.ipynb) ｜ 預估時間：2 小時（實驗訓練在 CPU 上約 15 分鐘）

## 這一課要回答的問題

- 512×512 的彩色圖有將近 80 萬個數字，直接在上面做擴散太貴，怎麼辦？
- 為什麼 2023 年之後的文生圖模型幾乎都把 U-Net 換成了 Transformer？
- 文字是怎麼「告訴」去噪網路要畫什麼的？
- Stable Diffusion 3、FLUX 這類模型，拆開來是哪些零件？用這些模型時，授權要注意什麼？

---

## 1. 白話版

### 1.1 先壓縮，再擴散

畫一幅畫，構圖（大象在左、太陽在右）和筆觸（毛髮的紋理）是兩回事。
擴散模型最花力氣的地方在構圖，但像素空間裡大部分的數字都在描述細微紋理。

**Latent diffusion** 的做法：先訓練一個自編碼器，把圖片壓成一個小很多的「潛表示」（例如 512×512×3 → 64×64×4，少了 48 倍），細節交給解碼器負責；擴散模型只在這個小空間裡做構圖。最後解碼回圖片。

### 1.2 Transformer 取代 U-Net

把潛表示切成一塊一塊（patch），每塊當成一個 token，丟進和 GPT 一樣的 Transformer 積木。
好處是：Transformer 的規模化規律大家很熟——**模型越大、算力越多，效果穩定地越好**，而且可以直接沿用語言模型的工程經驗與硬體最佳化。

### 1.3 文字怎麼進來

先用一個文字模型（CLIP、T5、甚至一個 LLM）把提示詞變成一串向量。去噪網路在處理影像 token 時，透過 **attention** 去「查閱」這些文字向量：這個位置該畫什麼，就去文字裡找相關的詞。

---

## 2. 正式版

### 2.1 Latent Diffusion（Rombach 等人 2021）

兩階段：

1. **感知壓縮**：訓練自編碼器 $\mathcal{E}$、$\mathcal{D}$，使 $\mathcal{D}(\mathcal{E}(x)) \approx x$。損失包含重建誤差、感知損失（LPIPS）、對抗損失，以及輕微的 KL 正則化（讓潛空間接近常態、尺度可控）。下採樣倍率 $f = 8$、潛通道 $c = 4$ 是 Stable Diffusion 1.x／2.x 的設定。
2. **潛空間擴散**：對 $z = \mathcal{E}(x)$ 訓練擴散模型，演算法和前面各課完全相同。生成時 $x = \mathcal{D}(z)$。

實務細節：潛表示要乘上一個尺度因子讓變異數接近 1（SD 1.x 用 0.18215），理由和第 03 課「資料要正規化到單位尺度」相同。

為什麼有效：Rombach 等人的分析是，自編碼器負責「感知上不重要的高頻細節」，擴散模型只需要學「語意與構圖」；在壓縮倍率適中時，品質幾乎不損失，計算量大幅下降。

### 2.2 Diffusion Transformer（DiT，Peebles & Xie 2022）

1. **Patchify**：把 $H \times W \times C$ 的輸入切成 $p \times p$ 的小塊，每塊線性投影成一個 $D$ 維 token，加上位置嵌入。$N = (H/p)(W/p)$ 個 token。
2. **Transformer blocks**：標準的 self-attention ＋ MLP。
3. **條件注入**：Peebles & Xie 比較了四種方式——in-context（把條件當成額外 token）、cross-attention、adaLN、**adaLN-Zero**——最後一種最好：

$$\mathrm{adaLN}(h, c) = \mathrm{LayerNorm}(h)\,(1 + \gamma(c)) + \beta(c), \qquad h \leftarrow h + \alpha(c)\odot\mathrm{Block}(\mathrm{adaLN}(h, c))$$

其中 $\gamma$、$\beta$、$\alpha$ 都由條件向量 $c$（時間＋類別嵌入）經線性層算出，且**初始化為 0**，使每個 block 一開始是恆等映射（和第 06 課 U-Net 輸出層初始化為 0 同一個想法）。

4. **Unpatchify**：最後一層把每個 token 投影回 $p \times p \times C$，拼回影像。

DiT 最重要的發現是**規模化**：模型的計算量（Gflops）與 FID 高度相關，加大模型或縮小 patch（更多 token）都穩定改善品質。

### 2.3 文字條件

**文字編碼器**：CLIP 的文字編碼器（對齊了影像與文字的語意）、T5 或 LLM（語言理解更強，能處理長而複雜的提示詞）。Imagen（Saharia 等人 2022）發現擴大文字編碼器比擴大擴散模型更能提升文字與影像的對齊。

**注入方式**：

- **Cross-attention**（Stable Diffusion 1.x／2.x、SDXL）：影像特徵當 query，文字 token 當 key／value，$\mathrm{softmax}\left(QK^\top/\sqrt{d}\right)V$。
- **Joint attention／MMDiT**（Stable Diffusion 3）：文字 token 與影像 token **串在一起**做 self-attention，但兩種模態各自有一套權重（QKV 投影、MLP）。資訊可以雙向流動。
- 池化後的文字向量（例如 CLIP 的整句嵌入）通常再經 adaLN 注入，和時間嵌入相加。

搭配第 08 課的 classifier-free guidance：訓練時隨機把文字換成空字串，取樣時用 (8.3)。

### 2.4 架構演進（2021–2025）

以下整理自 ICLR 2026 Blogposts〈From U-Nets to DiTs〉（二手整理，2026-09-25 查閱）與各模型的技術報告：

| 年份 | 模型 | 骨幹 | 文字編碼器 |
|---|---|---|---|
| 2022 | Stable Diffusion 1.x／2.x | U-Net（潛空間） | CLIP／OpenCLIP |
| 2023 | SDXL | 較大的 U-Net（約 2.6B） | 兩個 CLIP |
| 2023 | PixArt-α | DiT | T5 |
| 2024 | Stable Diffusion 3 | MMDiT、rectified flow | 兩個 CLIP ＋ T5 |
| 2024 | FLUX.1 | 混合 MMDiT（約 12B） | CLIP ＋ T5 |
| 2025 | SANA | 線性注意力的 DiT（約 0.6B） | 小型 LLM |
| 2025 | Qwen-Image | MMDiT（約 20B） | Qwen2.5-VL（多模態 LLM） |

趨勢：**U-Net → Transformer；CLIP → 大型語言／多模態模型；DDPM 式損失 → rectified flow**。

### 2.5 使用現成模型

實務上會用 Hugging Face `diffusers`（2026-09-25 查證的最新版為 0.40.0）。以下只是示意，**未在本課程環境執行**（需要 GPU 與數 GB 的權重下載）：

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("<model-id>", torch_dtype=torch.bfloat16).to("cuda")
image = pipe("a watercolor lighthouse at dawn", num_inference_steps=28, guidance_scale=4.0).images[0]
```

`num_inference_steps` 就是第 07、09 課的取樣步數，`guidance_scale` 就是第 08 課的 $s$。讀完這門課，pipeline 的每個參數都應該說得出它在做什麼。

### 2.6 授權：開放權重 ≠ 開源授權

下載權重之前，一定要讀模型卡上的授權。以下是 2026-09-25 從 Hugging Face 模型卡（`license` 欄位）讀到的例子：

| 模型 | 授權 | 意涵 |
|---|---|---|
| `black-forest-labs/FLUX.1-dev` | FLUX.1 [dev] Non-Commercial License | 非商用 |
| `stabilityai/stable-diffusion-3.5-large` | Stability AI Community License | 有營收門檻等條件，詳見授權全文 |
| `Qwen/Qwen-Image` | Apache-2.0 | 寬鬆授權，可商用（仍須保留授權聲明） |

授權會改版，以使用當下的模型卡為準。另外，**生成內容的著作權**與**訓練資料的授權**是兩個獨立的問題（第 06 課 §2.6 的記憶化問題就是例子），各國法律仍在發展中。

---

## 3. 對照程式碼

| 概念 | 位置（`src/diffusion_course/models/dit.py`） |
|---|---|
| Patchify（一個 stride = patch 的卷積） | `TinyDiT.patchify` |
| 位置嵌入 | `TinyDiT.pos_embed`（可學習；原始 DiT 用固定的 2D sin-cos） |
| adaLN-Zero | `DiTBlock.ada_ln`（輸出 6 組 shift／scale／gate，初始化為 0）、`modulate` |
| Unpatchify | `TinyDiT.unpatchify`（`tests/test_models.py` 驗證它是 patchify 排列的逆運算） |

`TinyDiT` 預設設定：28×28 輸入、patch 4 → 49 個 token、寬度 128、4 層、4 個 head，約 126 萬參數。實驗用 flow matching 損失 ＋ CFG 訓練它——也就是一個迷你版的「rectified flow Transformer」，和 SD3 同一套配方，只是沒有潛空間和文字編碼器。

## 4. 常見誤解

- **「latent diffusion 是另一種擴散演算法」**：演算法完全一樣，只是資料換成了自編碼器的潛表示。
- **「Transformer 在小資料上一定比 U-Net 好」**：U-Net 的卷積有「局部性」這個有用的先驗，在小模型、小資料時常常更有效率；Transformer 的優勢在規模化。
- **「模型可以下載就可以隨便用」**：見 §2.6。
- **「文字編碼器也要跟著訓練」**：通常凍結預訓練好的文字編碼器，只訓練去噪網路。

## 5. 練習

**想一想**

1. 512×512×3 的圖，用 $f = 8$、$c = 4$ 的自編碼器壓縮後有幾個數字？若在潛空間用 patch size 2 的 DiT，有幾個 token？自注意力的計算量和 token 數是什麼關係？
2. adaLN-Zero 的 gate $\alpha$ 初始化為 0，對訓練初期的梯度流有什麼影響？
3. Cross-attention 與 MMDiT 的 joint attention，在「文字能否被影像影響」上有什麼差別？

**動手改**（在 `10_tiny_dit.ipynb`）

4. 把 `PATCH_SIZE` 改成 2（196 個 token）或 7（16 個 token），比較品質與訓練速度。
5. 把位置嵌入改成固定的 2D sin-cos，和可學習的版本比較。
6. （進階）做一個迷你 latent diffusion：先訓練一個把 28×28 壓成 7×7×4 的小自編碼器，再在潛空間上訓練 flow matching。

## 6. 延伸閱讀

- Rombach 等人（2021），Latent Diffusion Models：<https://arxiv.org/abs/2112.10752>
- Peebles & Xie（2022），DiT：<https://arxiv.org/abs/2212.09748>
- Esser 等人（2024），Stable Diffusion 3（MMDiT、rectified flow 的大規模比較）：<https://arxiv.org/abs/2403.03206>
- Saharia 等人（2022），Imagen：<https://arxiv.org/abs/2205.11487>
- Qwen-Image 技術報告（2025）：<https://arxiv.org/abs/2508.02324>
- ICLR 2026 Blogposts，〈From U-Nets to DiTs〉：<https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/>
- Hugging Face Diffusion Models Course：<https://huggingface.co/learn/diffusion-course>——教材在 <https://github.com/huggingface/diffusion-models-class>，以 `diffusers` 為主，適合接著本課程學實務工具。
- `diffusers` 文件：<https://huggingface.co/docs/diffusers/index>
