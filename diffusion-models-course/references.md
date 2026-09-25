# 參考資料與延伸閱讀

> **查證日期：2026-09-25。** arXiv 論文都已逐篇打開摘要頁，核對標題與首次提交日期；網頁資源確認可以連線。
> 標記：**一手**＝論文、官方文件或官方課程；**二手**＝整理文章或部落格。
> 本清單只列連結，不轉載內容。程式庫的授權各自不同，使用前請看該 repo 的 `LICENSE`。

## 1. 先讀這些（入門）

| 資源 | 類型 | 為什麼推薦 |
|---|---|---|
| Lilian Weng，〈What are Diffusion Models?〉<https://lilianweng.github.io/posts/2021-07-11-diffusion-models/> | 二手 | 最常被引用的數學整理，DDPM／score／guidance 都有 |
| Yang Song，〈Generative Modeling by Estimating Gradients of the Data Distribution〉<https://yang-song.net/blog/2021/score/> | 一手（作者本人的部落格） | score 觀點與 SDE 的最佳入門 |
| 李宏毅，〈Diffusion Model 原理剖析〉（2023，中文影片）<https://www.youtube.com/watch?v=ifCDXFdeaaM>；課程頁 <https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php> | 一手（課程） | 中文講解，適合先建立直覺 |
| Nakkiran 等人，〈Step-by-Step Diffusion: An Elementary Tutorial〉（2024）<https://arxiv.org/abs/2406.08929> | 一手 | 用最少的機率知識推出 DDPM、DDIM、flow matching |
| Calvin Luo，〈Understanding Diffusion Models: A Unified Perspective〉（2022）<https://arxiv.org/abs/2208.11970> | 一手 | ELBO 推導最詳細，一行一行展開 |

## 2. 長篇教材與課程

| 資源 | 類型 |
|---|---|
| MIT 6.S184〈Introduction to Flow Matching and Diffusion Models〉：講義 <https://arxiv.org/abs/2506.02070>（2025-06），課程網站 <https://diffusion.csail.mit.edu/> | 一手 |
| Lipman 等人，〈Flow Matching Guide and Code〉（2024-12）<https://arxiv.org/abs/2412.06264> | 一手 |
| Lai、Song、Kim、Mitsufuji、Ermon，〈The Principles of Diffusion Models〉（2025-10，專書篇幅）<https://arxiv.org/abs/2510.21890> | 一手 |
| CVPR 2022 Tutorial〈Denoising Diffusion-based Generative Modeling〉<https://cvpr2022-tutorial-diffusion-models.github.io/> | 一手 |
| Hugging Face Diffusion Models Course <https://huggingface.co/learn/diffusion-course>；教材 repo <https://github.com/huggingface/diffusion-models-class> | 一手 |
| Hugging Face〈The Annotated Diffusion Model〉<https://huggingface.co/blog/annotated-diffusion> | 二手 |
| labml.ai 的 DDPM 逐行註解實作 <https://nn.labml.ai/diffusion/ddpm/index.html> | 二手 |
| Gao 等人，〈Diffusion Meets Flow Matching: Two Sides of the Same Coin〉<https://diffusionflow.github.io/> | 一手（作者群的部落格） |
| Sander Dieleman 的部落格：〈Guidance: a cheat code〉<https://sander.ai/2022/05/26/guidance.html>、〈Perspectives on diffusion〉<https://sander.ai/2023/07/20/perspectives.html>、〈Diffusion is spectral autoregression〉<https://sander.ai/2024/09/02/spectral-autoregression.html> | 二手（作者是業界研究者） |
| ICLR 2026 Blogposts，〈From U-Nets to DiTs〉<https://iclr-blogposts.github.io/2026/blog/2026/diffusion-architecture-evolution/> | 二手 |

## 3. 一手論文（依課程順序）

日期為 arXiv 首次提交日。

### 基礎（第 01–05 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Sohl-Dickstein 等人，Deep Unsupervised Learning using Nonequilibrium Thermodynamics <https://arxiv.org/abs/1503.03585> | 2015-03 | 01 |
| Song & Ermon，Generative Modeling by Estimating Gradients of the Data Distribution <https://arxiv.org/abs/1907.05600> | 2019-07 | 02、05 |
| Song & Ermon，Improved Techniques for Training Score-Based Generative Models <https://arxiv.org/abs/2006.09011> | 2020-06 | 02 |
| Ho、Jain、Abbeel，Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2006.11239> | 2020-06 | 03、04 |
| Song 等人，Score-Based Generative Modeling through Stochastic Differential Equations <https://arxiv.org/abs/2011.13456> | 2020-11 | 05 |
| Nichol & Dhariwal，Improved Denoising Diffusion Probabilistic Models <https://arxiv.org/abs/2102.09672> | 2021-02 | 03、04 |
| Kingma 等人，Variational Diffusion Models <https://arxiv.org/abs/2107.00630> | 2021-07 | 03 |
| Kingma & Gao，Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation <https://arxiv.org/abs/2303.00848> | 2023-03 | 09 |
| Lin 等人，Common Diffusion Noise Schedules and Sample Steps are Flawed <https://arxiv.org/abs/2305.08891> | 2023-05 | 03 |
| Hoogeboom 等人，simple diffusion <https://arxiv.org/abs/2301.11093> | 2023-01 | 03 |

非 arXiv 的經典：Hyvärinen（2005，*JMLR* 6，score matching）；Vincent（2011，*Neural Computation* 23(7)，denoising score matching）；Efron（2011，*JASA* 106，Tweedie 公式）；Anderson（1982，*Stochastic Processes and their Applications* 12，反向時間 SDE）。

### 影像、取樣與條件（第 06–08 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Ronneberger 等人，U-Net <https://arxiv.org/abs/1505.04597> | 2015-05 | 06 |
| Heusel 等人，FID（Two Time-Scale Update Rule） <https://arxiv.org/abs/1706.08500> | 2017-06 | 06 |
| Karras 等人，Analyzing and Improving the Training Dynamics of Diffusion Models（EDM2） <https://arxiv.org/abs/2312.02696> | 2023-12 | 06 |
| Carlini 等人，Extracting Training Data from Diffusion Models <https://arxiv.org/abs/2301.13188> | 2023-01 | 06 |
| Somepalli 等人，Diffusion Art or Digital Forgery? <https://arxiv.org/abs/2212.03860> | 2022-12 | 06 |
| Song、Meng、Ermon，Denoising Diffusion Implicit Models <https://arxiv.org/abs/2010.02502> | 2020-10 | 07 |
| Karras 等人，Elucidating the Design Space of Diffusion-Based Generative Models（EDM） <https://arxiv.org/abs/2206.00364> | 2022-06 | 05、07 |
| Lu 等人，DPM-Solver <https://arxiv.org/abs/2206.00927>；DPM-Solver++ <https://arxiv.org/abs/2211.01095> | 2022-06／11 | 07 |
| Zhao 等人，UniPC <https://arxiv.org/abs/2302.04867> | 2023-02 | 07 |
| Dhariwal & Nichol，Diffusion Models Beat GANs on Image Synthesis <https://arxiv.org/abs/2105.05233> | 2021-05 | 06、08 |
| Ho & Salimans，Classifier-Free Diffusion Guidance <https://arxiv.org/abs/2207.12598> | 2022-07 | 08 |
| Nichol 等人，GLIDE <https://arxiv.org/abs/2112.10741> | 2021-12 | 08 |
| Kynkäänniemi 等人，Applying Guidance in a Limited Interval <https://arxiv.org/abs/2404.07724> | 2024-04 | 08 |
| Karras 等人，Guiding a Diffusion Model with a Bad Version of Itself <https://arxiv.org/abs/2406.02507> | 2024-06 | 08 |

### Flow matching 與大模型（第 09–10 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Chen 等人，Neural Ordinary Differential Equations <https://arxiv.org/abs/1806.07366> | 2018-06 | 09 |
| Liu、Gong、Liu，Flow Straight and Fast（Rectified Flow） <https://arxiv.org/abs/2209.03003> | 2022-09 | 09 |
| Albergo & Vanden-Eijnden，Stochastic Interpolants <https://arxiv.org/abs/2209.15571> | 2022-09 | 09 |
| Lipman 等人，Flow Matching for Generative Modeling <https://arxiv.org/abs/2210.02747> | 2022-10 | 09 |
| Ma 等人，SiT <https://arxiv.org/abs/2401.08740> | 2024-01 | 09 |
| Rombach 等人，Latent Diffusion Models <https://arxiv.org/abs/2112.10752> | 2021-12 | 10 |
| Saharia 等人，Imagen <https://arxiv.org/abs/2205.11487> | 2022-05 | 06、10 |
| Peebles & Xie，Scalable Diffusion Models with Transformers（DiT） <https://arxiv.org/abs/2212.09748> | 2022-12 | 10 |
| Esser 等人，Scaling Rectified Flow Transformers（Stable Diffusion 3） <https://arxiv.org/abs/2403.03206> | 2024-03 | 09、10 |
| Qwen-Image Technical Report <https://arxiv.org/abs/2508.02324> | 2025-08 | 10 |

前沿（第 11 課）的論文清單在 [`lessons/11_frontier_2026.md`](lessons/11_frontier_2026.md)，那裡的每一筆也都依同樣方式查證過。

## 4. 程式庫與工具

| 資源 | 說明 |
|---|---|
| Hugging Face `diffusers` <https://huggingface.co/docs/diffusers/index> | 最常用的擴散模型程式庫；2026-09-25 PyPI 最新版為 0.40.0 |
| Meta `flow_matching` <https://github.com/facebookresearch/flow_matching> | 〈Flow Matching Guide and Code〉的官方實作；**CC BY-NC 授權（非商用）** |
| JiT 官方實作 <https://github.com/LTH14/JiT> | 第 11 課 §3 |

本課程的程式碼**沒有**複製上述任何程式庫，全部依論文公式重新撰寫（見 [`NOTICE.md`](NOTICE.md)）。
