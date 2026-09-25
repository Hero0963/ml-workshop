# 第 11 課：2026 前沿地圖——接下來讀什麼

> 前置：第 01–10 課 ｜ 本課沒有實驗 ｜ **查證日期：2026-09-25**

## 怎麼讀這一課

前十課是穩定的基礎，十年內大概不會過時。**這一課會過時**：它是一張截至 2026-09-25 的地圖，幫你知道有哪些方向、該從哪篇讀起。

- **一手**：論文（arXiv 摘要頁已逐篇核對標題與首次提交日期）或官方頁面。
- **二手**：整理文章、新聞、部落格。二手資訊會特別標注，數字盡量只引用論文摘要裡寫的。
- 這一課只寫「每個方向在做什麼、代表作是哪篇」，不替任何方法下「誰最好」的結論——不同論文的比較設定往往不一致（例如 Setting-Matched Benchmarking（2026-03）就是專門檢討一步模型與多步模型比較方式的論文：<https://arxiv.org/abs/2603.14186>）。

---

## 1. 少步與一步生成

第 07、09 課的結論是：ODE 求解器降到個位數步時，誤差會變大，要改變**訓練**才能再往下。這是近三年最熱的方向。

| 方法 | 時間 | 核心想法 | 論文 |
|---|---|---|---|
| Progressive Distillation | 2022-02 | 學生模型 1 步模仿老師 2 步，反覆減半 | <https://arxiv.org/abs/2202.00512> |
| Consistency Models | 2023-03 | 學一個函數，把 PF-ODE 軌跡上**任何一點**直接映到終點；可由既有模型蒸餾，也可從頭訓練 | <https://arxiv.org/abs/2303.01469> |
| InstaFlow | 2023-09 | rectified flow 的 reflow（第 09 課）＋蒸餾，文生圖一步生成 | <https://arxiv.org/abs/2309.06380> |
| Latent Consistency Models | 2023-10 | consistency 蒸餾搬到潛空間，2–4 步文生圖 | <https://arxiv.org/abs/2310.04378> |
| Adversarial Diffusion Distillation | 2023-11 | 蒸餾損失 ＋ 對抗損失（SDXL Turbo） | <https://arxiv.org/abs/2311.17042> |
| Distribution Matching Distillation | 2023-11 | 讓一步生成器的**分布**與老師的分布相符（用兩個 score 的差當梯度） | <https://arxiv.org/abs/2311.18828> |
| sCM | 2024-10 | 連續時間 consistency model 的簡化、穩定化與規模化 | <https://arxiv.org/abs/2410.11081> |
| Shortcut Models | 2024-10 | 網路額外輸入「步長」，同一個模型支援 1 步到多步 | <https://arxiv.org/abs/2410.12557> |
| MeanFlow | 2025-05 | 學某段時間區間內的**平均速度**而非瞬時速度；摘要報告 ImageNet 256×256、從頭訓練、1 步 FID 3.43 | <https://arxiv.org/abs/2505.13447> |
| Drifting Models | 2026-02 | 把迭代過程搬進**訓練**：學一個 drift 場推動生成分布，平衡時分布相符；天生一步推論。摘要報告 ImageNet 256×256 一步 FID 1.54（潛空間）／1.61（像素空間） | <https://arxiv.org/abs/2602.04770> |

讀的順序建議：Progressive Distillation → Consistency Models → sCM → MeanFlow → Drifting。Drifting 發表後已有多篇理論分析（例如把它解讀為 Wasserstein 梯度流或 score matching 的論文），代表這個方向在 2026 年仍很活躍。

## 2. 訓練更快：借用預訓練的表徵

| 方法 | 時間 | 核心想法 | 論文 |
|---|---|---|---|
| REPA | 2024-10 | 訓練 DiT 時，額外要求它的中間層特徵對齊預訓練視覺編碼器（如 DINOv2）的特徵，大幅加速收斂 | <https://arxiv.org/abs/2410.06940> |
| RAE | 2025-10 | 用預訓練的表徵編碼器（而非 VAE）當潛空間，只訓練解碼器，再在這個語意豐富的潛空間上訓練 DiT | <https://arxiv.org/abs/2510.11690> |

背後的想法：擴散模型從頭學「語意」很慢，而自監督視覺模型已經學好了，拿來當捷徑。

## 3. 回到像素：讓去噪模型真的去噪

**JiT**（Li & He，2025-11；據搜尋結果收錄於 CVPR 2026）：<https://arxiv.org/abs/2511.13720>

- 主張：今天的擴散模型大多預測 $\epsilon$ 或 velocity，不是直接預測乾淨圖。依流形假設，乾淨影像落在低維流形上，而雜訊與 velocity 是「滿維度」的；所以**直接預測乾淨影像**（$x$-prediction）讓容量看似不足的網路也能在高維空間運作。
- 做法：不用 tokenizer、不用潛空間、不用 U-Net，直接把像素 patch 丟進純 ViT（Just image Transformers）。
- 和本課程的連結：第 04 課 §2.9 的「三種等價參數化」——理論上等價，實務上（高維、有限容量時）差很多。

## 4. 新的生成框架

- **Transition Matching**（Meta，2025-06）：<https://arxiv.org/abs/2506.23589>——把擴散、flow matching 與自迴歸放進同一個「學習轉移核」的框架。
- **Drifting Models**（見 §1）：跳脫「推論時迭代」的框架。

## 5. 文生圖的產業配方

見第 10 課 §2.4。2024 年以後的主流開放模型大致收斂到：**潛空間（或表徵空間）＋ MMDiT 類 Transformer ＋ rectified flow／flow matching ＋ 大型語言或多模態模型當文字編碼器 ＋ CFG ＋ 少步蒸餾版本**。

## 6. 影片、科學與機器人

- **影片**：把 DiT 的 token 從 2D patch 延伸到時空 patch。開放模型的代表如 Wan（2025-03）：<https://arxiv.org/abs/2503.20314>。商用模型（OpenAI Sora、Google Veo 系列等）多半只有技術報告或產品頁，細節較少（二手資訊多）。
- **蛋白質結構**：AlphaFold 3（Abramson 等人，*Nature* 2024）用擴散模組生成原子座標。
- **機器人**：Diffusion Policy（2023-03）把機器人的動作序列當成要生成的「資料」：<https://arxiv.org/abs/2303.04137>

## 7. 文字也能用擴散生成

文字是離散的 token，不能直接加高斯雜訊。主流做法是**遮罩式（absorbing）離散擴散**：

- **前向**：逐步把 token 換成 `[MASK]`，到最後全部被遮住（對應「純雜訊」）。
- **反向**：網路看著部分遮住的句子，**同時**預測所有被遮住的 token，再依規則決定這一步揭開哪些。
- 和 BERT 的遮罩語言模型很像，差別在於遮罩比例涵蓋 0 到 100%，並有一套完整的生成程序。

| 工作 | 時間 | 重點 | 連結 |
|---|---|---|---|
| D3PM | 2021-07 | 離散狀態空間的擴散框架 | <https://arxiv.org/abs/2107.03006> |
| SEDD | 2023-10 | 用「機率比」取代 score，離散版的 score matching | <https://arxiv.org/abs/2310.16834> |
| MDLM | 2024-06 | 簡化的遮罩擴散語言模型 | <https://arxiv.org/abs/2406.07524> |
| RADD | 2024-06 | 證明遮罩擴散其實在學乾淨資料的條件分布 | <https://arxiv.org/abs/2406.03736> |
| LLaDA | 2025-02 | 從頭訓練 8B 規模的擴散式語言模型 | <https://arxiv.org/abs/2502.09992> |
| Mercury（Inception Labs） | 2025-06 | 商用擴散式語言模型的技術報告，強調生成速度 | <https://arxiv.org/abs/2506.17298> |
| Gemini Diffusion（Google DeepMind） | 2025 | 實驗性的文字擴散模型（官方頁面） | <https://deepmind.google/models/gemini-diffusion/> |

**現況（二手，請自行查證最新）**：擴散式語言模型的主要優勢是**平行解碼帶來的速度**；在推理深度等能力上，與同規模自迴歸模型的比較仍在研究中。2026 年的新聞報導了更新版的商用模型，但這裡沒有一手資料，不列入。

## 8. 怎麼評估

- **FID** 仍是影像生成最常見的指標，但它依賴 ImageNet 的 Inception 特徵，對現代模型有已知的偏差。Stein 等人（2023）建議改用 DINOv2 特徵：<https://arxiv.org/abs/2306.04675>
- 文生圖另外量「文字與影像是否對齊」與人類偏好，常見做法是 CLIP 分數、組合式的物件檢查基準、人類評比。
- 本課程用的「小分類器當評審」（第 06 課）是同一個精神的迷你版：用一個預訓練模型的判斷，代替昂貴的人工評估。

---

## 9. 接下來讀什麼

依你的興趣選一條路：

**把數學打穩**

1. Nakkiran 等人，〈Step-by-Step Diffusion: An Elementary Tutorial〉（2024）：<https://arxiv.org/abs/2406.08929>
2. MIT 6.S184 講義（2025）：<https://arxiv.org/abs/2506.02070>，課程網站 <https://diffusion.csail.mit.edu/>
3. Lipman 等人，〈Flow Matching Guide and Code〉（2024）：<https://arxiv.org/abs/2412.06264>
4. Lai、Song 等人，〈The Principles of Diffusion Models〉（2025-10，專書篇幅）：<https://arxiv.org/abs/2510.21890>

**做影像應用**

1. Hugging Face Diffusion Models Course：<https://huggingface.co/learn/diffusion-course>
2. Stable Diffusion 3 論文：<https://arxiv.org/abs/2403.03206>
3. `diffusers` 文件：<https://huggingface.co/docs/diffusers/index>

**做少步生成**：§1 的表格由上往下讀。

**做文字擴散**：MDLM → LLaDA → Mercury 技術報告。

**持續追蹤**：Sander Dieleman 的部落格（<https://sander.ai/>）長期寫擴散模型的深度文章，例如〈Diffusion is spectral autoregression〉（2024）：<https://sander.ai/2024/09/02/spectral-autoregression.html>

## 10. 更新這一課的方法

照 repo 的「技術 survey 保鮮紀律」（`AGENTS.md` §4）：

- 列表型文章只拿來知道「有哪些方向」，**版本與數字一律回到論文摘要或官方頁面查證**。
- 每個方向做一次反向探測：搜「<方法名> v2」「<方法名> 2027」，看有沒有新版。
- 更新時改掉本課開頭的查證日期，並在 `ai-collab/dev_log.md` 記一筆。
