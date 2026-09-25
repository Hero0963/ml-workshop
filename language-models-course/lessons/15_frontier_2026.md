# 第 15 課：2026 前沿地圖

> 前置：全部 ｜ 本課沒有實驗 ｜ 預估閱讀時間：1 小時
>
> **查證日期：2026-09-25。** 本課依 repo 的「技術 survey 保鮮紀律」（`AGENTS.md` §4）撰寫：模型版本與規格一律以**一手來源**為準——Hugging Face 上各組織依建立日期排序的模型清單與官方 model card、arXiv 摘要頁。二手文章只用來找線索。**這一課是全課程最快過期的部分**，建議每半年重查一次（見 §7）。

## 這一課要回答的問題

- 2026 年的開放權重模型長什麼樣？和本課程做的小 GPT 差在哪裡？
- 哪些方向正在改變「預訓練 → 後訓練 → 推論」這條流水線？
- 讀一份新的技術報告時，要看哪些地方？

---

## 1. 白話版：本課程的零件，放大一百萬倍之後

回頭看第 04–09 課的每一個零件，2026 年的旗艦模型幾乎都保留了它們，但在四個地方做了大改：

1. **不是每個參數都要用**：模型有上兆個參數，但每個 token 只啟用其中幾十億個（mixture of experts）。
2. **attention 不再全部是「每個字看每個字」**：長到一百萬 token 的 context，靠的是只看一部分（稀疏）或用固定大小的記憶（線性 attention）。
3. **會先想再答**：推理模式可以調整「想多久」，想得越久越準、越貴。
4. **不只文字**：同一個模型吃文字、圖片、影片。

---

## 2. Mixture of Experts：總參數與啟用參數

把 MLP 換成 $E$ 個「專家」MLP，加一個路由器，每個 token 只送進分數最高的 $k$ 個（Switch Transformer、Mixtral）。訓練與推論的 FLOPs 只和**啟用參數**有關（第 07 課的 $6ND$ 裡的 $N$ 要換成啟用參數），但記憶體要放下**全部參數**。

| 模型（開放權重） | 總參數／啟用參數 | 專家 | 一手來源 |
|---|---|---|---|
| gpt-oss-120b（2025-08） | 117B／5.1B | — | model card（Apache-2.0；MoE 權重以 MXFP4 發布，單張 80 GB GPU 可跑） |
| Mistral Small 4（版本 2603） | 119B／6.5B | 128 選 4 | model card（Apache-2.0，256K context） |
| MiniMax-M3（2026-06） | 約 428B／約 23B | — | model card、技術報告 arXiv 2606.13392 |
| DeepSeek-V4-Flash／Pro（2026） | 284B／13B、1.6T／49B | — | 技術報告 arXiv 2606.19348（MIT） |
| Qwen3.8-2.4T-A95B（2026-08） | 2.4T／95B | 512 | model card |
| Kimi K3（2026-06） | 2.8T | 896 選 16 | model card |

路由的難題：負載不均（有些專家太忙、有些閒置）需要輔助 loss 或偏差調整；推論時要用 expert parallelism（第 08 課 §2.7）把專家分散到多張卡。

## 3. 長 context：稀疏與線性 attention

第 08、09 課算過：attention 的計算與 KV cache 隨 context 長度成長，一百萬 token 時完整 attention 既算不動、也存不下。2026 年的旗艦多半標榜 1M context，做法分兩類：

- **稀疏 attention**：每個 query 只看挑出來的一部分 key。MiniMax Sparse Attention 在 GQA 上加一個輕量的「索引分支」替 KV 區塊打分、每組只選 top-k 區塊做精確 attention（摘要：1M context 時每個 token 的 attention 計算量降 28.4 倍）。DeepSeek-V4 結合兩種壓縮式 attention（CSA、HCA），摘要說在 1M context 下每個 token 的推論 FLOPs 只需 V3.2 的 27%、KV cache 只需 10%。
- **線性 attention（固定大小的狀態）**：把「看全部歷史」換成一個會更新的矩陣狀態，像 RNN 一樣每步成本固定。Gated DeltaNet（Yang 等人 2024）與 Kimi 的 KDA（Kimi Linear，2025）屬於這一類。實務上多採**混合**：Qwen3.8-2.4T 的 92 層是 23 組「3 層 Gated DeltaNet ＋ 1 層 gated attention」（model card）；Kimi Linear 的 model card 說 KV cache 最多減少 75%。

更早的路線：狀態空間模型（Mamba、Mamba-2）、滑動視窗（nanochat 的 SSSL 模式、Gemma 2）、attention sinks、RoPE 的外推技巧。

## 4. 最佳化器：Muon 走上大規模

第 06 課介紹的 Muon 已不只是 speedrun 的技巧：

- Kimi K2（2025）提出 MuonClip（Muon 加上 QK-clip 以避免 attention logit 爆大），在 15.5 兆 token 上預訓練「零 loss spike」（摘要）。
- DeepSeek-V4（2026）的摘要把 Muon 列為三項關鍵升級之一（另兩項是混合壓縮 attention 與 manifold-constrained hyper-connections）。
- nanochat 用 Polar Express 係數與 NorMuon 的變體（第 06 課 §2.5）。

另一個在 2026 年模型卡中常見的主題是**改造 residual 連接**：DeepSeek-V4 的 mHC、Kimi K3 的 Attention Residuals——第 04 課「residual stream」這個看似固定的設計，也在被重新設計。

## 5. 推理模式與「想多久」

DeepSeek-R1（2025）示範了用可驗證獎勵的 RL 讓模型寫出長推理（第 12 課）。到 2026 年，**可調的推理強度**已是開放模型的標準介面：DeepSeek-V4、Qwen3.8、GLM-5.3、Mistral Small 4 的 model card 都提供 `reasoning_effort` 之類的參數（low／high／max）。這把第 07 課的「算力預算」延伸到推論時（test-time compute）：同一個模型，多花推論算力可以換到更高的正確率。
第 12 課 §2.6 的提醒仍然適用：RL 提升的是 pass@1，至於是否真的擴展了模型能解的題目範圍，研究仍有爭論。

## 6. 其他值得追蹤的方向

- **原生多模態**：Kimi K3、MiniMax-M3 在同一個模型裡處理文字、圖片、影片；embedding 模型也一樣（Qwen3-VL-Embedding、jina-embeddings-v5-omni，第 13 課 §2.8）。
- **低精度發布**：MXFP4（gpt-oss）、NVFP4、FP8 的權重檔成為常態（第 08 課 §2.4、第 09 課 §2.4）。
- **內建 speculative decoding**：DeepSeek 在 Hugging Face 上發布了 EAGLE-3 草稿模型，DeepSeek-V4 的部署指令裡直接帶 speculative 設定；Mistral 也發布了 EAGLE 版本（第 09 課 §2.3）。
- **多 token 預測**：訓練時一次預測好幾個未來 token（Gloeckle 等人 2024；DeepSeek-V3 採用），可兼作推論加速。
- **Agent 導向的評估**：新模型卡的主要數字已從 MMLU 類的選擇題，轉向在容器裡完成軟體工程、終端機操作等多步任務的 benchmark（第 14 課的評估問題在這裡更難）。
- **不用 tokenizer**：Byte Latent Transformer 等（第 03 課 §2.9）。
- **授權各異**：同樣是「開放權重」，有 MIT（DeepSeek-V4）、Apache-2.0（gpt-oss、Mistral Small 4），也有各家自訂授權（Qwen3.8 旗艦、Kimi K3、GLM-5.3、MiniMax-M3 的 model card 都標示 `license: other`）。使用前務必讀授權全文。

## 7. 怎麼替這一課保鮮

照 `AGENTS.md` §4 的紀律：

1. **列表文章只用來列舉家族**，不用來決定「最新是第幾版」。
2. 對每個家族到一手來源確認：Hugging Face 的組織頁（依建立日期排序）、官方技術報告、arXiv。本課用的查詢形如 `https://huggingface.co/api/models?author=<org>&sort=createdAt&direction=-1&limit=10`。
3. 每個家族做一次**反向探測**：搜尋「<家族名> 下一個版本」，確認沒有更新的版本。
4. **驗證要對稱**：每個家族用同樣的來源標準。
5. 更新本課開頭的查證日期。

## 8. 讀技術報告的清單

讀一份新的模型報告時，用本課程的詞彙逐項找：

| 看什麼 | 對應課程 |
|---|---|
| tokenizer 與詞彙量 | 03 |
| 架構：norm、位置編碼、MLP、attention 變體、MoE、residual | 04–06、本課 §2–4 |
| 最佳化器、學習率排程、batch、精度 | 06、08 |
| 參數量、訓練 token 數、FLOPs、資料／參數比 | 07 |
| 資料來源、過濾、去重、配比、合成資料、污染處理 | 10 |
| 後訓練：SFT 資料、偏好資料、RL 的獎勵與演算法 | 11–12 |
| 推論：context 長度、KV cache、量化、speculative decoding | 09 |
| 評估：用了哪些 benchmark、怎麼抽答案、有沒有污染檢查 | 14 |

## 9. 延伸閱讀（一手來源）

- Fedus 等人（2021），Switch Transformer：<https://arxiv.org/abs/2101.03961>；Jiang 等人（2024），Mixtral：<https://arxiv.org/abs/2401.04088>
- DeepSeek-AI（2024），DeepSeek-V3：<https://arxiv.org/abs/2412.19437>；DeepSeek-AI（2026），DeepSeek-V4：<https://arxiv.org/abs/2606.19348>
- Kimi Team（2025），Kimi K2：<https://arxiv.org/abs/2507.20534>；Kimi Linear：<https://arxiv.org/abs/2510.26692>
- MiniMax（2026），MiniMax Sparse Attention：<https://arxiv.org/abs/2606.13392>
- OpenAI（2025），gpt-oss model card：<https://arxiv.org/abs/2508.10925>
- Qwen Team（2025），Qwen3 技術報告：<https://arxiv.org/abs/2505.09388>
- Yang 等人（2024），Gated DeltaNet：<https://arxiv.org/abs/2412.06464>；Gu & Dao（2023），Mamba：<https://arxiv.org/abs/2312.00752>；Dao & Gu（2024），Mamba-2：<https://arxiv.org/abs/2405.21060>
- Xiao 等人（2023），attention sinks：<https://arxiv.org/abs/2309.17453>
- Gloeckle 等人（2024），多 token 預測：<https://arxiv.org/abs/2404.19737>
- Hugging Face 模型卡（2026-09-25 查閱）：<https://huggingface.co/deepseek-ai>、<https://huggingface.co/Qwen>、<https://huggingface.co/moonshotai>、<https://huggingface.co/MiniMaxAI>、<https://huggingface.co/zai-org>、<https://huggingface.co/mistralai>、<https://huggingface.co/openai>
- 擴散式語言模型等「非自迴歸」路線：本 repo 的 [`diffusion-models-course/lessons/11_frontier_2026.md`](../../diffusion-models-course/lessons/11_frontier_2026.md) §7〈文字也能用擴散生成〉
