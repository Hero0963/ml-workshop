# 開發日誌（逆時序，最新在上）

## 2026-09-26：執行全部實驗、依結果修正程式與講義

- **做了什麼**：依序執行 Lab 01–13（一次一本，不並行），每本執行後逐條核對 markdown 的敘述與實際輸出，不符的改文字或改程式後重跑；每本各自 commit。
- **RL 任務換成「主題跟隨」**（第 12 課、Lab 12）：
  - 原設計是 3 位數加法（step-by-step 格式）的 GRPO。SFT 模型 greedy 約 0.78，但 temperature 1 只有約 0.23。試過 AdamW 學習率 1e-5／3e-5／5e-5／1e-4、有無 10 步 warmup、只訓練 Transformer 區塊、temperature 0.7、改用 SGD：沒有 warmup 會崩；有 warmup 時 150 步後 temperature 1 的正確率仍在 0.2 上下，greedy 反而掉到 0.5–0.7。
  - 梯度方向檢查（30 題 × 16 樣本，一個大批次的 policy gradient，沿梯度走一步後用同一組亂數重新量）：原本 0.246；步長 +0.5 → 0.252、+2.0 → 0.235、−2.0 → 0.069。梯度方向對，但「變好」的幅度遠小於雜訊。temperature 1 的錯誤幾乎都是某一位的個位數加法差 1，分散在上百個數字事實上。
  - 改用「Tell me a story about a {topic}.」＋「前 40 個 token 有沒有提到主題字」的獎勵：SFT 模型 pass@32 在沒看過的主題上已有 0.9，組內有好有壞，訊號集中。原型（60 步 × 8 prompt × 16 樣本，AdamW 3e-5，10 步 warmup，約 2.5 分鐘）：訓練主題 temperature 1 獎勵 0.36 → 0.81、greedy 0.70 → 0.95；沒看過的主題 0.13 → 0.47、greedy 0.3 → 0.6。
  - 副產品是清楚的 reward hacking（「the horse played with the horse」）；加上 KL 懲罰（k3 估計量，β = 0.1）後 KL/token 0.065 → 0.034、重複率 0.041 → 0.028，獎勵 0.81 → 0.73。DPO（187 對「有提到 vs 沒提到」、2 epoch）：0.36 → 0.49，KL/token 只有 0.005。
  - 程式：`rl.completion_logprobs` 加 `temperature`（抽樣用的溫度必須和算 log-prob 的一致）、`rl.kl_penalty`、`chat.mentions_topic`／`story_request`／`HELD_OUT_TOPICS`；加法 RL 保留為 Lab 12 §7 的反例。
- **n-gram 重寫**（Lab 01 的輸出和講義對不上）：
  - 固定幾何權重的插值（均勻分布拿 $0.5^n$、每低一階權重減半）在 $n \le 6$ 輸給加 k 平滑，抽樣還會冒出亂碼位元組；前文用 256 進位整數當鍵，最多 7 階，看不到驗證 bpb 的 U 形。
  - 改成 Witten–Bell 插值（$\lambda(h) = c(h) / (c(h) + u(h))$）與乘法雜湊的前文鍵（任意階）。結果：加 k 在 $n = 7$ 見底（1.405）後回升到 1.885（$n = 10$）；Witten–Bell 在 $n = 9$ 最好（1.240）。
  - 「長前文會背書」的敘述原本不成立（平滑過的 7-gram 抽樣幾乎沒有逐字複製）；改用不平滑的 MLE 抽樣：$n = 20$ 時 20 字元的片段 100%、40 字元的片段 34% 原封不動出現在訓練資料裡——拼貼。
- **word2vec 玩具語料的 bug**：先前修改 `toy_world_corpus` 時把 `with_roles` 的判斷弄丟了，永遠會加上角色情境，所以 Lab 02 §2 的反例（son 與 boy 分不開）不會出現。修正並加測試（`test_toy_world_roles_are_what_tell_son_from_boy`）；修正後沒有角色情境時性別類比 0.41，錯的答案都是「性別對、角色錯」。
- **執行時發現、改了程式的地方**：
  - Lab 05：把 0 維 tensor 直接交給 `tokenizer.decode` 會 KeyError（詞彙表的鍵是 int），改用 `.tolist()`。
  - Lab 08：`torch.cumsum` 在 CPU 上對 bf16 用較高精度累加，示範不出「小數字被吃掉」；改成逐一相加的迴圈（bf16 停在 0.5、fp16 停在 4.0）。roofline 的頻寬原本取最大值（10 萬個元素、在快取裡，155 GB/s），轉折點算成 3 FLOP/byte；改用 1,000 萬個元素量到的主記憶體頻寬。pipeline bubble 要 134 個 micro-batch 才低於 5%，原本的範圍只到 128。
  - Lab 09：speculative decoding 的分布檢查原本印了一個猜的雜訊範圍（0.03–0.05），實際 TV 距離 0.081；改成同時量「直接從目標模型抽 3,000 次」的 TV（0.079），兩者一樣。
  - Lab 10：Matplotlib 3.9 起 `boxplot(labels=)` 改名 `tick_labels=`。LSH 圖上 J≈0.85 的點是「只有 1 對文件」的組（那一對的命中機率本來就只有約 90%），改成只畫至少 5 對的組。
  - Lab 12 §7：「溫度 1 的錯誤通常是某一步差 1」只對一半——實際分類（313 個錯誤答案）：差 1 佔 50%、格式跑掉 42%、其他 8%；notebook 改成直接印出這個分類與兩種例子。
- **只改文字的修正**：BPE 最後的合併沒有「keleton」（改舉實際出現的「 blin」「ummer」）；課程 tokenizer 在 TinyStories 上的壓縮率是 4.05 位元組／token，GPT-2 是 4.04（原文寫「只差一些」）；PPMI-SVD 在 $k = 1$ 時全部答對（原文寫「比 SGNS 差」）；Lab 06 的 loss 曲線看不出 WSD 遞減段的「額外一截」（改寫並在講義加練習 9 讓讀者和固定學習率比較）；Lab 06 §5 的架構比較差距遠大於雜訊，但 GPT-2 變體少了約 100 萬參數、學習率沒有另外調（加註）；Lab 08 的 Python 分塊 attention 在 $T \ge 1024$ 反而比 naive 版快（快取）；Lab 10 的 Gopher 規則因為頁首頁尾是同一行而擋下 34 份近似重複、13-gram 重疊是 42% 而不是「大多數」；Lab 11 只算助理 token 的 SFT 在 step by step 高 10 個百分點（76% vs 66%，在雜訊範圍內）；Lab 13 原本寫「對比訓練前 alignment 小、uniformity 大」，實際上那是 word2vec 平均的樣子，預訓練模型的正例反而比較遠，對比訓練同時改善兩者。
- **實測數字**：base model 驗證 1.967 nats／token、0.705 bits/byte（GPT-2 124M 零樣本 0.872、最好的 byte n-gram 1.240）；Lab 07 的 IsoFLOP 擬合 $N^* \propto C^{0.63}$；SFT 後加法正確率（2 位數／3 位數）直接 14%／0%、step by step 89%／76%、計算機 100%／100%；embedding 檢索 recall@1：BM25 0.82、預訓練模型直接平均 0.34、對比訓練 300 步 0.82–0.84、從隨機初始化對比訓練 0.37。
- **授權清單**：用程式搜尋每本 notebook 輸出裡與 TinyStories 相同的 60 字元片段，更新 `NOTICE.md` 的節錄清單（原本列了 Lab 03，實際沒有；另外把模型生成、可能逐字重現故事片段的輸出也列出來）。

## 2026-09-25：建立課程 v1

- **任務**：本人交辦「參考 `diffusion-models-course` 的方式製作一份教程，涵蓋 nanochat、GPT-2、Stanford CS336、embedding model、word2vec 的核心內容，重疊處自行斟酌；注意版權授權；每做一個段落就 commit and push」。
- **做了什麼**：
  - 授權查證（2026-09-25）：nanochat MIT；CS336 作業 repo MIT 式（Stanford 版權）、講義 PDF 未附授權 → 只連結；GPT-2 Modified MIT（權重執行時下載）；TinyStories CDLA-Sharing-1.0；nanoGPT／minbpe／llm.c／tiktoken MIT。結果寫在 `NOTICE.md`。
  - 事實查證：約 180 篇 arXiv 論文逐篇核對標題、日期（部分核對第一作者）；GPT-2、word2vec、Gopher、Kaplan、Chinchilla、InstructGPT、DCLM、Llama 3、Gemma 2 的關鍵數字從論文 PDF 原文核對；H100 規格取自 NVIDIA 官網；nanochat 的設計讀自 master 的原始碼；2026 年的模型版本與規格從 Hugging Face 組織頁（依建立日期排序）與 model card 取得。
  - 參考實作 `src/lm_course/`（16 個模組）與測試；講義 16 課＋附錄 A；實驗 notebook 13 本。
- **關鍵驗證**（開發時用 scratch 環境的 transformers 與 tiktoken 交叉比對，兩者都不是本專案的相依套件）：
  - 自寫的 byte-level BPE 讀入 GPT-2 的 `vocab.json`／`merges.txt` 後，對 4 組測試字串（含 2 萬字元的 TinyStories）的編碼與 tiktoken 完全相同。
  - 自寫的 GPT 載入 OpenAI 的 GPT-2 124M 權重後，logits 與 Hugging Face `GPT2LMHeadModel` 最大差 2e-4（logit 量級 266），argmax 全部相同。
- **原型實驗**（決定實驗規模的依據，數字為 scratch 執行）：
  - CPU 吞吐量（4 核、batch 32×256）：GPT-2 形狀 d192 L4 約 11k token/s；nanochat 形狀 d256 L4 約 7.5k token/s；d384 L6 約 3.4k token/s → base model 選 d256 L4。
  - 預訓練原型 1,500 步：驗證 loss 1.97 nats／token、0.705 bits/byte，30 分鐘（與其他工作共用 CPU），故事已通順。
  - SFT 原型 v1（只用「直接寫答案」＋工具＋故事，600 步）：直接回答 2 位數加法只有 4%——模型抄對了兩個加數，但寫不出正確的和（包括訓練過的題目）。
  - SFT 原型 v2（加入 step-by-step 格式，800 步）：直接 14%／0%、step by step 89%／76%、計算機 100%／100%（2 位數／3 位數，各 100 題未見過的題目）。三種格式的差距成為第 11 課 §2.4 的主要教材。
- **踩到的坑**：
  - Hugging Face 的 548 MB 權重下載被代理截斷（少了約 0.5 MB）但沒有報錯；`data.download` 改成比對 Content-Length、用 HTTP Range 續傳。截斷的檔案依 repo 規則移到 `soft-delete/20260925_221657/`。
  - `GPTConfig` 原本在 `__post_init__` 填入 `n_kv_head`、`d_ff` 的預設值，`dataclasses.replace` 會帶著舊值（例如 12 個 KV head）去建新設定而出錯；改成 `kv_heads`、`ff_dim` 屬性。
  - PyTorch 的 `FlopCounterMode` 在 CPU 上看不到 fused 的 `scaled_dot_product_attention`，量到的剛好是 $6N$；測試改成分開驗證 $6N$ 與 attention 項（`attention_reference` 的 FLOPs 恰為 $12dT$／token／層）。
  - speculative decoding 的「接受率」若除以全部草稿 token，會把第一次拒絕之後沒被檢查的 token 也算進去；改成只除以被目標模型檢查過的 token（`SpeculativeStats.checked`），測試對上 $\sum\min(p, q)$。
  - word2vec 玩具語料一開始的性別類比只有約 50%：son 與 boy、brother 與 uncle 的上下文完全相同，模型本來就分不出來。加入每對人物專屬的情境（`ROLES`）後 100%；保留 `with_roles=False` 當作 Lab 02 的反例。
  - 查前沿模型時發現 2026 年的版本（DeepSeek-V4、Qwen3.8、Kimi K3、GLM-5.3、MiniMax-M3…）比助理的訓練資料新；第 15 課全部改以 model card 為準。
  - 第 07 課初稿寫「Kaplan 與 Chinchilla 的差異來自學習率排程沒降完」，讀了 Porian 等人（2024）的摘要後改正：他們的結論是 LM head 的算力、暖身長度、最佳化器調參，且學習率遞減並非關鍵。
