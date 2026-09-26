# 第 14 課：評估，以及 nanochat 的全流程導讀

> 前置：第 06–13 課 ｜ 本課沒有新的實驗（回顧 Lab 06、11、12 的評估方式） ｜ 預估時間：1.5 小時

## 這一課要回答的問題

- 怎麼知道一個語言模型「好」？perplexity、benchmark、人類偏好各量到什麼、漏掉什麼？
- base model 還不會對話，怎麼考它選擇題？
- nanochat 從零到一個能聊天的模型，每一步在做什麼？對應到本課程的哪一課？

---

## 1. 白話版

### 1.1 三種考法

- **看它有多不驚訝**（loss、bits-per-byte）：便宜、穩定、每一步都能量，但不直接等於「好用」。
- **考試**（benchmark）：選擇題（常識、科學、各科知識）、數學題、寫程式。分數好解讀，但題目可能被「看過」（污染），而且換個出題格式分數就會變。
- **請人評**（或請更強的模型評）：最接近真實使用，但貴、慢、主觀。

### 1.2 怎麼考還不會對話的模型

base model 只會接著寫。考選擇題時不叫它「回答」，而是把每個選項分別接在題目後面，看模型覺得哪一個**最順**（機率最高）——這樣連很小的模型都能考。

### 1.3 nanochat：一條龍

nanochat 把整條路放在一個小而完整的程式庫裡：訓練 tokenizer → 預訓練 → 評估 → 微調成聊天模型 → （選配）強化學習 → 網頁聊天介面。本課程的每一課，都對應到其中一段。

---

## 2. 正式版

### 2.1 內在指標

驗證集上的 loss 或 bits-per-byte（第 01 課式 1.4）。nanochat 以 `val_bpb` 為主要的預訓練指標（它與 tokenizer 無關，換詞彙量也能比）。限制：它衡量的是「像訓練資料的文字」預測得多好，不直接反映能力或有用性。

### 2.2 base model 的評估：用機率答題

選擇題（題目 $x$、選項 $c_1, \dots, c_K$）：

$$\hat k = \arg\max_k \frac{1}{|c_k|}\log p_\theta(c_k \mid x) \tag{14.1}$$

（除以選項長度是「長度正規化」，避免長選項吃虧；也有不正規化或除以位元組數的版本。）HellaSwag、ARC、PIQA 等都可以這樣考；few-shot 時在題目前面放幾個有答案的例子。

**CORE**（DCLM，Li 等人 2024）：22 個在小模型上也有穩定訊號的任務，每個任務的正確率先線性縮放成「隨機猜 = 0、全對 = 1」的**置中正確率**，再平均：

$$\text{centered acc} = \frac{\text{acc} - \text{acc}_{\text{random}}}{1 - \text{acc}_{\text{random}}}, \qquad \text{CORE} = \frac{1}{22}\sum_{\text{task}}\text{centered acc} \tag{14.2}$$

nanochat 的「time to GPT-2」以 GPT-2 1.5B 的 CORE 0.2565 為門檻（第 05 課 §2.7）。

### 2.3 chat model 的評估：生成後判分

- **選擇題**：讓模型輸出選項字母（MMLU、ARC）。
- **數學**（GSM8K）：抽出最後的數字與標準答案比對——和本課程 `extract_answer` 的做法一樣，也和它有一樣的風險（格式錯了就算錯）。
- **程式**（HumanEval）：執行單元測試，報告 pass@k（第 12 課式 12.11）。
- nanochat 的 **ChatCORE**：把 ARC-Easy、ARC-Challenge、MMLU（隨機基線 0.25）、GSM8K、HumanEval（基線 0）的置中正確率平均（查證 2026-09-25，`scripts/chat_sft.py`）。

### 2.4 開放式回答：人與模型當裁判

- **人類偏好**：Chatbot Arena 讓使用者對兩個匿名模型的回答投票，用 Bradley–Terry 模型（第 12 課式 12.4）估出每個模型的分數。
- **LLM-as-judge**：MT-Bench 用強模型打分。便宜，但有已知偏差：偏好較長的回答、偏好排在前面的回答、偏好和自己風格相似的回答（Zheng 等人 2023）。

### 2.5 評估的陷阱

- **污染**（第 10 課 §2.6）。
- **格式敏感**：few-shot 的例子數、選項的排列、是否長度正規化、答案抽取的規則，都能讓分數差好幾個百分點；OLMES（Gu 等人 2024）試圖把這些選擇標準化。
- **Goodhart 定律**：指標一旦成為目標，就不再是好指標；只看一個排行榜會導致過度擬合那個排行榜。
- **本課程的例子**：Lab 06 §5 的 250 步比較只有一個種子；Lab 12 的主題獎勵只用一組種子、每個主題 16 個樣本、一種提問句型量。報告數字時要附上「在什麼條件下」。

### 2.6 nanochat 全流程導讀

| 階段 | nanochat（查證 2026-09-25，master） | 本課程 |
|---|---|---|
| 資料 | 預訓練資料分片（2026-03 起為 NVIDIA ClimbMix）、`nanochat/dataset.py` 下載 | TinyStories（第 10 課講資料管線） |
| Tokenizer | `scripts/tok_train.py`：rustbpe 訓練、tiktoken 推論，32,768 個 token，GPT-4 式的預切（數字最多 2 位） | 第 03 課：純 Python 的 byte-level BPE，4,096 個 token，數字 1 位 |
| 模型 | `nanochat/gpt.py`：RoPE、QK-norm、ReLU²、無參數 RMSNorm、不綁 embedding、soft-cap、GQA、滑動視窗、value embeddings… | 第 04–06 課：`nanochat_style_config` 實作其中的核心 |
| 預訓練 | `scripts/base_train.py`：`--depth` 單一旋鈕、Muon＋AdamW、warmdown、資料／參數比、bf16／fp8、ZeRO-2 式的分散式最佳化器 | 第 06–08 課、Lab 06 |
| 評估 | `scripts/base_eval.py`：CORE、bits-per-byte、抽樣 | 本課 §2.1–2.2 |
| SFT | `scripts/chat_sft.py`：SmolTalk＋MMLU＋GSM8K 的混合、只算助理 token 的 loss、ChatCORE | 第 11 課、Lab 11 |
| RL（選配，speedrun 不含） | `scripts/chat_rl.py`：GSM8K 上的簡化 GRPO | 第 12 課、Lab 12 |
| 推論 | `nanochat/engine.py`：KV cache、批次抽樣、Python 工具；`scripts/chat_cli.py`、網頁介面 | 第 09 課、`chat.ChatEngine` |

**規模對照**（訓練 FLOPs）：

| 模型 | 訓練算力 | 出處 |
|---|---|---|
| 本課程 Lab 06 的 base model | 約 $6 \times 4.2\text{M} \times 12.3\text{M} \approx 3 \times 10^{14}$ | 第 07 課式 7.1（參與矩陣乘法的參數約 420 萬） |
| nanochat speedrun | 約 $4 \times 10^{19}$ | nanochat README |
| GPT-3 175B | 約 $3 \times 10^{23}$ | $6 \times 175\text{B} \times 300\text{B}$ |
| Llama 3 405B | $3.8 \times 10^{25}$ | Llama 3 技術報告 |

本課程的 CPU 實驗比 nanochat 的 speedrun 小約 10 萬倍、比前沿模型小約 1,000 億倍——但每一個步驟的原理都相同。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| bits-per-byte | `training.evaluate(..., token_bytes=...)` |
| 生成後判分（加法） | `chat.extract_answer`、`chat.addition_reward`、`chat.ChatEngine` |
| pass@k | `rl.pass_at_k` |
| 污染檢查 | `data_pipeline.contaminated` |
| 課程的迷你全流程 | `artifacts`（tokenizer → base model）→ Lab 11（SFT）→ Lab 12（RL） |

## 4. 常見誤解

- **「benchmark 分數高就代表好用」**：benchmark 只涵蓋少數能力，而且可能被污染或被過度最佳化。
- **「loss 低一點，所有 benchmark 都會好一點」**：大致如此，但個別任務的曲線可能很不平滑（第 07 課 §2.7）。
- **「LLM 當裁判很客觀」**：它有系統性的偏好（長度、位置、風格）。
- **「小模型的評估沒意義」**：選對任務（像 CORE 那樣在小規模也有訊號）與指標（連續的 loss、置中正確率），小模型的實驗可以預測大模型的趨勢——這正是 scaling laws 與 nanochat miniseries 的前提。

## 5. 練習

1. 四選一的題目，隨機猜的正確率是 25%。一個模型答對 40%，置中正確率是多少？
2. (14.1) 不做長度正規化時，會偏好長選項還是短選項？為什麼？
3. 設計一個實驗，檢查 Lab 12 的模型在「開頭提到主題」上的進步是否只是學會了特定的提問句型（提示：RL 只用了「Tell me a story about a ...」；換成 `chat.STORY_TEMPLATES` 裡的其他問法，或自己寫一個訓練時沒出現過的問法）。
4. 用第 07 課的公式驗證本課 §2.6 表格裡 Lab 06 的訓練算力。

## 6. 延伸閱讀

- Li 等人（2024），DataComp-LM（CORE 的定義在附錄 G）：<https://arxiv.org/abs/2406.11794>
- Hendrycks 等人（2020），MMLU：<https://arxiv.org/abs/2009.03300>；Clark 等人（2018），ARC：<https://arxiv.org/abs/1803.05457>；Zellers 等人（2019），HellaSwag：<https://arxiv.org/abs/1905.07830>；Cobbe 等人（2021），GSM8K：<https://arxiv.org/abs/2110.14168>；Chen 等人（2021），HumanEval 與 pass@k：<https://arxiv.org/abs/2107.03374>
- Zheng 等人（2023），MT-Bench 與 LLM-as-judge：<https://arxiv.org/abs/2306.05685>；Chiang 等人（2024），Chatbot Arena：<https://arxiv.org/abs/2403.04132>
- Gu 等人（2024），OLMES：<https://arxiv.org/abs/2406.08446>
- CS336 第 12 講（評估，Percy Liang）：<https://cs336.stanford.edu/>
- nanochat（MIT）：<https://github.com/karpathy/nanochat>；〈Beating GPT-2 for <<$100〉：<https://github.com/karpathy/nanochat/discussions/481>；原始發表文（2025-10，內容部分已過時）：<https://github.com/karpathy/nanochat/discussions/1>
