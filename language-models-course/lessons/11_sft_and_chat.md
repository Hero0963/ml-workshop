# 第 11 課：從 base model 到聊天助理——chat 格式、SFT 與工具使用

> 前置：第 03、06、09 課 ｜ 實驗：[`notebooks/11_sft.ipynb`](../notebooks/11_sft.ipynb)（需要 Lab 06 的模型） ｜ 預估時間：2 小時

## 這一課要回答的問題

- 預訓練完的模型（base model）為什麼不會「回答問題」？
- 對話要怎麼變成一串 token？SFT 的 loss 為什麼只算助理說的話？
- 模型怎麼學會「呼叫計算機」？工具的輸出為什麼不能拿來訓練？
- SFT 能教會模型什麼、教不會什麼？

---

## 1. 白話版

### 1.1 base model 只會「接著寫」

預訓練的模型學的是「網路上的文件接下來會寫什麼」。問它「法國的首都是哪裡？」，它可能接著寫「義大利的首都是哪裡？」——因為網路上的題庫就長這樣。它**有**知識，但不知道現在的角色是「回答問題的助理」。

### 1.2 示範幾萬次對話

**監督式微調（SFT）**：準備大量「使用者問、助理答」的範例，用固定的格式（特殊 token 標出誰在說話）串成序列，繼續做「猜下一個 token」的訓練。模型因此學到：看到「換助理說話」的記號，就該寫出一個好回答，寫完放一個「說完了」的記號。

### 1.3 只從助理的話學

使用者的問題是給定的，不需要學著「生成」它。所以 loss 只算助理的 token：使用者的部分當作上下文，不計分。

### 1.4 會用工具

小模型心算很差，但會寫算式。訓練資料示範：助理先寫「<|python_start|>512+389<|python_end|>」，程式替它算出 901 塞回去，助理再接著說「答案是 901」。模型要學的是**什麼時候該呼叫、該寫什麼**；計算的結果是程式給的，不是模型學的，所以那一段也不計分。

---

## 2. 正式版

### 2.1 對話的渲染

以 nanochat 的格式為例（本課程沿用相同的特殊 token 名稱）：

```
<|bos|><|user_start|>What is 23 + 45?<|user_end|><|assistant_start|>23 + 45 = 68. The answer is 68.<|assistant_end|>
```

工具呼叫出現在助理的回合裡：

```
<|assistant_start|><|python_start|>512+389<|python_end|><|output_start|>901<|output_end|>The answer is 901.<|assistant_end|>
```

渲染的結果是 token 序列 $x_{1:T}$ 與遮罩 $m_{1:T} \in \{0, 1\}$：助理寫的內容（包含 `<|python_start|>`、算式、`<|python_end|>`、`<|assistant_end|>`）為 1；`<|bos|>`、使用者的回合、`<|assistant_start|>` 本身、工具輸出 `<|output_start|>…<|output_end|>` 為 0。

### 2.2 SFT 的 loss

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{\sum_{t} m_t \log p_\theta(x_t \mid x_{<t})}{\sum_t m_t} \tag{11.1}$$

實作上是把 $m_t = 0$ 的目標設成 $-1$，讓交叉熵忽略它（`ignore_index`）。和預訓練的差別**只有資料與遮罩**：模型、最佳化器、loss 函數都一樣。

**為什麼要遮罩？** (1) 模型不需要學會生成使用者的話；(2) 長的使用者輸入（例如貼一整篇文章要摘要）會主宰 loss。不過這不是絕對的：Shi 等人（2024）發現在訓練資料少、回答短時，把指令也算進 loss 反而有幫助。Lab 11 §5 會比較兩者。

### 2.3 SFT 的資料

- **人寫的示範**：InstructGPT 的 SFT 資料約 1.3 萬個 prompt，由標註人員撰寫回答（Ouyang 等人 2022）。
- **把現有資料集改寫成指令格式**：FLAN（Wei 等人 2021；Chung 等人 2022）。
- **模型生成**：Self-Instruct（Wang 等人 2022）用模型自己產生指令與回答，再過濾。
- **nanochat 的混合**（查證 2026-09-25，`scripts/chat_sft.py`）：SmolTalk 的一般對話（約 46 萬筆）＋ MMLU 輔助訓練集 3 個 epoch（教選擇題格式）＋ GSM8K 4 個 epoch（教數學與工具使用）。
- **少而精**：LIMA（Zhou 等人 2023）只用 1,000 筆精選範例就得到不錯的助理，提出「表面對齊假說」：知識與能力幾乎都在預訓練時學到，SFT 主要教的是格式與風格。

**Midtraining**：nanochat 早期（2025-10 起，至 2026-01 底）在 base 與 SFT 之間多一個 midtraining 階段，用大量對話資料先教特殊 token、選擇題與工具使用；2026-02 初這一步被併入 SFT（`scripts/mid_train.py` 在 2026-01-29 的 commit 還在、2026-02-02 的 commit 已移除）。

### 2.4 工具使用

推論時，引擎（第 09 課 §2.5）在模型吐出 `<|python_start|>` 後開始收集 token，看到 `<|python_end|>` 就執行，把結果以 `<|output_start|>結果<|output_end|>` **強制**接到序列後面（這幾個 token 不是抽樣來的），然後讓模型繼續。
這就是 PAL（Gao 等人 2022，讓模型寫程式解題）與 Toolformer（Schick 等人 2023，讓模型學會插入 API 呼叫）的共同精神：**把模型不擅長的精確計算交給確定性的程式**。模型要學的只有兩件事：何時呼叫、寫出正確的呼叫。

### 2.5 SFT 的極限

- **行為複製**：SFT 讓模型模仿示範，不知道自己的答案對不對；遇到示範沒涵蓋的情況容易亂答，也可能學會「用自信的口吻編造」（幻覺）。
- **學新知識很沒效率**：SFT 的資料量小，用它灌輸新事實效果差，還可能增加幻覺。
- **只能跟示範一樣好**：要超越示範，需要一個「判斷好壞」的訊號——第 12 課的偏好學習與強化學習。

### 2.6 省參數的微調：LoRA

大模型全參數微調很貴。LoRA（Hu 等人 2021）凍結原權重 $W$，只學一個低秩的修正 $W + BA$（$B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$，$r \ll d$），可訓練參數少上千倍。本課程的模型很小，直接全參數微調。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/chat.py`） |
|---|---|
| 特殊 token（與 nanochat 同名） | `SPECIAL_TOKENS`（第 03 課訓練 tokenizer 時就已加入） |
| §2.1 渲染與遮罩 | `render_conversation`、`render_for_completion` |
| (11.1) 批次與忽略的目標 | `sft_batch`（目標 = 下一個 token，遮罩為 0 的位置設為 −1）；loss 用 `training.lm_loss` |
| 有標準答案的任務：兩數相加 | `AdditionProblem`、`addition_problems`、`extract_answer`、`addition_reward` |
| 講故事的對話（保留預訓練的能力） | `story_conversations` |
| §2.4 計算機工具與推論引擎 | `calculator`（用 `ast` 解析，只允許整數四則運算，不用 `eval`）、`ChatEngine` |

測試：`test_only_assistant_tokens_are_trained`、`test_tool_output_is_context_but_the_call_is_trained`、`test_sft_targets_are_shifted_and_masked`、`test_engine_runs_the_tool_when_the_model_calls_it`（用一個照劇本輸出的假模型，確認引擎在正確的時機執行工具、插入結果）、`test_calculator_is_safe`。

## 4. 常見誤解

- **「SFT 讓模型變聰明」**：它主要改變的是格式與行為；能力的上限在預訓練時就決定了（LIMA 的主張）。
- **「特殊 token 是寫在文字裡的標籤」**：它們是獨立的 token id，不能從一般文字編碼出來（第 03 課 §2.6）；模型在 SFT 之前從沒見過它們，embedding 是隨機的。
- **「工具的結果也要讓模型學」**：那會教模型「自己編出工具的輸出」；推論時結果由程式提供。
- **「loss 下降就代表對話變好」**：SFT 的 loss 只衡量模仿示範的程度；要用任務指標（Lab 11 的加法正確率）或人工評估。

## 5. 練習

**想一想**

1. 一段對話渲染成 60 個 token，其中使用者 15 個、助理回答 20 個、特殊 token 其餘。(11.1) 的分母是多少？
2. 為什麼 `<|assistant_start|>` 的遮罩是 0，而 `<|assistant_end|>` 是 1？
3. 如果訓練資料裡 2 位數的加法都不用工具、3 位數的都用工具，模型在 4 位數的題目上會怎麼做？
4. LoRA 在一個 $4096 \times 4096$ 的矩陣上用 $r = 8$，可訓練參數佔原矩陣的多少？

**動手改**（在 `11_sft.ipynb`）

5. 拿掉故事對話，只用加法資料做 SFT，模型還會講故事嗎？
6. 把工具範例的比例調低，模型在 3 位數加法時還會呼叫工具嗎？
7. 對 base model 用「少樣本」提示（在 prompt 裡放三題加法範例），不做 SFT，正確率是多少？

## 6. 延伸閱讀

- Ouyang 等人（2022），InstructGPT：<https://arxiv.org/abs/2203.02155>
- Wei 等人（2021），FLAN：<https://arxiv.org/abs/2109.01652>；Chung 等人（2022）：<https://arxiv.org/abs/2210.11416>
- Wang 等人（2022），Self-Instruct：<https://arxiv.org/abs/2212.10560>
- Zhou 等人（2023），LIMA：<https://arxiv.org/abs/2305.11206>
- Shi 等人（2024），〈Instruction Tuning With Loss Over Instructions〉：<https://arxiv.org/abs/2405.14394>
- Gao 等人（2022），PAL：<https://arxiv.org/abs/2211.10435>；Schick 等人（2023），Toolformer：<https://arxiv.org/abs/2302.04761>
- Hu 等人（2021），LoRA：<https://arxiv.org/abs/2106.09685>
- nanochat 的 `nanochat/tokenizer.py`（`render_conversation`）、`scripts/chat_sft.py`、`nanochat/engine.py`：<https://github.com/karpathy/nanochat>
- CS336 第 15 講（mid／post-training）與作業 5：<https://cs336.stanford.edu/>、<https://github.com/stanford-cs336/assignment5-alignment>
