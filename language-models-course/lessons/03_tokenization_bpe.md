# 第 03 課：Tokenization——從位元組到 byte-level BPE

> 前置：第 01 課 ｜ 實驗：[`notebooks/03_bpe.ipynb`](../notebooks/03_bpe.ipynb) ｜ 預估時間：2 小時

## 這一課要回答的問題

- 模型看到的不是「字」，是 token。token 是怎麼決定的？
- 詞彙量該多大？太大、太小各有什麼代價？
- GPT-2、GPT-4、Llama、nanochat 的 tokenizer 差在哪裡？為什麼數字要特別處理？
- 特殊 token（`<|endoftext|>`、`<|user_start|>`…）是做什麼的？

> 本課與 repo 裡的 [`deep-learning-karpathy/tutorials/01_tokenizer/`](../../deep-learning-karpathy/tutorials/01_tokenizer/)（minBPE 導讀）互補：那邊逐步帶讀程式，這邊著重設計取捨與效率，並把 GPT-2 的真實詞彙表載進自己的實作。

---

## 1. 白話版

### 1.1 切多細？

要把文字餵給模型，得先切成一塊一塊、每塊給一個編號：

- **切成單字**：「unbelievable」是一塊。但世界上的字太多，新字、拼錯的字、網址都會變成「不認識」。
- **切成字母（或位元組）**：永遠不會有不認識的東西，但一句話變得很長，模型要多做很多步。
- **中間路線（subword）**：常見的字整塊（「the」），少見的字拆成幾塊（「un」「believ」「able」）。BPE 就是自動找出這些塊的方法。

### 1.2 BPE：把最常黏在一起的兩塊合併

從最小的單位（位元組）開始，數一數整份資料裡**哪兩個相鄰的塊最常一起出現**，把它們合併成一個新塊、加進詞彙表。重複幾千次。
一開始合併的是「t＋h」「th＋e」這種，後來會出現「 happy」「 Once」這種整個字。合併的順序就是 tokenizer 的全部內容。

### 1.3 先切成詞再合併

如果放任 BPE 合併，會出現「dog.」「dog!」「dog?」各佔一個名額這種浪費。GPT-2 先用一個正規表示式把文字切成「字母一群、數字一群、標點一群、空白一群」，**合併只在同一群裡發生**。

### 1.4 特殊 token

有些編號不代表任何文字，而是訊號：「一篇新文件開始了」「使用者說完了」「換 AI 說話」。它們不能從普通文字裡被「拼」出來——否則使用者打一串字就能假冒系統訊號。

---

## 2. 正式版

### 2.1 從字元到位元組

Unicode 有十幾萬個字元（code point），GPT-2 論文當時的說法是「超過 130,000」。用它們當基本單位，詞彙表還沒合併就已經太大。
**UTF-8** 把每個字元編成 1–4 個位元組：ASCII 1 個、多數拉丁延伸字母 2 個、中日韓漢字 3 個、emoji 4 個。以位元組為基本單位，基本詞彙只有 256 個，而且**任何字串都能表示**——沒有「不認識的字」。這就是 byte-level BPE（GPT-2 首先使用）。

### 2.2 取捨：詞彙量 vs. 序列長度

設平均每個 token 代表 $b$ 個位元組（壓縮率）。對一段 $N$ 位元組的文字：

- 序列長度 $T \approx N / b$。每個 token 的計算量約是 $2P$ FLOPs（$P$ 為參數量，第 07 課），attention 還多一項和 $T$ 成正比的成本；**$b$ 越大，同樣的文字越便宜**，也讓固定的 context length 裝得下更多內容。
- 詞彙量 $|V|$ 越大，$b$ 越大，但 embedding 與 LM head 各多 $|V| \cdot d$ 個參數，而且稀有 token 的訓練樣本少。

典型值：GPT-2 50,257；GPT-4（cl100k_base）約 10 萬；nanochat 32,768（$2^{15}$）；本課程 4,096（CPU 上的小模型，embedding 不能太大）。

### 2.3 訓練演算法

令語料被預切（§2.4）成一串 pre-token，每個 pre-token 是一個位元組序列。

```
vocab ← 256 個單一位元組；merges ← []
重複 |V| − 256 − (特殊 token 數) 次：
    對所有 pre-token 內的相鄰配對 (a, b) 計數（乘上該 pre-token 的出現次數）
    (a*, b*) ← 次數最多的配對（平手時取位元組較大的，讓結果確定）
    新 token ← a* 與 b* 串接；把所有 pre-token 裡的 (a*, b*) 換成新 token
    merges.append((a*, b*))
```

效率的關鍵：

1. **pre-token 只數一次**：TinyStories 的訓練故事（約 20 MB）切出約 470 萬個 pre-token，但不同的只有約 1.3 萬個。之後每一步只在這 1.3 萬個「字」上工作。
2. **增量更新配對計數**：合併 $(a^*, b^*)$ 只影響含有它的字；維護「配對 → 含它的字」的索引，每步只更新那些字。

本課程的實作在 22 MB 的 TinyStories 上訓練 4,096 個 token 約 10 秒（純 Python）。

### 2.4 預切：正規表示式

| tokenizer | 預切規則的重點 |
|---|---|
| GPT-2（2019） | `'s|'t|…| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`：縮寫、（可帶一個前導空白的）字母串、數字串、符號串、空白 |
| GPT-4 cl100k（2023） | 縮寫不分大小寫；**數字最多 3 位一組** `\p{N}{1,3}`；換行獨立處理 |
| Llama 1（2023） | SentencePiece BPE，**每個數字拆成單一位數**，不認識的字元退回位元組 |
| nanochat | GPT-4 的規則，但數字改成最多 2 位一組 `\p{N}{1,2}`（作者實測 32K 詞彙時 2 最好） |
| 本課程 | GPT-4 的規則，數字**一律單一位數** `\p{N}` |

注意空白的處理：「 world」的前導空白和字黏在一起，所以「hello world」是 `hello`、` world` 兩個 token。

**為什麼數字要特別處理？** 若數字可以任意合併，「1234」可能被切成「12」「34」或「123」「4」，同一個數字在不同上下文的切法不同，模型要學的加法表爆炸。單一位數讓每一位都對齊；Singh & Strouse（2024）也發現切法（例如 3 位一組、從左或從右切）會明顯影響前沿模型的算術正確率。本課第 11–12 課要教小模型做加法，所以選單一位數。

### 2.5 編碼：照訓練順序套用合併

給一個新的 pre-token，從位元組開始，**反覆找「最早學到的」那個相鄰配對合併**，直到沒有可合併的配對。這保證編碼結果和訓練時對同一字串的切法一致。
因為 pre-token 高度重複，把每個 pre-token 的結果快取起來，編碼 22 MB 只要幾秒。

### 2.6 特殊 token

| token | 用途 |
|---|---|
| GPT-2 `<|endoftext|>`（id 50256） | 文件分隔 |
| nanochat `<|bos|>` | 每份文件／每段對話的開頭（名字不同，作用相同） |
| nanochat `<|user_start|>` … `<|output_end|>` | 對話角色與工具呼叫（第 11 課） |

規則：**特殊 token 只能由程式插入，不能由文字編碼產生**。本課程的 `encode()` 預設把 `"<|bos|>"` 這串字當普通文字切；只有明確傳 `allowed_special=True` 才轉成特殊 id（tiktoken 更嚴格：預設遇到就報錯）。

### 2.7 GPT-2 的 `vocab.json` 為什麼看起來怪怪的

GPT-2 把每個位元組先對應到一個「看得見的」Unicode 字元（`bytes_to_unicode`）：可列印的 Latin-1 位元組對應到自己，其餘 68 個（控制字元、空白…）搬到 U+0100 以後。所以詞彙表裡的空白顯示成「Ġ」，換行是「Ċ」。這只是儲存格式，和演算法無關。Lab 03 會把 GPT-2 的 `vocab.json` 與 `merges.txt` 讀進本課程的 `BPETokenizer`，並驗證編碼結果與 OpenAI 的 tiktoken 完全一致。

### 2.8 tokenizer 造成的怪現象

- **數不出 strawberry 有幾個 r**：模型看到的是「str」「aw」「berry」之類的塊，不是字母。nanochat 有一篇教模型做這件事的指南（見延伸閱讀）。
- **非英文比較貴**：同一句話翻成不同語言，token 數可以差好幾倍（Petrov 等人 2023），等於那些語言的使用者付更多錢、context 更短。
- **Glitch token**：詞彙表裡有、訓練資料裡幾乎沒出現的 token（例如 GPT-2／3 的「 SolidGoldMagikarp」），embedding 幾乎沒被訓練，模型遇到會行為異常。
- **結尾空白**：prompt 以空白結尾時，模型的下一個 token 通常「本來應該」帶著前導空白，結果分布被扭曲。

### 2.9 其他路線

- **SentencePiece／Unigram LM**（Kudo 2018；Kudo & Richardson 2018）：從大詞彙表往下刪、用機率選切法，Llama、T5 等採用。
- **不用 tokenizer**：ByT5 直接吃位元組；Byte Latent Transformer（2024）把位元組動態分組成 patch。代價是序列變長。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/tokenizer.py`） |
|---|---|
| 預切規則 | `GPT2_SPLIT_PATTERN`、`GPT4_SPLIT_PATTERN`、`COURSE_SPLIT_PATTERN` |
| §2.3 訓練（增量更新、確定的平手規則） | `BPETokenizer.train` |
| §2.5 編碼（最早的合併優先、pre-token 快取） | `BPETokenizer._encode_chunk`、`encode_ordinary` |
| §2.6 特殊 token | `encode(text, allowed_special=...)`、`special_id`、`is_special` |
| bits-per-byte 需要的「每個 token 幾個位元組」 | `token_byte_lengths` |
| §2.7 載入 GPT-2 的詞彙 | `BPETokenizer.from_gpt2_files`、`gpt2_byte_to_unicode`、`gpt2_tokenizer` |
| 課程共用的 tokenizer（4,096 個 token，含 chat 特殊 token） | `artifacts.course_tokenizer` |

測試：`test_every_learned_token_stays_inside_one_pre_token`（合併不跨越預切邊界）、`test_classic_example_first_merge_is_the_most_frequent_pair`、`test_gpt2_tokenizer_matches_known_ids`（需要網路）。

## 4. 常見誤解

- **「一個 token 就是一個字」**：英文平均約 4 個位元組一個 token；中文常常一個字就要 1–2 個 token（本課程的 tokenizer 只在英文故事上訓練，中文會退回位元組，一個字 3 個 token）。
- **「詞彙越大越好」**：壓縮率變好，但 embedding 參數變多、稀有 token 學不好；小模型尤其吃虧。
- **「tokenizer 和模型無關，可以隨時換」**：模型的 embedding 表是按 token 編號學的，換 tokenizer 等於換一個模型。
- **「BPE 學出來的都是有意義的詞根」**：它只看頻率，會學出「keleton」「opter」這種片段（Lab 03 看得到）。

## 5. 練習

**想一想**

1. 「你好」在 UTF-8 下是幾個位元組？一個只用英文訓練的 byte-level BPE 會把它切成幾個 token？
2. 為什麼編碼時要「照訓練順序」合併，而不是「每次合併出現最多次的配對」？舉一個兩者結果不同的例子。
3. 詞彙量從 4,096 增加到 32,768，對一個 $d = 256$ 的模型，embedding 加 LM head（不綁定）多了多少參數？和 4 層 Transformer 的參數量比一比。
4. 如果 `encode()` 預設允許特殊 token，會有什麼安全問題？

**動手改**（在 `03_bpe.ipynb`）

5. 用 GPT-2 的預切規則重新訓練，數字的切法有什麼不同？壓縮率呢？
6. 把詞彙量掃過 512、1024、4096、16384，畫出壓縮率曲線。它會一直變好嗎？
7. 找出本課 tokenizer 的「glitch token 候選」：詞彙表裡有、但在驗證資料裡從沒出現的 token。

## 6. 延伸閱讀

- Sennrich 等人（2015），BPE 用於神經機器翻譯：<https://arxiv.org/abs/1508.07909>
- Radford 等人（2019），GPT-2 論文 §2.2〈Input Representation〉：<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- Karpathy，〈Let's build the GPT Tokenizer〉（影片）：<https://www.youtube.com/watch?v=zduSFxRajkE>；程式 minbpe（MIT）：<https://github.com/karpathy/minbpe>
- OpenAI tiktoken（MIT）：<https://github.com/openai/tiktoken>
- CS336 作業 1（BPE tokenizer 是第一部分）：<https://github.com/stanford-cs336/assignment1-basics>
- nanochat 的 tokenizer（rustbpe 訓練、tiktoken 推論）與〈counting r in strawberry〉指南：<https://github.com/karpathy/nanochat/discussions/164>
- Singh & Strouse（2024），〈Tokenization counts〉：<https://arxiv.org/abs/2402.14903>
- Petrov 等人（2023），tokenizer 對不同語言的不公平：<https://arxiv.org/abs/2305.15425>
- Rumbelow & Watkins（2023），〈SolidGoldMagikarp〉：<https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation>
- Kudo & Richardson（2018），SentencePiece：<https://arxiv.org/abs/1808.06226>
