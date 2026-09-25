# 第 10 課：資料——從網頁到訓練資料

> 前置：第 01、06 課 ｜ 實驗：[`notebooks/10_data.ipynb`](../notebooks/10_data.ipynb) ｜ 預估時間：2 小時

## 這一課要回答的問題

- 大型語言模型的訓練資料從哪裡來？原始的網頁長什麼樣子？
- 過濾規則、去重、品質分類器各在處理什麼問題？MinHash 為什麼能在幾十億份文件裡找近似重複？
- 資料配比怎麼決定？「評估資料污染」是什麼、怎麼檢查？
- 合成資料能取代真實資料嗎？

> CS336 用兩講（第 13、14 講）和一整份作業（作業 4：把 Common Crawl 的原始 dump 變成可用的預訓練資料）講這件事——它的重要性常被低估：在同樣的模型與算力下，資料品質的差異往往比架構的差異更大。

---

## 1. 白話版

### 1.1 網路是一座很亂的圖書館

大部分訓練資料來自網路爬蟲（例如 Common Crawl，每月抓幾十億個網頁）。但抓回來的東西大多不是好文章：導覽列、「點此訂閱」、商品關鍵字堆疊、亂碼、表格、一模一樣的轉貼。直接拿來訓練，模型會學到很多沒用、甚至有害的模式。

### 1.2 清理的流水線

1. **抽出正文**：把 HTML 裡的選單、廣告拿掉。
2. **語言辨識**：只留下要的語言。
3. **規則過濾**：太短、太長、符號太多、沒有常見虛詞（the、and…）的文件丟掉。
4. **去重**：一模一樣的丟掉；**幾乎一樣的**（改了幾個字、加了頁首頁尾）也丟掉。
5. **品質模型**：訓練一個分類器判斷「像不像好文章」，只留分數高的。
6. **配比**：網頁、程式碼、書、論文、數學各放多少。

### 1.3 考題不能出現在課本裡

評估模型用的考題（benchmark）如果出現在訓練資料裡，模型就是在背答案，分數沒有意義。所以要檢查訓練資料和考題有沒有大段重疊，這叫**污染檢查**。

---

## 2. 正式版

### 2.1 資料來源與規模

| 資料集 | 年份 | 重點 |
|---|---|---|
| C4（T5） | 2019 | Common Crawl 一個月的快照＋啟發式規則；Dodge 等人 2021 做了詳細的內容稽核 |
| The Pile | 2020 | 22 個來源（學術、程式、書、網頁…），不同來源重複不同次數 |
| RefinedWeb（Falcon） | 2023 | 主張「只用網頁、但認真過濾與去重」就夠好 |
| Dolma（OLMo） | 2024 | 3 兆 token，完整公開處理流程 |
| FineWeb／FineWeb-Edu | 2024 | 15 兆 token；Edu 版用 LLM 標註「教育價值」後訓練分類器篩選 |
| DCLM | 2024 | 一個「固定模型、比較資料」的競賽框架；基線用 fastText 品質分類器 |
| NVIDIA ClimbMix | 2025 | 依主題分群後迭代調整配比；nanochat 自 2026-03 起的預訓練資料 |

GPT-2 的 WebText 是另一種思路：只收 Reddit 上有人推薦（≥3 karma）的外連網頁——用人的判斷當品質過濾器（第 05 課）。
**授權與著作權**：網頁資料的法律地位在各司法管轄區仍在變動；本課程只使用授權明確的 TinyStories（CDLA-Sharing-1.0）。

### 2.2 規則過濾：以 Gopher 為例

Rae 等人（2021，附錄 A）的 MassiveWeb 規則（本課程的 `gopher_failures` 照原文實作）：

- 字數在 50 到 100,000 之間；平均字長 3 到 10 個字元；
- 「#」或「...」與字數的比例不超過 0.1；
- 以項目符號開頭的行不超過 90%；以刪節號結尾的行不超過 30%；
- 至少 80% 的字含有字母；
- 至少包含 the、be、to、of、and、that、have、with 中的兩個（擋掉「看起來是英文、其實不是句子」的東西）；
- 重複性（表 A1）：重複行的比例 ≤ 0.30、最常見的 2-gram 佔的字元比例 ≤ 0.20…（本課程實作其中兩項）。

Gopher 論文刻意**不用髒話字表**過濾（改用 SafeSearch），因為 Dodge 等人發現字表過濾會不成比例地刪掉少數族群相關的無害內容。

### 2.3 去重

**完全重複**：對正規化（小寫、去標點、合併空白）後的文字取雜湊。

**近似重複**：把文件表示成 word $n$-gram 的集合（shingles），用 Jaccard 相似度

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|} \tag{10.1}$$

兩兩比較 $N$ 份文件需要 $O(N^2)$，不可行。**MinHash**（Broder 1997）：取一個隨機雜湊函數 $h$，令 $m_h(A) = \min_{a \in A} h(a)$，則

$$\Pr\left[m_h(A) = m_h(B)\right] = J(A, B) \tag{10.2}$$

（$A \cup B$ 中雜湊值最小的元素，恰好落在 $A \cap B$ 的機率。）用 $K$ 個雜湊函數得到長度 $K$ 的簽章，相同的比例就是 Jaccard 的不偏估計。

**LSH 分帶**：把簽章切成 $b$ 帶、每帶 $r$ 個值（$K = br$），只要有一帶完全相同就列為候選。相似度 $s$ 的兩份文件成為候選的機率

$$P(s) = 1 - \left(1 - s^{r}\right)^{b} \tag{10.3}$$

是一條 S 形曲線，轉折點約在 $(1/b)^{1/r}$：只需要比對同一個桶裡的文件。

為什麼重要：Lee 等人（2021）發現常用資料集裡有大量近似重複，去重後模型更好、背誦訓練資料的情況大幅減少，而且訓練與驗證集之間的重疊也被揭露。

### 2.4 用模型過濾

- **困惑度過濾**（CCNet，Wenzek 等人 2019）：在 Wikipedia 上訓練一個 n-gram 語言模型，網頁的困惑度越低越像「好文字」。
- **品質分類器**：GPT-3 用「WebText 等高品質來源 vs 原始 Common Crawl」訓練分類器；DCLM 用 fastText（hashed n-gram 的線性模型）；FineWeb-Edu 先讓大模型替樣本打分再訓練小分類器。
- **風險**：分類器學的是「像不像正例」，正例的風格（例如 Wikipedia）會被放大，其他寫法的好文字可能被誤殺。

### 2.5 配比與退火

不同來源的權重決定模型的能力分布（程式碼多→寫程式強）。做法從手調（The Pile 讓高品質來源多重複幾次）到自動化（DoReMi 用小模型學權重、ClimbMix 迭代搜尋主題配比）。另一個常見技巧：在學習率遞減的最後一段（第 06 課的 warmdown）換成更高品質的資料，稱為退火（annealing）或 mid-training。

### 2.6 污染

benchmark 的題目常出現在網路上，進而進入訓練資料。檢查方式：

- **n-gram 重疊**：GPT-3 以 13-gram 為單位，找出與訓練資料有重疊的測試題，並報告乾淨子集的分數；
- **canary string**：benchmark 在檔案裡放一串獨特的字串，模型若能補完它，代表見過資料；
- 更精細的偵測：看模型對某段文字的機率是否異常地高（Shi 等人 2023 的 Min-K% Prob）。

### 2.7 合成資料

TinyStories 本身就是合成資料（GPT-3.5／4 生成）；Phi 系列用「教科書風格」的合成資料訓練小模型（Gunasekar 等人 2023）。優點是可控；風險是分布變窄、錯誤被放大，模型只在自己生成的資料上反覆訓練會逐漸退化（Shumailov 等人 2023 稱為 model collapse）。Lab 10 §5 會看到 TinyStories 的另一個特徵：故事之間高度相似。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/data_pipeline.py`） |
|---|---|
| §2.2 Gopher 規則 | `QualityRules`、`gopher_failures`、`top_ngram_char_fraction` |
| 完全重複 | `normalize`、`exact_duplicates` |
| (10.1)–(10.3) | `shingles`、`jaccard`、`MinHasher`、`minhash_similarity`、`lsh_candidate_probability`、`lsh_candidate_pairs`、`near_duplicates`（union-find 分群） |
| §2.4 品質分類器 | `BagOfNgramsClassifier`、`train_quality_classifier`；困惑度過濾用第 01 課的 `ngram.InterpolatedLM` |
| §2.6 污染 | `word_ngrams`、`contaminated` |
| 練習用的「網頁語料」 | `make_web_corpus`（真實故事＋完全重複＋近似重複＋五種垃圾） |

測試：`test_minhash_estimates_jaccard`、`test_lsh_probability_is_an_s_curve`、`test_each_rule_catches_its_kind_of_junk`、`test_near_duplicates_are_clustered_and_distinct_texts_kept`。

## 4. 常見誤解

- **「資料越多越好」**：在固定算力下，更乾淨的資料通常勝過更多的髒資料；但過濾太兇會丟掉多樣性。
- **「去重只是為了省算力」**：它也減少背誦、降低隱私風險、避免評估被高估。
- **「品質分類器分數高 ＝ 好文章」**：它只代表「像正例」；正例怎麼選，決定了模型的風格與偏見。
- **「污染檢查做了就安全」**：改寫過的題目、翻譯過的題目，n-gram 比對抓不到。

## 5. 練習

**想一想**

1. 證明 (10.2)。提示：$A \cup B$ 中每個元素成為最小雜湊值的機率相同。
2. $K = 128$、$(b, r) = (32, 4)$ 與 $(16, 8)$ 的 S 曲線轉折點各在哪裡？想抓 Jaccard ≥ 0.8 的配對，該選哪一個？
3. 一個 13-gram 比對，對「數學題改了數字」的污染有用嗎？對程式題呢？
4. 如果品質分類器的正例全是 Wikipedia，哪些類型的好文字最可能被誤殺？

**動手改**（在 `10_data.ipynb`）

5. 把近似重複的門檻從 0.7 調到 0.5 與 0.9，漏抓與誤抓各怎麼變？
6. 替 Gopher 規則加上「重複 5-gram 的字元比例 ≤ 0.15」，能多抓到哪一類垃圾？
7. 困惑度過濾的 n-gram 模型如果改在垃圾文件上訓練，分數會怎麼顛倒？

## 6. 延伸閱讀

- Rae 等人（2021），Gopher（附錄 A 的資料管線）：<https://arxiv.org/abs/2112.11446>
- Lee 等人（2021），〈Deduplicating Training Data Makes Language Models Better〉：<https://arxiv.org/abs/2107.06499>
- Broder（1997），〈On the resemblance and containment of documents〉（MinHash 原始論文，*Compression and Complexity of Sequences*）；Leskovec、Rajaraman、Ullman，《Mining of Massive Datasets》第 3 章（LSH）：<http://www.mmds.org/>
- Wenzek 等人（2019），CCNet：<https://arxiv.org/abs/1911.00359>
- Raffel 等人（2019），T5／C4：<https://arxiv.org/abs/1910.10683>；Dodge 等人（2021），C4 的稽核：<https://arxiv.org/abs/2104.08758>
- Penedo 等人（2023），RefinedWeb：<https://arxiv.org/abs/2306.01116>；Penedo 等人（2024），FineWeb：<https://arxiv.org/abs/2406.17557>
- Li 等人（2024），DataComp-LM（DCLM）：<https://arxiv.org/abs/2406.11794>
- Soldaini 等人（2024），Dolma：<https://arxiv.org/abs/2402.00159>
- Xie 等人（2023），DoReMi：<https://arxiv.org/abs/2305.10429>；Diao 等人（2025），Nemotron-CLIMB：<https://arxiv.org/abs/2504.13161>
- Shi 等人（2023），偵測預訓練資料（Min-K% Prob）：<https://arxiv.org/abs/2310.16789>
- Gunasekar 等人（2023），〈Textbooks Are All You Need〉：<https://arxiv.org/abs/2306.11644>；Shumailov 等人（2023），〈The Curse of Recursion〉：<https://arxiv.org/abs/2305.17493>
- CS336 第 13–14 講與作業 4：<https://cs336.stanford.edu/>、<https://github.com/stanford-cs336/assignment4-data>
