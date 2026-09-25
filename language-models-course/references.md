# 參考資料與延伸閱讀

> **查證日期：2026-09-25。** arXiv 論文都已逐篇打開摘要頁，核對標題與首次提交日期；網頁資源確認可以連線；模型與資料集的授權取自官方 repo 的 `LICENSE` 或 Hugging Face 的 model／dataset card。
> 標記：**一手**＝論文、官方文件、官方課程或程式庫；**二手**＝整理文章或部落格。
> 本清單只列連結，不轉載內容。程式庫的授權各自不同，使用前請看該 repo 的 `LICENSE`。

## 1. 本課程整合的五個來源

| 資源 | 類型 | 授權 | 本課 |
|---|---|---|---|
| Stanford CS336〈Language Modeling from Scratch〉，Percy Liang、Tatsunori Hashimoto：<https://cs336.stanford.edu/>（2026 春；2025 春：<https://cs336.stanford.edu/spring2025/>）；作業：<https://github.com/stanford-cs336/assignment1-basics>（2–5 同名 repo） | 一手（課程） | 作業 repo：MIT 式，Stanford 版權；講義 PDF：未附授權，只連結 | 骨幹 |
| nanochat，Andrej Karpathy：<https://github.com/karpathy/nanochat>；〈Beating GPT-2 for <<$100〉：<https://github.com/karpathy/nanochat/discussions/481>；〈miniseries v1〉：<https://github.com/karpathy/nanochat/discussions/420> | 一手（程式與作者的說明） | MIT | 06、09、11、12、14 |
| GPT-2：Radford 等人（2019）<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>；程式與 model card <https://github.com/openai/gpt-2> | 一手 | Modified MIT | 03–05 |
| word2vec：Mikolov 等人（2013a）<https://arxiv.org/abs/1301.3781>；（2013b）<https://arxiv.org/abs/1310.4546> | 一手 | 論文 | 02 |
| Embedding 模型：Reimers & Gurevych（2019）Sentence-BERT <https://arxiv.org/abs/1908.10084>；MTEB <https://arxiv.org/abs/2210.07316> | 一手 | 論文 | 13 |

## 2. 教材、影片與程式

| 資源 | 類型 |
|---|---|
| Karpathy，〈Let's build GPT: from scratch, in code, spelled out.〉<https://www.youtube.com/watch?v=kCc8FmEb1nY>；〈Let's build the GPT Tokenizer〉<https://www.youtube.com/watch?v=zduSFxRajkE>；〈Let's reproduce GPT-2 (124M)〉<https://www.youtube.com/watch?v=l8pRSuU81PU> | 一手（作者的影片） |
| nanoGPT（MIT）<https://github.com/karpathy/nanoGPT>；minbpe（MIT）<https://github.com/karpathy/minbpe>；llm.c（MIT）<https://github.com/karpathy/llm.c>；tiktoken（MIT）<https://github.com/openai/tiktoken> | 一手（程式） |
| 本 repo 的 [`deep-learning-karpathy/`](../deep-learning-karpathy/)（minBPE、nanoGPT 導讀）與 [`diffusion-models-course/`](../diffusion-models-course/) | 同 repo |
| Jurafsky & Martin，《Speech and Language Processing》第 3 版草稿（第 3 章 n-gram、第 6 章詞向量）<https://web.stanford.edu/~jurafsky/slp3/> | 一手（教科書） |
| Stanford CS224N <https://web.stanford.edu/class/cs224n/> | 一手（課程） |
| Keller Jordan，〈Muon: An optimizer for hidden layers in neural networks〉（2024-12）<https://kellerjordan.github.io/posts/muon/> | 一手（作者的部落格） |
| Leskovec、Rajaraman、Ullman，《Mining of Massive Datasets》（第 3 章 MinHash／LSH）<http://www.mmds.org/> | 一手（教科書） |
| Rumbelow & Watkins，〈SolidGoldMagikarp〉<https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation> | 二手 |

## 3. 一手論文（依課程順序）

日期為 arXiv 首次提交日（非 arXiv 者註明出處）。

### 基礎（第 01–03 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Shannon，Prediction and Entropy of Printed English，*Bell System Technical Journal* 30(1) | 1951 | 01 |
| Bengio 等人，A Neural Probabilistic Language Model，*JMLR* 3 <https://www.jmlr.org/papers/v3/bengio03a.html> | 2003 | 01 |
| Mikolov 等人，Efficient Estimation of Word Representations in Vector Space <https://arxiv.org/abs/1301.3781> | 2013-01 | 02 |
| Mikolov 等人，Distributed Representations of Words and Phrases and their Compositionality <https://arxiv.org/abs/1310.4546> | 2013-10 | 02 |
| Goldberg & Levy，word2vec Explained <https://arxiv.org/abs/1402.3722> | 2014-02 | 02 |
| Levy & Goldberg，Neural Word Embedding as Implicit Matrix Factorization（NeurIPS）<https://proceedings.neurips.cc/paper_files/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html> | 2014 | 02 |
| Pennington 等人，GloVe（EMNLP）<https://aclanthology.org/D14-1162/> | 2014 | 02 |
| Levy、Goldberg、Dagan，Improving Distributional Similarity…（TACL）<https://aclanthology.org/Q15-1016/> | 2015 | 02 |
| Bojanowski 等人，Enriching Word Vectors with Subword Information（fastText）<https://arxiv.org/abs/1607.04606> | 2016-07 | 02 |
| Bolukbasi 等人，Man is to Computer Programmer as Woman is to Homemaker? <https://arxiv.org/abs/1607.06520> | 2016-07 | 02 |
| Nissim 等人，Fair is Better than Sensational <https://arxiv.org/abs/1905.09866> | 2019-05 | 02 |
| Sennrich 等人，Neural Machine Translation of Rare Words with Subword Units <https://arxiv.org/abs/1508.07909> | 2015-08 | 03 |
| Kudo，Subword Regularization <https://arxiv.org/abs/1804.10959>；Kudo & Richardson，SentencePiece <https://arxiv.org/abs/1808.06226> | 2018 | 03 |
| Xue 等人，ByT5 <https://arxiv.org/abs/2105.13626> | 2021-05 | 03 |
| Touvron 等人，LLaMA <https://arxiv.org/abs/2302.13971> | 2023-02 | 03 |
| Petrov 等人，Language Model Tokenizers Introduce Unfairness Between Languages <https://arxiv.org/abs/2305.15425> | 2023-05 | 03 |
| Singh & Strouse，Tokenization counts <https://arxiv.org/abs/2402.14903> | 2024-02 | 03 |
| Pagnoni 等人，Byte Latent Transformer <https://arxiv.org/abs/2412.09871> | 2024-12 | 03 |

### 模型與預訓練（第 04–06 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Bahdanau 等人，Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/abs/1409.0473> | 2014-09 | 04 |
| Vaswani 等人，Attention Is All You Need <https://arxiv.org/abs/1706.03762> | 2017-06 | 04 |
| Ba 等人，Layer Normalization <https://arxiv.org/abs/1607.06450>；Zhang & Sennrich，RMSNorm <https://arxiv.org/abs/1910.07467> | 2016-07／2019-10 | 04 |
| Hendrycks & Gimpel，GELU <https://arxiv.org/abs/1606.08415>；Shazeer，GLU Variants <https://arxiv.org/abs/2002.05202>；So 等人，Primer <https://arxiv.org/abs/2109.08668> | 2016／2020／2021 | 04 |
| Xiong 等人，On Layer Normalization in the Transformer Architecture <https://arxiv.org/abs/2002.04745> | 2020-02 | 04 |
| Su 等人，RoFormer <https://arxiv.org/abs/2104.09864>；Press 等人，ALiBi <https://arxiv.org/abs/2108.12409>；Haviv 等人，NoPE <https://arxiv.org/abs/2203.16634> | 2021／2021／2022 | 04 |
| Shazeer，MQA <https://arxiv.org/abs/1911.02150>；Ainslie 等人，GQA <https://arxiv.org/abs/2305.13245>；Dehghani 等人，ViT-22B（QK-norm）<https://arxiv.org/abs/2302.05442> | 2019／2023／2023 | 04 |
| Press & Wolf，Using the Output Embedding to Improve Language Models <https://arxiv.org/abs/1608.05859> | 2016-08 | 02、05 |
| Holtzman 等人，The Curious Case of Neural Text Degeneration <https://arxiv.org/abs/1904.09751> | 2019-04 | 05 |
| Brown 等人，GPT-3 <https://arxiv.org/abs/2005.14165> | 2020-05 | 05、07 |
| Kingma & Ba，Adam <https://arxiv.org/abs/1412.6980>；Loshchilov & Hutter，AdamW <https://arxiv.org/abs/1711.05101> | 2014／2017 | 06 |
| Bernstein & Newhouse，Old Optimizer, New Norm <https://arxiv.org/abs/2409.20325> | 2024-09 | 06 |
| Liu 等人，Muon is Scalable for LLM Training <https://arxiv.org/abs/2502.16982> | 2025-02 | 06 |
| Amsel 等人，The Polar Express <https://arxiv.org/abs/2505.16932>；NorMuon <https://arxiv.org/abs/2510.05491> | 2025-05／10 | 06 |
| Hu 等人，MiniCPM（WSD）<https://arxiv.org/abs/2404.06395> | 2024-04 | 06 |
| Riviere 等人，Gemma 2（logit soft-capping）<https://arxiv.org/abs/2408.00118> | 2024-07 | 06 |
| Eldan & Li，TinyStories <https://arxiv.org/abs/2305.07759> | 2023-05 | 06 |
| Muennighoff 等人，Scaling Data-Constrained Language Models <https://arxiv.org/abs/2305.16264> | 2023-05 | 06、07 |

### 規模、系統與推論（第 07–09 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Kaplan 等人，Scaling Laws for Neural Language Models <https://arxiv.org/abs/2001.08361> | 2020-01 | 07 |
| Hoffmann 等人，Training Compute-Optimal Large Language Models <https://arxiv.org/abs/2203.15556> | 2022-03 | 07 |
| Porian 等人，Resolving Discrepancies in Compute-Optimal Scaling <https://arxiv.org/abs/2406.19146>；Pearce & Song，Reconciling Kaplan and Chinchilla <https://arxiv.org/abs/2406.12907> | 2024-06 | 07 |
| Chowdhery 等人，PaLM <https://arxiv.org/abs/2204.02311> | 2022-04 | 07 |
| Yang 等人，Tensor Programs V（μP）<https://arxiv.org/abs/2203.03466> | 2022-03 | 07 |
| Schaeffer 等人，Are Emergent Abilities… a Mirage? <https://arxiv.org/abs/2304.15004> | 2023-04 | 07 |
| Grattafiori 等人，The Llama 3 Herd of Models <https://arxiv.org/abs/2407.21783> | 2024-07 | 07、08 |
| Korthikanti 等人，Reducing Activation Recomputation <https://arxiv.org/abs/2205.05198> | 2022-05 | 07、08 |
| Milakov & Gimelshein，Online normalizer calculation for softmax <https://arxiv.org/abs/1805.02867> | 2018-05 | 08 |
| Rabe & Staats，Self-attention Does Not Need O(n²) Memory <https://arxiv.org/abs/2112.05682> | 2021-12 | 08 |
| Dao 等人，FlashAttention <https://arxiv.org/abs/2205.14135>；FlashAttention-2 <https://arxiv.org/abs/2307.08691>；FlashAttention-3 <https://arxiv.org/abs/2407.08608> | 2022／2023／2024 | 08 |
| Micikevicius 等人，Mixed Precision Training <https://arxiv.org/abs/1710.03740> | 2017-10 | 08 |
| Rajbhandari 等人，ZeRO <https://arxiv.org/abs/1910.02054>；Shoeybi 等人，Megatron-LM <https://arxiv.org/abs/1909.08053>；Liu 等人，Ring Attention <https://arxiv.org/abs/2310.01889> | 2019／2019／2023 | 08 |
| Leviathan 等人，Speculative Decoding <https://arxiv.org/abs/2211.17192>；Chen 等人，Speculative Sampling <https://arxiv.org/abs/2302.01318> | 2022-11／2023-02 | 09 |
| Cai 等人，Medusa <https://arxiv.org/abs/2401.10774>；Li 等人，EAGLE <https://arxiv.org/abs/2401.15077> | 2024-01 | 09 |
| Kwon 等人，PagedAttention（vLLM）<https://arxiv.org/abs/2309.06180> | 2023-09 | 09 |
| DeepSeek-AI，DeepSeek-V2（MLA）<https://arxiv.org/abs/2405.04434> | 2024-05 | 09 |
| Dettmers 等人，LLM.int8() <https://arxiv.org/abs/2208.07339>；Frantar 等人，GPTQ <https://arxiv.org/abs/2210.17323>；Lin 等人，AWQ <https://arxiv.org/abs/2306.00978> | 2022／2022／2023 | 09 |

### 資料與評估（第 10、14 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Rae 等人，Gopher（附錄 A：資料管線）<https://arxiv.org/abs/2112.11446> | 2021-12 | 10 |
| Lee 等人，Deduplicating Training Data Makes Language Models Better <https://arxiv.org/abs/2107.06499> | 2021-07 | 10 |
| Wenzek 等人，CCNet <https://arxiv.org/abs/1911.00359> | 2019-11 | 10 |
| Raffel 等人，T5／C4 <https://arxiv.org/abs/1910.10683>；Dodge 等人，Documenting Large Webtext Corpora <https://arxiv.org/abs/2104.08758> | 2019／2021 | 10 |
| Gao 等人，The Pile <https://arxiv.org/abs/2101.00027>；Penedo 等人，RefinedWeb <https://arxiv.org/abs/2306.01116>；Soldaini 等人，Dolma <https://arxiv.org/abs/2402.00159>；Penedo 等人，FineWeb <https://arxiv.org/abs/2406.17557> | 2020–2024 | 10 |
| Li 等人，DataComp-LM（DCLM，CORE）<https://arxiv.org/abs/2406.11794> | 2024-06 | 10、14 |
| Xie 等人，DoReMi <https://arxiv.org/abs/2305.10429>；Diao 等人，Nemotron-CLIMB <https://arxiv.org/abs/2504.13161> | 2023／2025 | 10 |
| Shi 等人，Detecting Pretraining Data（Min-K% Prob）<https://arxiv.org/abs/2310.16789> | 2023-10 | 10 |
| Gunasekar 等人，Textbooks Are All You Need <https://arxiv.org/abs/2306.11644>；Shumailov 等人，The Curse of Recursion <https://arxiv.org/abs/2305.17493> | 2023 | 10 |
| Hendrycks 等人，MMLU <https://arxiv.org/abs/2009.03300>；Clark 等人，ARC <https://arxiv.org/abs/1803.05457>；Zellers 等人，HellaSwag <https://arxiv.org/abs/1905.07830>；Cobbe 等人，GSM8K <https://arxiv.org/abs/2110.14168>；Chen 等人，HumanEval／pass@k <https://arxiv.org/abs/2107.03374> | 2018–2021 | 14 |
| Zheng 等人，MT-Bench／LLM-as-judge <https://arxiv.org/abs/2306.05685>；Chiang 等人，Chatbot Arena <https://arxiv.org/abs/2403.04132>；Gu 等人，OLMES <https://arxiv.org/abs/2406.08446> | 2023／2024 | 14 |

### 後訓練（第 11–12 課）

| 論文 | 日期 | 課 |
|---|---|---|
| Ouyang 等人，InstructGPT <https://arxiv.org/abs/2203.02155> | 2022-03 | 11、12 |
| Wei 等人，FLAN <https://arxiv.org/abs/2109.01652>；Chung 等人，Scaling Instruction-Finetuned LMs <https://arxiv.org/abs/2210.11416>；Wang 等人，Self-Instruct <https://arxiv.org/abs/2212.10560> | 2021／2022 | 11 |
| Zhou 等人，LIMA <https://arxiv.org/abs/2305.11206>；Shi 等人，Instruction Tuning With Loss Over Instructions <https://arxiv.org/abs/2405.14394> | 2023／2024 | 11 |
| Nye 等人，Scratchpads <https://arxiv.org/abs/2112.00114>；Wei 等人，Chain-of-Thought <https://arxiv.org/abs/2201.11903>；Lee 等人，Teaching Arithmetic to Small Transformers <https://arxiv.org/abs/2307.03381> | 2021–2023 | 11 |
| Gao 等人，PAL <https://arxiv.org/abs/2211.10435>；Schick 等人，Toolformer <https://arxiv.org/abs/2302.04761> | 2022／2023 | 11 |
| Hu 等人，LoRA <https://arxiv.org/abs/2106.09685> | 2021-06 | 11 |
| Christiano 等人，Deep RL from Human Preferences <https://arxiv.org/abs/1706.03741>；Ziegler 等人 <https://arxiv.org/abs/1909.08593>；Stiennon 等人 <https://arxiv.org/abs/2009.01325> | 2017–2020 | 12 |
| Schulman 等人，PPO <https://arxiv.org/abs/1707.06347> | 2017-07 | 12 |
| Rafailov 等人，DPO <https://arxiv.org/abs/2305.18290>；Ethayarajh 等人，KTO <https://arxiv.org/abs/2402.01306> | 2023／2024 | 12 |
| Shao 等人，DeepSeekMath（GRPO）<https://arxiv.org/abs/2402.03300>；DeepSeek-AI，DeepSeek-R1 <https://arxiv.org/abs/2501.12948> | 2024／2025 | 12 |
| Yu 等人，DAPO <https://arxiv.org/abs/2503.14476>；Liu 等人，Understanding R1-Zero-Like Training（Dr. GRPO）<https://arxiv.org/abs/2503.20783>；Yue 等人，Does RL Really Incentivize Reasoning…? <https://arxiv.org/abs/2504.13837> | 2025 | 12 |
| Gao 等人，Scaling Laws for Reward Model Overoptimization <https://arxiv.org/abs/2210.10760>；Bai 等人，Constitutional AI <https://arxiv.org/abs/2212.08073>；Lambert 等人，Tülu 3 <https://arxiv.org/abs/2411.15124> | 2022／2024 | 12 |

### Embedding 模型（第 13 課）

| 論文 | 日期 |
|---|---|
| van den Oord 等人，CPC／InfoNCE <https://arxiv.org/abs/1807.03748>；Chen 等人，SimCLR <https://arxiv.org/abs/2002.05709>；Wang & Isola，Alignment and Uniformity <https://arxiv.org/abs/2005.10242> | 2018–2020 |
| Gao 等人，SimCSE <https://arxiv.org/abs/2104.08821>；Izacard 等人，Contriever <https://arxiv.org/abs/2112.09118> | 2021 |
| Karpukhin 等人，DPR <https://arxiv.org/abs/2004.04906>；Xiong 等人，ANCE <https://arxiv.org/abs/2007.00808>；Khattab & Zaharia，ColBERT <https://arxiv.org/abs/2004.12832> | 2020 |
| Wang 等人，E5 <https://arxiv.org/abs/2212.03533>；Li 等人，GTE <https://arxiv.org/abs/2308.03281>；Chen 等人，BGE M3 <https://arxiv.org/abs/2402.03216>；Wang 等人，Multilingual E5 <https://arxiv.org/abs/2402.05672> | 2022–2024 |
| Wang 等人，E5-Mistral <https://arxiv.org/abs/2401.00368>；BehnamGhader 等人，LLM2Vec <https://arxiv.org/abs/2404.05961>；Lee 等人，NV-Embed <https://arxiv.org/abs/2405.17428> | 2023–2024 |
| Kusupati 等人，Matryoshka <https://arxiv.org/abs/2205.13147> | 2022-05 |
| Muennighoff 等人，MTEB <https://arxiv.org/abs/2210.07316>；Enevoldsen 等人，MMTEB <https://arxiv.org/abs/2502.13595>；Su 等人，BRIGHT <https://arxiv.org/abs/2407.12883> | 2022–2025 |
| Zhang 等人，Qwen3 Embedding <https://arxiv.org/abs/2506.05176>；Lee 等人，Gemini Embedding <https://arxiv.org/abs/2503.07891>；Vera 等人，EmbeddingGemma <https://arxiv.org/abs/2509.20354>；Li 等人，BitNet Text Embeddings <https://arxiv.org/abs/2606.25674> | 2025–2026 |
| Malkov & Yashunin，HNSW <https://arxiv.org/abs/1603.09320>；Johnson 等人，FAISS <https://arxiv.org/abs/1702.08734> | 2016／2017 |

### 前沿（第 15 課）

| 論文 | 日期 |
|---|---|
| Fedus 等人，Switch Transformers <https://arxiv.org/abs/2101.03961>；Jiang 等人，Mixtral <https://arxiv.org/abs/2401.04088> | 2021／2024 |
| DeepSeek-AI，DeepSeek-V3 <https://arxiv.org/abs/2412.19437>；DeepSeek-V4 <https://arxiv.org/abs/2606.19348> | 2024-12／2026 |
| Kimi Team，Kimi K2 <https://arxiv.org/abs/2507.20534>；Kimi Linear <https://arxiv.org/abs/2510.26692> | 2025 |
| MiniMax，MiniMax Sparse Attention <https://arxiv.org/abs/2606.13392> | 2026-06 |
| OpenAI，gpt-oss-120b & gpt-oss-20b Model Card <https://arxiv.org/abs/2508.10925> | 2025-08 |
| Qwen Team，Qwen3 Technical Report <https://arxiv.org/abs/2505.09388> | 2025-05 |
| Yang 等人，Gated Delta Networks <https://arxiv.org/abs/2412.06464>；Gu & Dao，Mamba <https://arxiv.org/abs/2312.00752>；Dao & Gu，Mamba-2 <https://arxiv.org/abs/2405.21060> | 2023–2024 |
| Xiao 等人，Attention Sinks <https://arxiv.org/abs/2309.17453>；Gloeckle 等人，Multi-token Prediction <https://arxiv.org/abs/2404.19737> | 2023／2024 |

## 4. 模型、資料與規格頁（一手，查證 2026-09-25）

| 資源 | 內容 |
|---|---|
| GPT-2 124M：<https://huggingface.co/openai-community/gpt2> | Lab 05 下載的權重與詞彙檔 |
| TinyStories：<https://huggingface.co/datasets/roneneldan/TinyStories>；CDLA-Sharing-1.0 全文：<https://cdla.dev/sharing-1-0/> | 全部實驗的資料 |
| NVIDIA H100 規格：<https://www.nvidia.com/en-us/data-center/h100/> | 第 07、08 課的峰值與頻寬 |
| MTEB 排行榜：<https://huggingface.co/spaces/mteb/leaderboard> | 第 13 課 |
| Hugging Face 上各組織的模型清單（依建立日期排序）：deepseek-ai、Qwen、moonshotai、MiniMaxAI、zai-org、mistralai、openai、google、jinaai、nvidia、BAAI、microsoft | 第 13、15 課的「最新版本」查證 |
