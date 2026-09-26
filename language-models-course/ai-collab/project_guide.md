# language-models-course 專案指南

> 供接手的 agent／開發者快速上手：架構、模組職責、怎麼執行、怎麼驗證。
> Last Updated: 2026-09-26

## 產品概述

**高中生版**：一門教你「ChatGPT 這類模型是怎麼從零做出來」的課。從最簡單的「數數猜下一個字」開始，一路做出會講故事、會用計算機、能用強化學習變強的小模型，最後把它變成搜尋用的向量模型。每一課先用白話講直覺，再推公式，最後在筆電 CPU 上跑出結果。

**專業版**：語言模型的自學課程（繁體中文），整合 word2vec、GPT-2、Stanford CS336、nanochat、text embedding 五個主題，依「做出一個模型的順序」重排：n-gram 與評估指標 → word2vec → byte-level BPE → Transformer 與 GPT-2（載入 OpenAI 權重驗證）→ 預訓練（Muon、WSD）→ scaling laws → 系統 → 推論 → 資料 → SFT 與工具使用 → RL（GRPO、DPO）→ embedding 模型 → 評估與 nanochat 導讀 → 2026 前沿。
所有演算法在 `src/lm_course/` 有經過測試的參考實作，每個實驗 notebook 直接使用它。

## 目錄結構

```
language-models-course/
├── lessons/          # 講義 00–15 ＋ 附錄 A（繁中 Markdown）
├── notebooks/        # 已執行的實驗 01–13（輸出是實跑紀錄）
├── src/lm_course/
│   ├── ngram.py          # NGramCounts（前文雜湊成 55 位元整數、排序＋二分搜尋）、AddKLM、InterpolatedLM（Witten–Bell）
│   ├── word2vec.py       # Vocab、skipgram_pairs（動態視窗）、SkipGram（SGNS）、PPMI／SVD、sgns_objective、類比
│   ├── tokenizer.py      # BPETokenizer（增量更新的訓練、快取的編碼、特殊 token、GPT-2 詞彙載入）
│   ├── model.py          # GPTConfig／GPT（可切換 norm、位置、MLP、GQA、QK-norm、soft-cap、causal）、KVCache、
│   │                     # attention_reference、read_safetensors、gpt2_from_state_dict、load_gpt2_pretrained
│   ├── sampling.py       # filter_logits、next_token_probs、generate（有／無 cache、同 prompt 批次）、speculative_*
│   ├── optim.py          # newton_schulz_orthogonalize、Muon、CombinedOptimizer、build_optimizer、cosine／wsd 排程
│   ├── training.py       # tokenize_stories、random_batch、lm_loss、evaluate（loss＋bits-per-byte）、train
│   ├── scaling.py        # matmul_parameters、training_flops_per_token、measured_training_flops、記憶體帳、擬合
│   ├── kernels.py        # online_softmax、tiled_attention（FlashAttention 的 forward）、roofline 量測、資料平行模擬
│   ├── quantization.py   # quantize_absmax、fake_quantize、quantize_model、weight_bytes
│   ├── data_pipeline.py  # gopher_failures、exact_duplicates、MinHasher、LSH、near_duplicates、分類器、contaminated
│   ├── chat.py           # SPECIAL_TOKENS、render_conversation、sft_batch、AdditionProblem（三種回答格式）、
│   │                     # 故事請求與主題獎勵（mentions_topic、HELD_OUT_TOPICS）、calculator、ChatEngine
│   ├── rl.py             # completion_logprobs（可指定 temperature）、group_advantages、policy_gradient_loss、
│   │                     # kl_penalty（k3 估計量）、dpo_loss、pass_at_k
│   ├── embeddings.py     # pooling、TextEncoder、info_nce_loss、matryoshka_loss、train_contrastive、retrieval_metrics、BM25
│   ├── artifacts.py      # 課程共用：course_stories、course_tokenizer、course_tokens、base_model_config、load_base_model
│   ├── data.py           # download（檢查長度＋斷點續傳）、TinyStories、GPT-2 檔案、玩具世界語料
│   └── utils.py          # PROJECT_ROOT、DATA_DIR、CHECKPOINT_DIR、get_device、set_seed、notebook_logging
├── scripts/pretrain.py   # Lab 06 的預訓練（命令列版）
├── tests/                # pytest；多數測試拿封閉解或參考輸出對答案
├── references.md         # 參考資料（查證日期、一手／二手）
├── NOTICE.md             # 第三方授權與標示
└── ai-collab/            # 本資料夾
```

被 git 忽略、執行時才產生：`data/`（TinyStories、GPT-2 檔案、token 快取）、`checkpoints/`（tokenizer、base／SFT 模型）、`outputs/`。

## 核心設計

**一個可切換的 GPT**：`GPTConfig` 的開關涵蓋 GPT-2（LayerNorm、learned position、GELU、bias、綁定 embedding）、Llama 類（RMSNorm、RoPE、SwiGLU、無 bias、不綁）、nanochat 類（無參數 RMSNorm、ReLU²、QK-norm、soft-cap）。`n_kv_head` 做 GQA；`causal=False` 變成雙向 encoder（第 13 課）。`n_kv_head`、`d_ff` 為 `None` 時由 `kv_heads`、`ff_dim` 屬性推出（用屬性而非 `__post_init__` 填值，才不會在 `dataclasses.replace` 時帶著舊值出錯）。

**GPT-2 權重相容**：`gpt2_from_state_dict` 把 OpenAI 的 Conv1D（轉置）與 fused `c_attn` 對應到本實作；開發時用 Hugging Face transformers 交叉驗證，logits 最大差 2e-4（logit 量級 266）。測試裡的參考值（token id、前 5 名、最大 logit）來自那次交叉驗證。

**共用資產**（`artifacts.py`）：TinyStories 的 GPT-4 驗證檔依故事 90／10 切分（seed 0）；tokenizer 4,096 個 token（含 9 個 chat 特殊 token，數字單一位數切分）；base model 是 nanochat 類、4 層、寬 256、context 256（5.24M 參數）。

**加法任務**（`chat.AdditionProblem`）：同一題有三種回答格式，由問法決定——直接（`What is a + b?`）、step by step（加 `Think step by step.`，直式逐位計算）、計算機（加 `Use the calculator.`，呼叫 `<|python_start|>`）。這是第 11 課的主任務（格式決定正確率）。

**RL 任務**（`chat.story_request`、`chat.mentions_topic`）：「Tell me a story about a {topic}.」，獎勵 ＝ 回答前 40 個 token 有沒有提到主題字。20 個主題訓練（與 SFT 資料相同）、10 個主題（`HELD_OUT_TOPICS`）只用來評估。加法題的 RL 在這個規模學不起來（見 `dev_log.md` 2026-09-26），Lab 12 §7 把它當成反例保留。

**n-gram 的計數**（`ngram.NGramCounts`）：前文用乘法雜湊壓成 55 位元整數（`(key + byte + 1) * 常數`，取 uint64 乘積的高位元），配對鍵 ＝ 雜湊 × 256 ＋ 下一個位元組，所以任意階數都放得進 int64。Lab 01 的資料上，第 3、5、7 階的雜湊前文數與精確前文數完全相同。

## 執行

```bash
cd language-models-course
uv add torch numpy matplotlib loguru regex ipykernel   # 一次性，由本人執行
uv add --dev pytest
uv run pytest                                         # 需要網路的測試標記為 network，離線時自動略過
```

notebook 用本子專案的 `.venv` 當 kernel。**依賴順序**：Lab 03 存 tokenizer（缺少時其他實驗會自己訓練一個相同的）→ Lab 06 存 `checkpoints/base_model.pt`（Lab 09、11、13 需要）→ Lab 11 存 `checkpoints/sft_model.pt`（Lab 12 需要）。

雲端 4 核心 CPU 的實測執行時間（2026-09-26，整本 notebook 從頭執行到存檔；Lab 04 為 2026-09-25）：

| notebook | 時間 | 備註 |
|---|---|---|
| 01 n-gram | 2 分鐘 | 1–10 階計數約 20 秒 |
| 02 word2vec | 1 分鐘 | |
| 03 BPE | 1 分鐘 | 16,384 詞彙的 tokenizer 佔大部分；第一次下載 GPT-2 詞彙檔 |
| 04 attention | < 1 分鐘 | |
| 05 GPT-2 | 1 分鐘 | 不含第一次下載 548 MB 權重 |
| 06 預訓練 | 40 分鐘 | 主訓練 27.4 分鐘（約 7,500 token/s），§5 三個短訓練約 12 分鐘 |
| 07 scaling | 79 分鐘 | §4 的 21 次訓練；小模型步數多、每步有固定開銷 |
| 08 系統 | < 1 分鐘 | |
| 09 推論 | 4 分鐘 | 草稿模型訓練 2.5 分鐘 |
| 10 資料 | 1 分鐘 | |
| 11 SFT | 18 分鐘 | 兩次 SFT 各約 8 分鐘 |
| 12 RL | 8 分鐘 | 兩次 GRPO 各 2.5 分鐘、DPO 約 1 分鐘、§7 的梯度檢查約 1 分鐘 |
| 13 embedding | 23 分鐘 | 四次對比訓練各約 5.5 分鐘 |

**不要同時跑兩本訓練型 notebook**：兩個 PyTorch 行程搶 4 顆核心會互相拖慢好幾倍（diffusion 課程已踩過，本課程開發時也觀察到預訓練原型在旁邊跑測試時變慢）。

## 測試策略

- **封閉解**：SGNS 的最佳內積 ＝ PMI − log k；unigram 的 bpb ＝ 位元組熵；RoPE 的分數只依相對距離；tiled attention ＝ 參考 attention（含 log-sum-exp）；資料平行的平均梯度 ＝ 全 batch 梯度；speculative sampling 的輸出分布 ＝ 目標分布（2 萬次抽樣）且接受率 ＝ Σmin(p, q)；on-policy 的 policy-gradient loss 的梯度 ＝ REINFORCE。
- **參考輸出**：GPT-2 small／medium 的參數量；GPT-2 tokenizer 的 token id 與 OpenAI tiktoken 相同；載入的 GPT-2 前 5 名 token 與 Hugging Face 相同（後兩者需要網路）。
- **行為**：KV cache（prefill、分段、逐一 decode）與完整 forward 相同；causal 模型不看未來；encoder 模式不受 padding 影響；ChatEngine 用照劇本輸出的假模型驗證工具呼叫的時機。
- `ruff`：本子專案的 `pyproject.toml` 排除 `notebooks/`（與 repo 根的 pre-commit 一致）；repo 的 pre-commit 釘在 ruff 0.14.1。

## 修改 notebook 的注意事項

notebook 的輸出是實跑紀錄，改了程式碼就要**整本重新執行**再存檔，避免輸出與程式不一致；只改 markdown 可以不重跑，但改完要再核對一次文字與輸出是否相符。輸出中不應出現本機絕對路徑。
TinyStories 的故事片段出現在部分 notebook 輸出中，授權標示在 `NOTICE.md`；新增會印出故事原文的輸出時，記得同步更新 `NOTICE.md` 的清單。
