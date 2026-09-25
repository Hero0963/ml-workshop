# src/lm_course/__init__.py
"""Reference implementations for the language models course.

Every lab notebook builds on these modules; each one maps to a lesson:

- ``ngram``          -> lesson 01 (count-based language models, bits per byte)
- ``word2vec``       -> lesson 02 (skip-gram with negative sampling, PPMI + SVD)
- ``tokenizer``      -> lesson 03 (byte-level BPE, GPT-2's vocabulary)
- ``model``          -> lessons 04-06 (attention, GPT-2, modern variants, KV cache)
- ``sampling``       -> lessons 05 and 09 (temperature / top-k / top-p, speculative decoding)
- ``optim``          -> lesson 06 (AdamW, Muon, learning-rate schedules)
- ``training``       -> lessons 06-07 (pretraining loop, loss, bits per byte)
- ``scaling``        -> lesson 07 (FLOPs and memory accounting, scaling-law fits)
- ``kernels``        -> lesson 08 (online softmax, tiled attention, roofline, data parallel)
- ``quantization``   -> lesson 09 (weight-only int8 / int4)
- ``data_pipeline``  -> lesson 10 (quality rules, MinHash dedup, classifiers, contamination)
- ``chat``           -> lesson 11 (conversation format, SFT masks, tool use)
- ``rl``             -> lesson 12 (policy gradient / GRPO, DPO, pass@k)
- ``embeddings``     -> lesson 13 (pooling, InfoNCE, Matryoshka, retrieval metrics, BM25)
- ``data``           -> downloads (TinyStories, GPT-2) and generated corpora
"""
