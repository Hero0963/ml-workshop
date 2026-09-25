# src/lm_course/__init__.py
"""Reference implementations for the language models course.

Every lab notebook builds on these modules; each one maps to a lesson:

- ``tokenizer``   -> lesson 03 (byte-level BPE, GPT-2's vocabulary)
- ``model``       -> lessons 04-06 (attention, GPT-2, modern variants, KV cache)
- ``sampling``    -> lessons 05 and 09 (temperature / top-k / top-p, speculative decoding)
- ``optim``       -> lesson 06 (AdamW, Muon, learning-rate schedules)
- ``training``    -> lessons 06-07 (pretraining loop, loss, bits-per-byte)
- ``data``        -> downloads (TinyStories, GPT-2) and generated corpora
"""
