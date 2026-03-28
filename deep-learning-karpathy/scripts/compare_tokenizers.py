# scripts/compare_tokenizers.py
"""
Compare different tokenization approaches: character-level, BasicBPE, and tiktoken (GPT-4).

Usage:
    uv run python scripts/compare_tokenizers.py
"""

import os
import sys

# Add project root to path — must happen before local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)

from train_tokenizer import BasicTokenizer  # noqa: E402


def character_level_encode(text: str) -> list[int]:
    """Simple character-level encoding."""
    return list(text.encode("utf-8"))


def main():
    # Test texts in different languages/domains
    test_texts = {
        "English": "The quick brown fox jumps over the lazy dog. This is a simple English sentence with common words.",
        "Chinese": "快速的棕色狐狸跳過了懶惰的狗。這是一個簡單的中文句子。",
        "Code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Numbers": "The population of Earth is approximately 8,045,311,447 as of 2024. Pi is 3.14159265358979.",
        "Mixed": "Hello! 你好！ 42 is the answer. def f(x): return x**2",
    }

    # Train BasicTokenizer on a larger corpus
    text_path = os.path.join(
        project_dir, "references", "minbpe", "tests", "taylorswift.txt"
    )
    if os.path.exists(text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            training_text = f.read()
    else:
        training_text = " ".join(test_texts.values()) * 50

    print("=" * 70)
    print("TOKENIZER COMPARISON")
    print("=" * 70)

    # Train Basic BPE
    print("\nTraining BasicTokenizer (vocab_size=500)...")
    basic_tok = BasicTokenizer()
    basic_tok.train(training_text, vocab_size=500)
    print("Done!")

    # Try to import tiktoken
    try:
        import tiktoken

        gpt4_enc = tiktoken.get_encoding("cl100k_base")
        has_tiktoken = True
        print("tiktoken (GPT-4) loaded successfully!")
    except ImportError:
        has_tiktoken = False
        print("tiktoken not installed, skipping GPT-4 comparison")
        print("Install with: uv add tiktoken")

    # Compare
    print("\n" + "=" * 70)
    print(f"{'Text':<10} {'Bytes':<8} {'Char':<8} {'BPE-500':<10} {'GPT-4':<8}")
    print("-" * 70)

    for name, text in test_texts.items():
        n_bytes = len(text.encode("utf-8"))
        n_char = len(character_level_encode(text))
        n_bpe = len(basic_tok.encode(text))

        if has_tiktoken:
            n_gpt4 = len(gpt4_enc.encode(text))
            print(f"{name:<10} {n_bytes:<8} {n_char:<8} {n_bpe:<10} {n_gpt4:<8}")
        else:
            print(f"{name:<10} {n_bytes:<8} {n_char:<8} {n_bpe:<10} {'N/A':<8}")

    # Compression ratios
    print("\n" + "=" * 70)
    print("COMPRESSION RATIOS (bytes / tokens)")
    print("-" * 70)
    print(f"{'Text':<10} {'Char':<10} {'BPE-500':<12} {'GPT-4':<10}")
    print("-" * 70)

    for name, text in test_texts.items():
        n_bytes = len(text.encode("utf-8"))
        n_char = len(character_level_encode(text))
        n_bpe = len(basic_tok.encode(text))

        ratio_char = n_bytes / n_char
        ratio_bpe = n_bytes / n_bpe

        if has_tiktoken:
            n_gpt4 = len(gpt4_enc.encode(text))
            ratio_gpt4 = n_bytes / n_gpt4
            print(
                f"{name:<10} {ratio_char:<10.2f} {ratio_bpe:<12.2f} {ratio_gpt4:<10.2f}"
            )
        else:
            print(f"{name:<10} {ratio_char:<10.2f} {ratio_bpe:<12.2f} {'N/A':<10}")

    # Show how GPT-4 tokenizes specific examples
    if has_tiktoken:
        print("\n" + "=" * 70)
        print("GPT-4 TOKENIZATION EXAMPLES")
        print("-" * 70)

        examples = [
            "Hello world!",
            "你好世界！",
            "123 + 456 = 579",
            "def hello():",
            "<|endoftext|>",
        ]

        for text in examples:
            ids = gpt4_enc.encode(text, allowed_special="all")
            tokens = [gpt4_enc.decode([id]) for id in ids]
            tokens_repr = [ascii(t) for t in tokens]
            print(f"\n  {ascii(text)}")
            print(f"  -> IDs:    {ids}")
            print(f"  -> Tokens: {tokens_repr}")
            print(f"  -> Count:  {len(ids)}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS:")
    print("  1. Character-level has compression ratio ~= 1.0x (no compression)")
    print("  2. BPE with vocab_size=500 gives moderate compression")
    print("  3. GPT-4 (100K vocab) gives the best compression for English")
    print("  4. Non-English text (Chinese) uses more tokens per character")
    print("=" * 70)


if __name__ == "__main__":
    main()
