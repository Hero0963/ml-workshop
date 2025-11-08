# src/evaluation/text_evaluator.py
import re
from pathlib import Path
from loguru import logger


def normalize_text(text: str) -> list[str]:
    """
    Cleans and normalizes a text string for comparison.
    - Converts to lowercase
    - Replaces punctuation with spaces
    - Splits into a list of words
    """
    text = text.lower()
    # Replace punctuation with spaces
    cleaned_text = re.sub(r"[^ʭ\w\s]", " ", text)
    # Split into words and remove any empty strings that might result from multiple spaces
    return [word for word in cleaned_text.split() if word]


def calculate_wer(reference: list[str], hypothesis: list[str]) -> float:
    """
    Calculates the Word Error Rate (WER) using Levenshtein distance.
    WER = (Substitutions + Deletions + Insertions) / N
    where N is the number of words in the reference.
    """
    m, n = len(reference), len(hypothesis)

    # Initialize DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    # The Levenshtein distance is the final value in the matrix
    distance = dp[m][n]

    if m == 0:
        return 1.0 if n > 0 else 0.0

    wer = distance / m
    return wer


def evaluate_from_texts(reference_text: str, hypothesis_text: str) -> float:
    """
    Evaluates the Word Error Rate (WER) between a reference text and a hypothesis text.
    """
    ref_words = normalize_text(reference_text)
    hyp_words = normalize_text(hypothesis_text)

    logger.info(f"Reference (ground truth) words: {ref_words}")
    logger.info(f"Hypothesis (transcribed) words: {hyp_words}")

    wer = calculate_wer(ref_words, hyp_words)
    return wer


def _run():
    """
    Runs a hardcoded evaluation for a specific test case.
    """
    # Hardcoded file paths for a specific test run
    reference_file = Path("test_data/this_is_a_simple_test.txt")
    hypothesis_file = Path("output/stt/this_is_a_simple_test_20251108_154016_04.txt")

    logger.info(f"Evaluating hardcoded reference file: {reference_file}")
    logger.info(f"Evaluating hardcoded hypothesis file: {hypothesis_file}")

    ref_text = reference_file.read_text(encoding="utf-8")
    hyp_text = hypothesis_file.read_text(encoding="utf-8")

    wer_score = evaluate_from_texts(ref_text, hyp_text)

    logger.success(f"Word Error Rate (WER): {wer_score:.2%}")
    logger.info("A lower WER is better. 0% means a perfect match.")


if __name__ == "__main__":
    _run()
