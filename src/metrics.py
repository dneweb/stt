"""
Evaluation Metrics for Speech-to-Text:
Character Error Rate (CER) and Word Error Rate (WER).
"""

from typing import List


def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """Computes Levenshtein edit distance between two token lists."""
    r_len, h_len = len(ref), len(hyp)
    if r_len == 0:
        return h_len
    if h_len == 0:
        return r_len

    # Distance matrix rows (previous and current)
    prev_row = list(range(h_len + 1))
    curr_row = [0] * (h_len + 1)

    for i, r_tok in enumerate(ref):
        curr_row[0] = i + 1
        for j, h_tok in enumerate(hyp):
            cost = 0 if r_tok == h_tok else 1
            curr_row[j + 1] = min(
                curr_row[j] + 1,       # Insertion
                prev_row[j + 1] + 1,   # Deletion
                prev_row[j] + cost     # Substitution
            )
        prev_row = list(curr_row)

    return prev_row[h_len]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER) between reference and hypothesis strings."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    dist = levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) between reference and hypothesis strings."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)
