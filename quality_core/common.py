from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

import numpy as np
from sklearn.metrics import f1_score


_TAG_RE = re.compile(r"<[^>]*>")
_SPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def clean_text(value: object) -> str:
    """Convert marketplace HTML-ish text into a stable plain-text form."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = _TAG_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def normalize_exact(value: object) -> str:
    """Aggressive normalization used only for exact duplicate matching."""
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold().replace("ё", "е")
    return _NON_WORD_RE.sub("", text)


def normalize_tokens(value: object) -> str:
    """Normalize punctuation while preserving word boundaries."""
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold().replace("ё", "е")
    return _SPACE_RE.sub(" ", _NON_WORD_RE.sub(" ", text)).strip()


def build_sparse_text(name: object, description: object) -> str:
    """Give the short, high-signal title extra weight without dropping the body."""
    title = clean_text(name)
    body = clean_text(description)
    return f"название {title} название {title} описание {body}"


def category_f1(y_true: Iterable[int], y_pred: Iterable[int], categories: Iterable[str]) -> tuple[float, dict[str, float]]:
    """Mean positive-class F1 across business categories."""
    y_true_array = np.asarray(y_true, dtype=np.int8)
    y_pred_array = np.asarray(y_pred, dtype=np.int8)
    category_array = np.asarray(categories)
    per_category: dict[str, float] = {}
    for category in sorted(np.unique(category_array)):
        mask = category_array == category
        per_category[str(category)] = float(f1_score(y_true_array[mask], y_pred_array[mask]))
    return float(np.mean(list(per_category.values()))), per_category


def best_f1_threshold(y_true: Iterable[int], scores: Iterable[float]) -> tuple[float, float]:
    """Find the exact score cutoff maximizing positive-class F1."""
    y = np.asarray(y_true, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    sorted_y = y[order]
    sorted_scores = values[order]
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    positives = int(sorted_y.sum())
    denominator = positives + tp + fp
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    valid_end = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    candidate_indices = np.flatnonzero(valid_end)
    best_index = int(candidate_indices[np.argmax(f1[candidate_indices])])
    if best_index == len(sorted_scores) - 1:
        threshold = float(np.nextafter(sorted_scores[best_index], -np.inf))
    else:
        threshold = float((sorted_scores[best_index] + sorted_scores[best_index + 1]) / 2)
    return threshold, float(f1[best_index])


def apply_category_thresholds(
    scores: Iterable[float], categories: Iterable[str], thresholds: dict[str, float]
) -> np.ndarray:
    score_array = np.asarray(scores, dtype=np.float64)
    category_array = np.asarray(categories)
    predictions = np.zeros(len(score_array), dtype=np.int8)
    for category, threshold in thresholds.items():
        mask = category_array == category
        predictions[mask] = score_array[mask] >= threshold
    return predictions
