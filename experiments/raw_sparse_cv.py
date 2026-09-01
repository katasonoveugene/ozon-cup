from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_core.common import (
    apply_category_thresholds,
    best_f1_threshold,
    category_f1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--char-weight", type=float, default=0.375)
    parser.add_argument("--word-features", type=int, default=0)
    parser.add_argument("--char-features", type=int, default=300_000)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--majority-positive-weight", type=float)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def raw_value(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def build_raw_text(name: object, description: object) -> str:
    title = raw_value(name)
    body = raw_value(description)
    return f"название {title} название {title} описание {body}"


def main() -> None:
    args = parse_args()
    started = time.time()
    frame = pd.read_csv(args.data)
    texts = [
        build_raw_text(name, description)
        for name, description in zip(frame["name"], frame["description"])
    ]

    word_features = args.word_features or None
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=word_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=2,
        max_features=args.char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    print("fitting raw vectorizers", flush=True)
    word_matrix = word.fit_transform(texts)
    char_matrix = char.fit_transform(texts)
    matrix = hstack(
        [word_matrix, char_matrix * args.char_weight],
        format="csr",
        dtype=np.float32,
    )
    vectorize_seconds = time.time() - started
    print(
        f"matrix={matrix.shape} nnz={matrix.nnz} "
        f"vectorize_seconds={vectorize_seconds:.2f}",
        flush=True,
    )

    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy()
    unique_categories = sorted(np.unique(categories))
    positive_weights = {category: args.positive_weight for category in unique_categories}
    if args.majority_positive_weight is not None:
        for category in unique_categories:
            category_mask = categories == category
            if labels[category_mask].mean() >= 0.5:
                positive_weights[category] = args.majority_positive_weight
    strata = np.char.add(np.char.add(categories, "__"), labels.astype(str))
    splits = list(
        StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed,
        ).split(matrix, strata)
    )

    def fit_partition(fold: int, category: str) -> tuple[int, str, np.ndarray, np.ndarray]:
        train_index, valid_index = splits[fold]
        fit_index = train_index[categories[train_index] == category]
        score_index = valid_index[categories[valid_index] == category]
        positive_weight = positive_weights[category]
        class_weight = None if positive_weight == 1.0 else {0: 1.0, 1: positive_weight}
        model = LinearSVC(
            C=args.c,
            class_weight=class_weight,
            dual="auto",
            max_iter=10_000,
        )
        model.fit(matrix[fit_index], labels[fit_index])
        return fold, category, score_index, model.decision_function(matrix[score_index])

    model_started = time.time()
    partitions = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(fit_partition)(fold, category)
        for fold in range(args.folds)
        for category in unique_categories
    )
    oof = np.zeros(len(frame), dtype=np.float64)
    for _, _, score_index, scores in partitions:
        oof[score_index] = scores
    model_seconds = time.time() - model_started

    for fold, (_, valid_index) in enumerate(splits):
        fold_score, fold_parts = category_f1(
            labels[valid_index], oof[valid_index] >= 0, categories[valid_index]
        )
        print(f"fold={fold} threshold0={fold_score:.6f} parts={fold_parts}", flush=True)

    thresholds = {
        category: best_f1_threshold(
            labels[categories == category], oof[categories == category]
        )[0]
        for category in unique_categories
    }
    predictions = apply_category_thresholds(oof, categories, thresholds)
    tuned_score, tuned_parts = category_f1(labels, predictions, categories)
    zero_score, zero_parts = category_f1(labels, oof >= 0, categories)
    config = {
        "raw_html": True,
        "title_repetitions": 2,
        "word_ngram_range": [1, 2],
        "word_min_df": 2,
        "word_max_features": word_features,
        "char_analyzer": "char_wb",
        "char_ngram_range": [3, 6],
        "char_min_df": 2,
        "char_max_features": args.char_features,
        "char_weight": args.char_weight,
        "c": args.c,
        "positive_weights": positive_weights,
        "folds": args.folds,
        "seed": args.seed,
    }
    result = {
        "rows": len(frame),
        "matrix_shape": matrix.shape,
        "config": config,
        "threshold_zero_score": zero_score,
        "threshold_zero_parts": zero_parts,
        "tuned_score": tuned_score,
        "tuned_parts": tuned_parts,
        "thresholds": thresholds,
        "vectorize_seconds": vectorize_seconds,
        "model_seconds": model_seconds,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            id=frame["id"].to_numpy(),
            label=labels,
            category=categories,
            score=oof,
            prediction=predictions,
            threshold_category=np.asarray(unique_categories),
            threshold_value=np.asarray([thresholds[c] for c in unique_categories]),
            config_json=np.asarray(json.dumps(config, ensure_ascii=False)),
        )


if __name__ == "__main__":
    main()
