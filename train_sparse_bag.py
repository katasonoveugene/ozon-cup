from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from quality_core.common import apply_category_thresholds, best_f1_threshold, category_f1, clean_text
from quality_core.rule_features import apply_conservative_overrides
from quality_core.sparse_model import SparseBagModel, build_raw_text


DEFAULT_SEEDS = (42, 3407, 1337, 2025, 20260829, 20260701, 777)
SUPPLEMENT_CATEGORY = "БАД"
FLAMMABLE_CATEGORY = "Легковоспламеняющиеся"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/sparse_bag.joblib"))
    parser.add_argument("--oof-output", type=Path, default=Path("outputs/sparse_bag_oof.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/sparse_bag_report.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--char-weight", type=float, default=0.375)
    parser.add_argument("--char-features", type=int, default=300_000)
    parser.add_argument("--compression", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    frame = pd.read_csv(args.data)
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    category_values = sorted(np.unique(categories))
    strata = np.char.add(np.char.add(categories, "__"), labels.astype(str))
    texts = [
        build_raw_text(name, description)
        for name, description in zip(frame["name"], frame["description"])
    ]

    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=None,
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
    word_matrix = word.fit_transform(texts)
    char_matrix = char.fit_transform(texts)
    matrix = hstack(
        [word_matrix, char_matrix * args.char_weight], format="csr", dtype=np.float32
    )
    del word_matrix, char_matrix
    print(f"matrix={matrix.shape} nnz={matrix.nnz}", flush=True)

    category_options: dict[str, dict[str, object]] = {
        SUPPLEMENT_CATEGORY: {"C": 1.0, "class_weight": {0: 1.0, 1: 2.0}},
        FLAMMABLE_CATEGORY: {"C": 1.0, "class_weight": None},
    }
    splits_by_seed = {
        seed: list(
            StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed).split(
                matrix, strata
            )
        )
        for seed in seeds
    }

    def fit_partition(seed: int, fold: int, category: str):
        train_index, valid_index = splits_by_seed[seed][fold]
        fit_index = train_index[categories[train_index] == category]
        score_index = valid_index[categories[valid_index] == category]
        options = category_options[category]
        model = LinearSVC(
            C=float(options["C"]),
            class_weight=options["class_weight"],
            dual="auto",
            max_iter=10_000,
            random_state=seed + fold,
        )
        model.fit(matrix[fit_index], labels[fit_index])
        score = model.decision_function(matrix[score_index]).astype(np.float32)
        return seed, fold, category, score_index, score, model

    fitted = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(fit_partition)(seed, fold, category)
        for seed in seeds
        for fold in range(args.folds)
        for category in category_values
    )
    seed_position = {seed: position for position, seed in enumerate(seeds)}
    seed_oof = np.zeros((len(frame), len(seeds)), dtype=np.float32)
    models: dict[str, list[list[LinearSVC]]] = {
        category: [[None for _ in range(args.folds)] for _ in seeds]  # type: ignore[list-item]
        for category in category_values
    }
    for seed, fold, category, score_index, score, model in fitted:
        position = seed_position[seed]
        seed_oof[score_index, position] = score
        models[category][position][fold] = model

    ensemble_oof = np.median(seed_oof, axis=1)
    thresholds = {
        category: best_f1_threshold(
            labels[categories == category], ensemble_oof[categories == category]
        )[0]
        for category in category_values
    }
    prediction = apply_category_thresholds(ensemble_oof, categories, thresholds)
    overridden = apply_conservative_overrides(frame.fillna(""), prediction)
    base_score, base_parts = category_f1(labels, prediction, categories)
    override_score, override_parts = category_f1(labels, overridden, categories)
    per_seed: dict[str, object] = {}
    for position, seed in enumerate(seeds):
        local_thresholds = {
            category: best_f1_threshold(
                labels[categories == category], seed_oof[categories == category, position]
            )[0]
            for category in category_values
        }
        local_prediction = apply_category_thresholds(seed_oof[:, position], categories, local_thresholds)
        local_override = apply_conservative_overrides(frame.fillna(""), local_prediction)
        per_seed[str(seed)] = {
            "base": category_f1(labels, local_prediction, categories)[0],
            "override": category_f1(labels, local_override, categories)[0],
            "thresholds": local_thresholds,
        }

    config = {
        "seeds": seeds,
        "folds": args.folds,
        "aggregation": "mean folds within seed, median across seeds",
        "title_repetitions": 2,
        "word_ngram_range": [1, 2],
        "word_min_df": 2,
        "word_max_features": None,
        "char_analyzer": "char_wb",
        "char_ngram_range": [3, 6],
        "char_min_df": 2,
        "char_max_features": args.char_features,
        "char_weight": args.char_weight,
        "category_options": category_options,
    }
    description_groups: dict[tuple[str, str], list[int]] = {}
    for category, description, label in zip(categories, frame["description"], labels):
        key = clean_text(description).casefold()
        if key:
            description_groups.setdefault((category, key), []).append(int(label))
    description_labels = {
        key: values[0]
        for key, values in description_groups.items()
        if min(values) == max(values)
    }
    artifact = SparseBagModel(
        word_vectorizer=word,
        char_vectorizer=char,
        char_weight=args.char_weight,
        models=models,
        thresholds=thresholds,
        description_labels=description_labels,
        seeds=seeds,
        config=config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=args.compression)
    report = {
        "rows": len(frame),
        "matrix_shape": list(matrix.shape),
        "thresholds": thresholds,
        "base_score": base_score,
        "base_parts": base_parts,
        "override_score": override_score,
        "override_parts": override_parts,
        "per_seed": per_seed,
        "artifact_bytes": args.output.stat().st_size,
        "pure_description_keys": len(description_labels),
        "elapsed_seconds": time.time() - started,
        "config": config,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.oof_output,
        id=frame["id"].to_numpy(),
        label=labels,
        category=categories,
        seed_score=seed_oof,
        score=ensemble_oof,
        prediction=prediction,
        override_prediction=overridden,
        threshold_category=np.asarray(category_values),
        threshold_value=np.asarray([thresholds[category] for category in category_values]),
        config_json=np.asarray(json.dumps(config, ensure_ascii=False)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
