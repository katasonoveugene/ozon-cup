from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC, SVC

from quality_core.dense_model import normalize_rows
from quality_core.validated_blend import (
    COMMON,
    COMPONENTS,
    RARE,
    ValidatedBlendModel,
    current_text,
    mean2_margin,
)


WEIGHTS = {
    "sparse": 0.25,
    "separated": 0.25,
    "mixed_linear": 1.0 / 6.0,
    "mixed_rbf": 1.0 / 6.0,
    "mixed_mean2": 1.0 / 6.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/mixed.npy"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/source_template_stress_manifest_v2.npz"),
    )
    parser.add_argument("--fold-key", default="fold__seed_15485863")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validated_blend/validated_blend.joblib"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/validated_blend/training_report.json"),
    )
    parser.add_argument("--retrieval-chunk-size", type=int, default=192)
    return parser.parse_args()


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order].astype(np.int64)
    true_positive = np.cumsum(ordered_labels)
    predicted_positive = np.arange(1, len(labels) + 1, dtype=np.int64)
    values = 2.0 * true_positive / (predicted_positive + int(labels.sum()))
    distinct = np.r_[scores[order][:-1] != scores[order][1:], True]
    positions = np.flatnonzero(distinct)
    selected = int(positions[np.argmax(values[positions])])
    return float(scores[order][selected]), float(values[selected])


def make_vectorizers(frame: pd.DataFrame) -> tuple[Any, ...]:
    texts = current_text(frame)
    title = frame["name"].fillna("").astype(str).to_numpy(dtype=object)
    description = frame["description"].fillna("").astype(str).to_numpy(dtype=object)
    current_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    current_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    separated = (
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=2,
            max_features=160_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float32,
        ),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=2,
            max_features=260_000,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    )
    current_matrix = __import__("scipy").sparse.hstack(
        [
            current_word.fit_transform(texts),
            current_char.fit_transform(texts) * np.float32(0.375),
        ],
        format="csr",
        dtype=np.float32,
    )
    fields = (title, title, description, description)
    scales = (0.5, 0.1875, 1.0, 0.375)
    separated_matrix = __import__("scipy").sparse.hstack(
        [
            vectorizer.fit_transform(field) * np.float32(scale)
            for vectorizer, field, scale in zip(separated, fields, scales)
        ],
        format="csr",
        dtype=np.float32,
    )
    return current_word, current_char, separated, current_matrix, separated_matrix


def main() -> None:
    args = parse_args()
    started = time.time()
    frame = pd.read_csv(args.data).fillna("")
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    vectors = normalize_rows(np.load(args.embeddings, allow_pickle=False))
    with np.load(args.manifest, allow_pickle=False) as manifest:
        if not np.array_equal(
            manifest["id"].astype(np.int64), frame["id"].to_numpy(dtype=np.int64)
        ):
            raise RuntimeError("manifest IDs differ from training data")
        if args.fold_key not in manifest.files:
            raise RuntimeError(f"missing deployment fold key: {args.fold_key}")
        folds = manifest[args.fold_key].astype(np.int8)
        protocol = str(manifest["protocol_version"].item())
    fold_values = sorted(int(value) for value in np.unique(folds))
    if fold_values != [0, 1, 2, 3, 4]:
        raise RuntimeError("deployment bag requires five folds")

    current_word, current_char, separated_vectorizers, current_matrix, separated_matrix = (
        make_vectorizers(frame)
    )
    oof = {name: np.zeros(len(frame), dtype=np.float32) for name in COMPONENTS}
    text_models = {
        category: {"sparse": [], "separated": []} for category in (COMMON, RARE)
    }
    dense_models = {
        category: {"mixed_linear": [], "mixed_rbf": []}
        for category in (COMMON, RARE)
    }
    fold_reports: list[dict[str, Any]] = []
    for fold in fold_values:
        fold_started = time.time()
        for category in (COMMON, RARE):
            fit = np.flatnonzero((folds != fold) & (categories == category))
            valid = np.flatnonzero((folds == fold) & (categories == category))
            sparse_model = LinearSVC(C=3.0, dual="auto", max_iter=20_000, random_state=fold)
            sparse_model.fit(current_matrix[fit], labels[fit])
            oof["sparse"][valid] = sparse_model.decision_function(
                current_matrix[valid]
            ).astype(np.float32)
            text_models[category]["sparse"].append(sparse_model)

            separated_model = LinearSVC(
                C=1.0, dual="auto", max_iter=20_000, random_state=fold + 17
            )
            separated_model.fit(separated_matrix[fit], labels[fit])
            oof["separated"][valid] = separated_model.decision_function(
                separated_matrix[valid]
            ).astype(np.float32)
            text_models[category]["separated"].append(separated_model)

            linear_model = LinearSVC(
                C=1.0,
                class_weight="balanced",
                dual="auto",
                max_iter=20_000,
                random_state=fold + 31,
            )
            linear_model.fit(vectors[fit], labels[fit])
            oof["mixed_linear"][valid] = linear_model.decision_function(
                vectors[valid]
            ).astype(np.float32)
            dense_models[category]["mixed_linear"].append(linear_model)

            rbf_model = SVC(
                C=10.0 if category == COMMON else 3.0,
                gamma=1.5,
                class_weight=None if category == COMMON else "balanced",
                cache_size=2048,
            )
            rbf_model.fit(vectors[fit], labels[fit])
            oof["mixed_rbf"][valid] = rbf_model.decision_function(
                vectors[valid]
            ).astype(np.float32)
            dense_models[category]["mixed_rbf"].append(rbf_model)

            oof["mixed_mean2"][valid] = mean2_margin(
                vectors[valid], vectors[fit], labels[fit], args.retrieval_chunk_size
            )
        fold_reports.append(
            {
                "fold": fold,
                "rows": int(np.sum(folds == fold)),
                "elapsed_seconds": time.time() - fold_started,
            }
        )
        print(json.dumps(fold_reports[-1], ensure_ascii=False), flush=True)

    calibration: dict[str, dict[str, Any]] = {}
    prediction = np.zeros(len(frame), dtype=np.int8)
    blended = np.zeros(len(frame), dtype=np.float64)
    for category in (COMMON, RARE):
        rows = categories == category
        centers = {name: float(np.mean(oof[name][rows])) for name in COMPONENTS}
        scales = {
            name: float(np.std(oof[name][rows]) + 1e-12) for name in COMPONENTS
        }
        score = sum(
            WEIGHTS[name] * (oof[name][rows] - centers[name]) / scales[name]
            for name in COMPONENTS
        )
        threshold, local_f1 = best_threshold(labels[rows], score)
        blended[rows] = score
        prediction[rows] = score >= threshold
        calibration[category] = {
            "centers": centers,
            "scales": scales,
            "weights": WEIGHTS,
            "threshold": threshold,
            "source_group_oof_f1": local_f1,
        }

    model = ValidatedBlendModel(
        current_word=current_word,
        current_char=current_char,
        separated_vectorizers=separated_vectorizers,
        text_models=text_models,
        dense_models=dense_models,
        reference_vectors=vectors.astype(np.float16),
        reference_labels=labels,
        reference_categories=categories,
        reference_folds=folds,
        calibration=calibration,
        config={
            "protocol": "source-group-crossfit-fixed-five-component-blend-v1",
            "manifest_protocol": protocol,
            "fold_key": args.fold_key,
            "folds": 5,
            "aggregation": "mean of five source-group fold models",
            "retrieval": "mean of fold-local top-two class margins",
            "retrieval_chunk_size": args.retrieval_chunk_size,
            "policy_or_posthoc_rules": False,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output, compress=3)
    parts = {
        category: float(f1_score(labels[categories == category], prediction[categories == category]))
        for category in (COMMON, RARE)
    }
    report = {
        "status": "complete_source_group_crossfit_production_fit",
        "protocol": model.config["protocol"],
        "rows": len(frame),
        "data_sha256": sha256(args.data),
        "embedding_sha256": sha256(args.embeddings),
        "manifest_sha256": sha256(args.manifest),
        "fold_key": args.fold_key,
        "weights": WEIGHTS,
        "calibration": calibration,
        "source_group_oof_macro_f1": float(np.mean(list(parts.values()))),
        "source_group_oof_category_f1": parts,
        "positive_predictions": {
            category: int(prediction[categories == category].sum())
            for category in (COMMON, RARE)
        },
        "artifact": str(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "artifact_sha256": sha256(args.output),
        "fold_reports": fold_reports,
        "elapsed_seconds": time.time() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
