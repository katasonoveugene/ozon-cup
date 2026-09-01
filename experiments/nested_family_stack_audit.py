from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC, SVC

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality_core.common import category_f1
from quality_core.dense_model import normalize_rows


RARE = "Легковоспламеняющиеся"
COMMON = "БАД"
BASE_NAMES = (
    "sparse",
    "separated",
    "mixed_linear",
    "mixed_rbf",
    "mixed_mean2",
    "mixed_knn5",
    "image_margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict nested stacking on label-blind product-family folds. All base "
            "models, meta-models, and thresholds exclude the target outer fold."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/family_manifest.npz")
    )
    parser.add_argument("--mixed", type=Path, default=Path("outputs/mixed.npy"))
    parser.add_argument("--image", type=Path, default=Path("outputs/image.npy"))
    parser.add_argument("--scheme", default="family_close")
    parser.add_argument(
        "--group-key",
        help=(
            "Explicit manifest array containing indivisible groups. When omitted, "
            "the legacy group__<scheme> convention is used."
        ),
    )
    parser.add_argument(
        "--fold-key",
        help=(
            "Explicit manifest array containing outer fold assignments. When "
            "omitted, fold__<scheme> is used."
        ),
    )
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument(
        "--merge-partials",
        action="store_true",
        help="Merge five completed per-outer-fold archives without refitting.",
    )
    parser.add_argument("--char-features", type=int, default=300_000)
    parser.add_argument("--title-char-features", type=int, default=160_000)
    parser.add_argument("--description-char-features", type=int, default=260_000)
    parser.add_argument("--neighbor-chunk", type=int, default=256)
    parser.add_argument("--kernel-cache-mib", type=float, default=2_048.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def raw(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def current_text(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            f"название {raw(name)} название {raw(name)} описание {raw(description)}"
            for name, description in zip(frame["name"], frame["description"])
        ],
        dtype=object,
    )


def make_current_matrix(
    texts: np.ndarray,
    fit_index: np.ndarray,
    score_index: np.ndarray,
    char_features: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 6),
        min_df=2,
        max_features=char_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    fit_word = word.fit_transform(texts[fit_index])
    score_word = word.transform(texts[score_index])
    fit_char = char.fit_transform(texts[fit_index])
    score_char = char.transform(texts[score_index])
    fit_matrix = sparse.hstack(
        [fit_word, fit_char * np.float32(0.375)], format="csr", dtype=np.float32
    )
    score_matrix = sparse.hstack(
        [score_word, score_char * np.float32(0.375)],
        format="csr",
        dtype=np.float32,
    )
    return fit_matrix, score_matrix


def make_separated_matrix(
    title: np.ndarray,
    description: np.ndarray,
    fit_index: np.ndarray,
    score_index: np.ndarray,
    title_char_features: int,
    description_char_features: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    vectorizers = (
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
            max_features=title_char_features,
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
            max_features=description_char_features,
            sublinear_tf=True,
            dtype=np.float32,
        ),
    )
    fields = (title, title, description, description)
    scales = (0.5, 0.1875, 1.0, 0.375)
    fit_blocks: list[sparse.csr_matrix] = []
    score_blocks: list[sparse.csr_matrix] = []
    for vectorizer, field, scale in zip(vectorizers, fields, scales):
        fit_blocks.append(vectorizer.fit_transform(field[fit_index]) * scale)
        score_blocks.append(vectorizer.transform(field[score_index]) * scale)
    return (
        sparse.hstack(fit_blocks, format="csr", dtype=np.float32),
        sparse.hstack(score_blocks, format="csr", dtype=np.float32),
    )


def fit_sparse_score(
    fit_matrix: sparse.csr_matrix,
    score_matrix: sparse.csr_matrix,
    fit_index: np.ndarray,
    score_index: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    c_value: float,
    seed: int,
) -> np.ndarray:
    result = np.zeros(len(score_index), dtype=np.float32)
    for category in (COMMON, RARE):
        local_fit = np.flatnonzero(categories[fit_index] == category)
        local_score = np.flatnonzero(categories[score_index] == category)
        model = LinearSVC(
            C=c_value,
            dual="auto",
            max_iter=20_000,
            random_state=seed,
        )
        model.fit(fit_matrix[local_fit], labels[fit_index][local_fit])
        result[local_score] = model.decision_function(
            score_matrix[local_score]
        ).astype(np.float32)
    return result


def neighbor_scores(
    fit_vectors: np.ndarray,
    fit_labels: np.ndarray,
    score_vectors: np.ndarray,
    chunk_size: int,
    *,
    image_only: bool,
) -> dict[str, np.ndarray]:
    output = {
        "margin": np.zeros(len(score_vectors), dtype=np.float32),
    }
    if not image_only:
        output["mean2"] = np.zeros(len(score_vectors), dtype=np.float32)
        output["knn5"] = np.zeros(len(score_vectors), dtype=np.float32)
    positive = fit_labels == 1
    negative = ~positive
    for start in range(0, len(score_vectors), chunk_size):
        end = min(start + chunk_size, len(score_vectors))
        similarity = score_vectors[start:end] @ fit_vectors.T
        positive_similarity = similarity[:, positive]
        negative_similarity = similarity[:, negative]
        output["margin"][start:end] = np.max(
            positive_similarity, axis=1
        ) - np.max(negative_similarity, axis=1)
        if image_only:
            continue
        positive_top2 = np.partition(positive_similarity, -2, axis=1)[:, -2:]
        negative_top2 = np.partition(negative_similarity, -2, axis=1)[:, -2:]
        output["mean2"][start:end] = np.mean(
            positive_top2, axis=1
        ) - np.mean(negative_top2, axis=1)
        available = min(5, len(fit_labels))
        chosen = np.argpartition(similarity, -available, axis=1)[:, -available:]
        chosen_similarity = np.take_along_axis(similarity, chosen, axis=1)
        order = np.argsort(-chosen_similarity, axis=1)
        chosen = np.take_along_axis(chosen, order, axis=1)
        chosen_similarity = np.take_along_axis(chosen_similarity, order, axis=1)
        chosen_labels = fit_labels[chosen]
        weights = np.exp(
            np.clip(
                (chosen_similarity - chosen_similarity[:, :1]) / 0.04,
                -30.0,
                0.0,
            )
        )
        output["knn5"][start:end] = np.sum(
            weights * chosen_labels, axis=1
        ) / np.sum(weights, axis=1)
    return output


def fit_dense_scores(
    fit_index: np.ndarray,
    score_index: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    mixed: np.ndarray,
    image: np.ndarray,
    seed: int,
    neighbor_chunk: int,
    kernel_cache_mib: float,
) -> dict[str, np.ndarray]:
    output = {
        name: np.zeros(len(score_index), dtype=np.float32)
        for name in (
            "mixed_linear",
            "mixed_rbf",
            "mixed_mean2",
            "mixed_knn5",
            "image_margin",
        )
    }
    for category in (COMMON, RARE):
        fit_position = np.flatnonzero(categories[fit_index] == category)
        score_position = np.flatnonzero(categories[score_index] == category)
        local_fit = fit_index[fit_position]
        local_score = score_index[score_position]
        local_labels = labels[local_fit]

        linear = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=20_000,
            random_state=seed,
        )
        linear.fit(mixed[local_fit], local_labels)
        output["mixed_linear"][score_position] = linear.decision_function(
            mixed[local_score]
        ).astype(np.float32)

        if category == RARE:
            kernel = SVC(
                C=3.0,
                gamma=1.5,
                class_weight="balanced",
                cache_size=kernel_cache_mib,
            )
        else:
            kernel = SVC(
                C=10.0,
                gamma=1.5,
                class_weight=None,
                cache_size=kernel_cache_mib,
            )
        kernel.fit(mixed[local_fit], local_labels)
        output["mixed_rbf"][score_position] = kernel.decision_function(
            mixed[local_score]
        ).astype(np.float32)

        mixed_neighbors = neighbor_scores(
            mixed[local_fit],
            local_labels,
            mixed[local_score],
            neighbor_chunk,
            image_only=False,
        )
        output["mixed_mean2"][score_position] = mixed_neighbors["mean2"]
        output["mixed_knn5"][score_position] = mixed_neighbors["knn5"]
        image_neighbors = neighbor_scores(
            image[local_fit],
            local_labels,
            image[local_score],
            neighbor_chunk,
            image_only=True,
        )
        output["image_margin"][score_position] = image_neighbors["margin"]
    return output


def fit_base_scores(
    frame: pd.DataFrame,
    fit_index: np.ndarray,
    score_index: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    texts: np.ndarray,
    title: np.ndarray,
    description: np.ndarray,
    mixed: np.ndarray,
    image: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, np.ndarray]:
    started = time.time()
    current_fit, current_score = make_current_matrix(
        texts, fit_index, score_index, args.char_features
    )
    sparse_score = fit_sparse_score(
        current_fit,
        current_score,
        fit_index,
        score_index,
        labels,
        categories,
        c_value=3.0,
        seed=seed,
    )
    del current_fit, current_score
    gc.collect()

    separated_fit, separated_score = make_separated_matrix(
        title,
        description,
        fit_index,
        score_index,
        args.title_char_features,
        args.description_char_features,
    )
    separated_values = fit_sparse_score(
        separated_fit,
        separated_score,
        fit_index,
        score_index,
        labels,
        categories,
        c_value=1.0,
        seed=seed + 17,
    )
    del separated_fit, separated_score
    gc.collect()

    dense = fit_dense_scores(
        fit_index,
        score_index,
        labels,
        categories,
        mixed,
        image,
        seed + 31,
        args.neighbor_chunk,
        args.kernel_cache_mib,
    )
    print(
        f"base_fit train={len(fit_index)} score={len(score_index)} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return {"sparse": sparse_score, "separated": separated_values, **dense}


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.int64)
    true_positive = np.cumsum(sorted_labels)
    predicted_positive = np.arange(1, len(labels) + 1, dtype=np.int64)
    denominator = predicted_positive + int(np.sum(labels))
    values = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros(len(labels), dtype=np.float64),
        where=denominator > 0,
    )
    distinct = np.r_[scores[order][:-1] != scores[order][1:], True]
    eligible = np.flatnonzero(distinct)
    position = int(eligible[np.argmax(values[eligible])])
    threshold = float(scores[order][position])
    return threshold, float(values[position])


def fixed_score_library(
    train_base: dict[str, np.ndarray],
    target_base: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    names = list(BASE_NAMES)
    train_matrix = np.column_stack([train_base[name] for name in names]).astype(
        np.float64
    )
    target_matrix = np.column_stack([target_base[name] for name in names]).astype(
        np.float64
    )
    center = np.mean(train_matrix, axis=0)
    scale = np.std(train_matrix, axis=0) + 1e-12
    train_matrix = np.clip((train_matrix - center) / scale, -8.0, 8.0)
    target_matrix = np.clip((target_matrix - center) / scale, -8.0, 8.0)
    by_name_train = {name: train_matrix[:, i] for i, name in enumerate(names)}
    by_name_target = {name: target_matrix[:, i] for i, name in enumerate(names)}
    specs: dict[str, dict[str, float]] = {
        name: {name: 1.0} for name in names
    }
    specs.update(
        {
            "blend_text": {"sparse": 0.5, "separated": 0.5},
            "blend_sep_mean2": {"separated": 0.5, "mixed_mean2": 0.5},
            "blend_sparse_rbf": {"sparse": 0.5, "mixed_rbf": 0.5},
            "blend_text_dense": {
                "sparse": 0.25,
                "separated": 0.25,
                "mixed_linear": 1.0 / 6.0,
                "mixed_rbf": 1.0 / 6.0,
                "mixed_mean2": 1.0 / 6.0,
            },
            "blend_text_retrieval": {
                "sparse": 0.25,
                "separated": 0.25,
                "mixed_mean2": 0.25,
                "mixed_knn5": 0.25,
            },
            "blend_sparse_rbf_image": {
                "sparse": 0.50,
                "mixed_rbf": 0.25,
                "image_margin": 0.25,
            },
            "blend_all": {name: 1.0 / len(names) for name in names},
        }
    )
    train_scores: dict[str, np.ndarray] = {}
    target_scores: dict[str, np.ndarray] = {}
    for spec_name, weights in specs.items():
        train_scores[spec_name] = sum(
            weight * by_name_train[name] for name, weight in weights.items()
        )
        target_scores[spec_name] = sum(
            weight * by_name_target[name] for name, weight in weights.items()
        )
    return train_scores, target_scores, specs


def add_fold_bagged_meta(
    train_scores: dict[str, np.ndarray],
    target_scores: dict[str, np.ndarray],
    labels: np.ndarray,
    inner_folds: np.ndarray,
) -> None:
    base_names = list(BASE_NAMES)
    train_matrix = np.column_stack([train_scores[name] for name in base_names])
    target_matrix = np.column_stack([target_scores[name] for name in base_names])
    for c_name, c_value in (("003", 0.03), ("01", 0.1), ("03", 0.3), ("1", 1.0)):
        for weight_name, class_weight in (("plain", None), ("bal", "balanced")):
            name = f"meta_lr_c{c_name}_{weight_name}"
            meta_oof = np.zeros(len(labels), dtype=np.float64)
            target_members: list[np.ndarray] = []
            for fold in sorted(np.unique(inner_folds)):
                fit = inner_folds != fold
                valid = inner_folds == fold
                model = LogisticRegression(
                    C=c_value,
                    class_weight=class_weight,
                    max_iter=5_000,
                    random_state=731 + int(fold),
                )
                model.fit(train_matrix[fit], labels[fit])
                meta_oof[valid] = model.decision_function(train_matrix[valid])
                target_members.append(model.decision_function(target_matrix))
            train_scores[name] = meta_oof
            target_scores[name] = np.mean(target_members, axis=0)


def calibrate_candidate(
    labels: np.ndarray,
    train_score: np.ndarray,
    target_score: np.ndarray,
    inner_folds: np.ndarray,
) -> dict[str, object]:
    threshold, train_f1 = best_threshold(labels, train_score)
    fold_thresholds: list[float] = []
    crossfit_prediction = np.zeros(len(labels), dtype=np.int8)
    fold_f1: list[float] = []
    for fold in sorted(np.unique(inner_folds)):
        tune = inner_folds != fold
        valid = inner_folds == fold
        local_threshold, _ = best_threshold(labels[tune], train_score[tune])
        fold_thresholds.append(local_threshold)
        crossfit_prediction[valid] = train_score[valid] >= local_threshold
        fold_f1.append(
            float(f1_score(labels[valid], crossfit_prediction[valid], zero_division=0))
        )
    median_threshold = float(np.median(fold_thresholds))
    return {
        "global_threshold": threshold,
        "median_threshold": median_threshold,
        "train_f1": train_f1,
        "crossfit_f1": float(
            f1_score(labels, crossfit_prediction, zero_division=0)
        ),
        "crossfit_fold_f1": fold_f1,
        "crossfit_fold_min": float(np.min(fold_f1)),
        "global_prediction": (target_score >= threshold).astype(np.int8),
        "median_prediction": (target_score >= median_threshold).astype(np.int8),
    }


def validate_inner_groups(groups: np.ndarray, folds: np.ndarray) -> None:
    observed: dict[int, int] = {}
    for group, fold in zip(groups.tolist(), folds.tolist()):
        previous = observed.setdefault(int(group), int(fold))
        if previous != int(fold):
            raise AssertionError("an inner family crosses folds")


def evaluate_outer_fold(
    outer_fold: int,
    frame: pd.DataFrame,
    labels: np.ndarray,
    categories: np.ndarray,
    groups: np.ndarray,
    outer_folds: np.ndarray,
    mixed: np.ndarray,
    image: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    started = time.time()
    outer_train = np.flatnonzero(outer_folds != outer_fold)
    outer_valid = np.flatnonzero(outer_folds == outer_fold)
    if np.intersect1d(groups[outer_train], groups[outer_valid]).size:
        raise AssertionError("an outer family crosses train and validation")

    title = np.asarray([raw(value) for value in frame["name"]], dtype=object)
    description = np.asarray(
        [raw(value) for value in frame["description"]], dtype=object
    )
    texts = current_text(frame)
    strata = np.char.add(
        np.char.add(categories[outer_train].astype(str), "__"),
        labels[outer_train].astype(str),
    )
    splitter = StratifiedGroupKFold(
        n_splits=args.inner_folds,
        shuffle=True,
        random_state=91_003 + 101 * outer_fold,
    )
    inner_folds = np.full(len(outer_train), -1, dtype=np.int8)
    split_positions = list(
        splitter.split(
            np.zeros(len(outer_train)),
            strata,
            groups=groups[outer_train],
        )
    )
    for inner_fold, (_, valid_position) in enumerate(split_positions):
        inner_folds[valid_position] = inner_fold
    if np.any(inner_folds < 0):
        raise AssertionError("inner fold assignment is incomplete")
    validate_inner_groups(groups[outer_train], inner_folds)

    inner_base = {
        name: np.zeros(len(outer_train), dtype=np.float32) for name in BASE_NAMES
    }
    for inner_fold, (fit_position, valid_position) in enumerate(split_positions):
        values = fit_base_scores(
            frame,
            outer_train[fit_position],
            outer_train[valid_position],
            labels,
            categories,
            texts,
            title,
            description,
            mixed,
            image,
            args,
            seed=202_000 + 1_003 * outer_fold + inner_fold,
        )
        for name, score in values.items():
            inner_base[name][valid_position] = score
        print(
            f"outer={outer_fold} inner={inner_fold} complete "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    outer_base = fit_base_scores(
        frame,
        outer_train,
        outer_valid,
        labels,
        categories,
        texts,
        title,
        description,
        mixed,
        image,
        args,
        seed=303_000 + outer_fold,
    )

    prediction_output: dict[str, np.ndarray] = {}
    base_output: dict[str, np.ndarray] = {}
    for name, score in outer_base.items():
        values = np.full(len(frame), np.nan, dtype=np.float32)
        values[outer_valid] = score
        base_output[f"score__{name}"] = values

    fold_report: dict[str, object] = {
        "outer_fold": outer_fold,
        "outer_train": len(outer_train),
        "outer_valid": len(outer_valid),
        "outer_group_overlap": 0,
        "inner_group_overlap": 0,
        "inner_fold_counts": np.bincount(inner_folds).astype(int).tolist(),
        "categories": {},
    }
    selected_global = np.full(len(outer_valid), -1, dtype=np.int8)
    selected_robust = np.full(len(outer_valid), -1, dtype=np.int8)

    for category in (COMMON, RARE):
        train_mask = categories[outer_train] == category
        valid_mask = categories[outer_valid] == category
        local_inner_base = {
            name: values[train_mask] for name, values in inner_base.items()
        }
        local_outer_base = {
            name: values[valid_mask] for name, values in outer_base.items()
        }
        train_scores, target_scores, specs = fixed_score_library(
            local_inner_base, local_outer_base
        )
        add_fold_bagged_meta(
            train_scores,
            target_scores,
            labels[outer_train][train_mask],
            inner_folds[train_mask],
        )
        calibrated: dict[str, dict[str, object]] = {}
        for name in sorted(train_scores):
            calibrated[name] = calibrate_candidate(
                labels[outer_train][train_mask],
                train_scores[name],
                target_scores[name],
                inner_folds[train_mask],
            )
            for mode in ("global", "median"):
                key = f"prediction__{name}__{mode}"
                values = prediction_output.setdefault(
                    key, np.full(len(frame), -1, dtype=np.int8)
                )
                values[outer_valid[valid_mask]] = calibrated[name][
                    f"{mode}_prediction"
                ]

        chosen_global = max(
            calibrated,
            key=lambda name: (
                float(calibrated[name]["train_f1"]),
                name,
            ),
        )
        chosen_robust = max(
            calibrated,
            key=lambda name: (
                float(calibrated[name]["crossfit_f1"]),
                float(calibrated[name]["crossfit_fold_min"]),
                name,
            ),
        )
        selected_global[valid_mask] = calibrated[chosen_global][
            "global_prediction"
        ]
        selected_robust[valid_mask] = calibrated[chosen_robust][
            "median_prediction"
        ]
        fold_report["categories"][category] = {
            "inner_rows": int(train_mask.sum()),
            "inner_positive": int(labels[outer_train][train_mask].sum()),
            "valid_rows": int(valid_mask.sum()),
            "valid_positive": int(labels[outer_valid][valid_mask].sum()),
            "fixed_specs": specs,
            "chosen_global": chosen_global,
            "chosen_robust": chosen_robust,
            "candidates": {
                name: {
                    key: value
                    for key, value in values.items()
                    if not key.endswith("_prediction")
                }
                for name, values in calibrated.items()
            },
        }

    for name, local in (
        ("selected_global", selected_global),
        ("selected_robust", selected_robust),
    ):
        values = np.full(len(frame), -1, dtype=np.int8)
        values[outer_valid] = local
        prediction_output[f"prediction__{name}"] = values
    fold_score, fold_parts = category_f1(
        labels[outer_valid], selected_robust, categories[outer_valid]
    )
    fold_report["selected_robust_outer_score"] = float(fold_score)
    fold_report["selected_robust_outer_parts"] = fold_parts
    fold_report["elapsed_seconds"] = time.time() - started
    return {**base_output, **prediction_output}, fold_report


def prediction_metrics(
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, object]:
    score, parts = category_f1(labels, prediction, categories)
    fold_scores: list[float] = []
    fold_parts: list[dict[str, float]] = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        local_score, local_parts = category_f1(
            labels[mask], prediction[mask], categories[mask]
        )
        fold_scores.append(float(local_score))
        fold_parts.append(local_parts)
    rare = categories == RARE
    return {
        "score": float(score),
        "parts": parts,
        "fold_scores": fold_scores,
        "fold_parts": fold_parts,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_min": float(np.min(fold_scores)),
        "rare_tp": int(np.sum((labels[rare] == 1) & (prediction[rare] == 1))),
        "rare_fp": int(np.sum((labels[rare] == 0) & (prediction[rare] == 1))),
        "rare_fn": int(np.sum((labels[rare] == 1) & (prediction[rare] == 0))),
    }


def diversity_report(
    labels: np.ndarray,
    categories: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> dict[str, object]:
    names = [
        f"prediction__{name}__median"
        for name in BASE_NAMES
        if f"prediction__{name}__median" in predictions
    ]
    output: dict[str, object] = {}
    for category in (COMMON, RARE):
        mask = categories == category
        pairs: list[dict[str, object]] = []
        for left_position, left in enumerate(names):
            for right in names[left_position + 1 :]:
                left_values = predictions[left][mask]
                right_values = predictions[right][mask]
                left_error = left_values != labels[mask]
                right_error = right_values != labels[mask]
                pairs.append(
                    {
                        "left": left.removeprefix("prediction__").removesuffix(
                            "__median"
                        ),
                        "right": right.removeprefix("prediction__").removesuffix(
                            "__median"
                        ),
                        "disagreement": float(np.mean(left_values != right_values)),
                        "both_wrong": float(np.mean(left_error & right_error)),
                        "either_correct": float(np.mean(~(left_error & right_error))),
                    }
                )
        output[category] = sorted(
            pairs, key=lambda item: float(item["disagreement"]), reverse=True
        )
    return output


def merge_partial(
    frame: pd.DataFrame,
    labels: np.ndarray,
    categories: np.ndarray,
    folds: np.ndarray,
    partials: list[tuple[dict[str, np.ndarray], dict[str, object]]],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    names = sorted({name for values, _ in partials for name in values})
    merged: dict[str, np.ndarray] = {}
    for name in names:
        if name.startswith("score__"):
            result = np.full(len(frame), np.nan, dtype=np.float32)
            for values, _ in partials:
                if name in values:
                    available = np.isfinite(values[name])
                    result[available] = values[name][available]
            if np.any(~np.isfinite(result)):
                raise AssertionError(f"missing outer scores for {name}")
        else:
            result = np.full(len(frame), -1, dtype=np.int8)
            for values, _ in partials:
                if name in values:
                    available = values[name] >= 0
                    result[available] = values[name][available]
            if np.any(result < 0):
                raise AssertionError(f"missing outer predictions for {name}")
        merged[name] = result

    metrics = {
        name.removeprefix("prediction__"): prediction_metrics(
            labels, categories, folds, values
        )
        for name, values in merged.items()
        if name.startswith("prediction__")
    }
    ranking = sorted(
        ({"name": name, **values} for name, values in metrics.items()),
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    report = {
        "rows": len(frame),
        "uses_policy_or_posthoc_rules": False,
        "protocol": (
            "Outer family folds are untouched by base fitting, meta fitting, model "
            "selection, calibration, and threshold fitting. Inner folds keep the "
            "same label-blind family groups intact. Meta models are fold-bagged and "
            "trained from inner OOF base scores."
        ),
        "folds": [report for _, report in partials],
        "ranking": ranking,
        "diversity": diversity_report(labels, categories, merged),
    }
    return merged, report


def main() -> None:
    args = parse_args()
    started = time.time()
    frame = pd.read_csv(args.data).fillna("")
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    manifest = np.load(args.manifest, allow_pickle=False)
    if not np.array_equal(manifest["id"], frame["id"].to_numpy()):
        raise ValueError("family manifest IDs do not match data")
    group_key = args.group_key or f"group__{args.scheme.split('_seed_')[0]}"
    fold_key = args.fold_key or f"fold__{args.scheme}"
    if group_key not in manifest.files or fold_key not in manifest.files:
        raise ValueError(f"unknown family scheme: {args.scheme}")
    groups = manifest[group_key].astype(np.int64)
    folds = manifest[fold_key].astype(np.int8)
    if args.merge_partials:
        partials: list[tuple[dict[str, np.ndarray], dict[str, object]]] = []
        for fold in sorted(np.unique(folds).astype(int).tolist()):
            path = Path(
                f"outputs/nested_family_stack_{args.scheme}_fold{fold}.npz"
            )
            report_path = path.with_suffix(".json")
            with np.load(path, allow_pickle=False) as archive:
                values = {
                    name: archive[name]
                    for name in archive.files
                    if name.startswith("score__") or name.startswith("prediction__")
                }
            partials.append(
                (
                    values,
                    json.loads(report_path.read_text(encoding="utf-8")),
                )
            )
        merged, report = merge_partial(frame, labels, categories, folds, partials)
        report["scheme"] = args.scheme
        report["inner_folds"] = args.inner_folds
        report["elapsed_seconds"] = time.time() - started
        output = args.output or Path(
            f"outputs/nested_family_stack_{args.scheme}.npz"
        )
        report_path = args.report or output.with_suffix(".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            id=frame["id"].to_numpy(),
            label=labels,
            category=categories,
            fold=folds,
            **merged,
        )
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    mixed = normalize_rows(np.load(args.mixed, allow_pickle=False))
    image = normalize_rows(np.load(args.image, allow_pickle=False))
    if len(mixed) != len(frame) or len(image) != len(frame):
        raise ValueError("embedding row count does not match data")

    target_folds = (
        [args.outer_fold]
        if args.outer_fold is not None
        else sorted(np.unique(folds).astype(int).tolist())
    )
    partials = [
        evaluate_outer_fold(
            fold,
            frame,
            labels,
            categories,
            groups,
            folds,
            mixed,
            image,
            args,
        )
        for fold in target_folds
    ]

    suffix = f"_fold{args.outer_fold}" if args.outer_fold is not None else ""
    output = args.output or Path(
        f"outputs/nested_family_stack_{args.scheme}{suffix}.npz"
    )
    report_path = args.report or output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.outer_fold is None:
        merged, report = merge_partial(
            frame, labels, categories, folds, partials
        )
        report["scheme"] = args.scheme
        report["inner_folds"] = args.inner_folds
        report["elapsed_seconds"] = time.time() - started
        save = {
            "id": frame["id"].to_numpy(),
            "label": labels,
            "category": categories,
            "fold": folds,
            **merged,
        }
    else:
        values, fold_report = partials[0]
        report = {
            "scheme": args.scheme,
            "inner_folds": args.inner_folds,
            "uses_policy_or_posthoc_rules": False,
            "partial": True,
            **fold_report,
        }
        save = {
            "id": frame["id"].to_numpy(),
            "label": labels,
            "category": categories,
            "fold": folds,
            **values,
        }
    np.savez_compressed(output, **save)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
