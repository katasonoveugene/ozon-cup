from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from quality_core.dense_model import normalize_rows


COMMON = "БАД"
RARE = "Легковоспламеняющиеся"
COMPONENTS = ("sparse", "separated", "mixed_linear", "mixed_rbf", "mixed_mean2")


def _raw(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def current_text(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            f"название {_raw(name)} название {_raw(name)} описание {_raw(description)}"
            for name, description in zip(frame["name"], frame["description"])
        ],
        dtype=object,
    )


def _mean_model_score(models: list[Any], matrix: Any) -> np.ndarray:
    if not models:
        raise ValueError("a deployed model bag must not be empty")
    values = np.column_stack(
        [np.asarray(model.decision_function(matrix), dtype=np.float64) for model in models]
    )
    if not np.isfinite(values).all():
        raise RuntimeError("a deployed model produced non-finite scores")
    return np.mean(values, axis=1)


def mean2_margin(
    query: np.ndarray,
    reference: np.ndarray,
    labels: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    query = normalize_rows(query)
    reference = normalize_rows(reference)
    labels = np.asarray(labels, dtype=np.int8)
    positive = labels == 1
    negative = labels == 0
    if int(positive.sum()) < 2 or int(negative.sum()) < 2:
        raise ValueError("top-two retrieval requires two references in each class")
    output = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), chunk_size):
        end = min(start + chunk_size, len(query))
        similarity = query[start:end] @ reference.T
        positive_top2 = np.partition(similarity[:, positive], -2, axis=1)[:, -2:]
        negative_top2 = np.partition(similarity[:, negative], -2, axis=1)[:, -2:]
        output[start:end] = (
            np.mean(positive_top2, axis=1) - np.mean(negative_top2, axis=1)
        ).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("retrieval produced non-finite scores")
    return output


@dataclass
class ValidatedBlendModel:
    current_word: Any
    current_char: Any
    separated_vectorizers: tuple[Any, Any, Any, Any]
    text_models: dict[str, dict[str, list[Any]]]
    dense_models: dict[str, dict[str, list[Any]]]
    reference_vectors: np.ndarray
    reference_labels: np.ndarray
    reference_categories: np.ndarray
    reference_folds: np.ndarray
    calibration: dict[str, dict[str, Any]]
    config: dict[str, Any]

    def _current_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        texts = current_text(frame)
        word = self.current_word.transform(texts)
        char = self.current_char.transform(texts)
        return sparse.hstack(
            [word, char * np.float32(0.375)], format="csr", dtype=np.float32
        )

    def _separated_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        title = np.asarray([_raw(value) for value in frame["name"]], dtype=object)
        description = np.asarray(
            [_raw(value) for value in frame["description"]], dtype=object
        )
        fields = (title, title, description, description)
        scales = (0.5, 0.1875, 1.0, 0.375)
        blocks = [
            vectorizer.transform(field) * np.float32(scale)
            for vectorizer, field, scale in zip(
                self.separated_vectorizers, fields, scales
            )
        ]
        return sparse.hstack(blocks, format="csr", dtype=np.float32)

    def component_scores(
        self, frame: pd.DataFrame, embeddings: np.ndarray
    ) -> dict[str, np.ndarray]:
        categories = frame["category"].astype(str).to_numpy(dtype=str)
        vectors = normalize_rows(embeddings)
        if len(vectors) != len(frame):
            raise ValueError("embedding and metadata row counts differ")
        current_matrix = self._current_matrix(frame)
        separated_matrix = self._separated_matrix(frame)
        result = {
            name: np.zeros(len(frame), dtype=np.float64) for name in COMPONENTS
        }
        reference_vectors = normalize_rows(self.reference_vectors)
        reference_labels = np.asarray(self.reference_labels, dtype=np.int8)
        reference_categories = np.asarray(self.reference_categories, dtype=str)
        reference_folds = np.asarray(self.reference_folds, dtype=np.int8)
        fold_values = sorted(int(value) for value in np.unique(reference_folds))
        if fold_values != list(range(int(self.config["folds"]))):
            raise RuntimeError("reference folds differ from the deployment contract")

        for category in (COMMON, RARE):
            rows = np.flatnonzero(categories == category)
            if not len(rows):
                continue
            result["sparse"][rows] = _mean_model_score(
                self.text_models[category]["sparse"], current_matrix[rows]
            )
            result["separated"][rows] = _mean_model_score(
                self.text_models[category]["separated"], separated_matrix[rows]
            )
            result["mixed_linear"][rows] = _mean_model_score(
                self.dense_models[category]["mixed_linear"], vectors[rows]
            )
            result["mixed_rbf"][rows] = _mean_model_score(
                self.dense_models[category]["mixed_rbf"], vectors[rows]
            )
            fold_margins: list[np.ndarray] = []
            for fold in fold_values:
                reference_mask = (reference_categories == category) & (
                    reference_folds != fold
                )
                fold_margins.append(
                    mean2_margin(
                        vectors[rows],
                        reference_vectors[reference_mask],
                        reference_labels[reference_mask],
                        int(self.config["retrieval_chunk_size"]),
                    )
                )
            result["mixed_mean2"][rows] = np.mean(
                np.column_stack(fold_margins), axis=1
            )
        if any(not np.isfinite(values).all() for values in result.values()):
            raise RuntimeError("blend components contain non-finite values")
        return result

    def predict(
        self, frame: pd.DataFrame, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        categories = frame["category"].astype(str).to_numpy(dtype=str)
        components = self.component_scores(frame, embeddings)
        blended = np.zeros(len(frame), dtype=np.float64)
        prediction = np.zeros(len(frame), dtype=np.int8)
        for category in (COMMON, RARE):
            rows = categories == category
            if not np.any(rows):
                continue
            values = self.calibration[category]
            score = np.zeros(int(rows.sum()), dtype=np.float64)
            for name, weight in values["weights"].items():
                center = float(values["centers"][name])
                scale = float(values["scales"][name])
                if not np.isfinite(scale) or scale <= 0:
                    raise RuntimeError("invalid deployment component scale")
                score += float(weight) * (components[name][rows] - center) / scale
            blended[rows] = score
            prediction[rows] = score >= float(values["threshold"])
        return blended, prediction, components
