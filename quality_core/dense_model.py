from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


@dataclass
class DenseBagModel:
    models: dict[str, list[list[BaseEstimator]]]
    thresholds: dict[str, float]
    seeds: list[int]
    config: dict[str, object]

    def decision_function(self, embeddings: np.ndarray, categories: np.ndarray) -> np.ndarray:
        vectors = normalize_rows(embeddings)
        category_values = np.asarray(categories, dtype=str)
        if vectors.ndim != 2 or len(vectors) != len(category_values):
            raise ValueError("dense inputs have incompatible shapes")
        if not np.isfinite(vectors).all():
            raise ValueError("dense inputs contain non-finite values")
        output = np.zeros(len(vectors), dtype=np.float64)
        for category, seed_models in self.models.items():
            indices = np.flatnonzero(category_values == category)
            if not len(indices):
                continue
            if not seed_models or not seed_models[0]:
                raise ValueError(f"dense model set is empty for category {category}")
            expected_features = getattr(seed_models[0][0], "n_features_in_", vectors.shape[1])
            if vectors.shape[1] != expected_features:
                raise ValueError(
                    f"dense embedding width {vectors.shape[1]} != expected {expected_features}"
                )
            per_seed = np.column_stack(
                [
                    np.mean(
                        np.column_stack(
                            [model.decision_function(vectors[indices]) for model in fold_models]
                        ),
                        axis=1,
                    )
                    for fold_models in seed_models
                ]
            )
            output[indices] = np.median(per_seed, axis=1)
        if not np.isfinite(output).all():
            raise RuntimeError("dense model produced non-finite scores")
        return output
