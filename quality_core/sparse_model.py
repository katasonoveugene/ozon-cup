from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from quality_core.common import clean_text
from quality_core.rule_features import apply_conservative_overrides


def raw_value(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def build_raw_text(name: object, description: object) -> str:
    title = raw_value(name)
    body = raw_value(description)
    return f"название {title} название {title} описание {body}"


@dataclass
class SparseBagModel:
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    char_weight: float
    models: dict[str, list[list[LinearSVC]]]
    thresholds: dict[str, float]
    description_labels: dict[tuple[str, str], int]
    seeds: list[int]
    config: dict[str, object]

    def transform(self, frame: pd.DataFrame):
        texts = [
            build_raw_text(name, description)
            for name, description in zip(frame["name"], frame["description"])
        ]
        word_matrix = self.word_vectorizer.transform(texts)
        char_matrix = self.char_vectorizer.transform(texts)
        return hstack(
            [word_matrix, char_matrix * self.char_weight],
            format="csr",
            dtype=np.float32,
        )

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self.transform(frame)
        categories = frame["category"].astype(str).to_numpy()
        output = np.zeros(len(frame), dtype=np.float64)
        for category, seed_models in self.models.items():
            indices = np.flatnonzero(categories == category)
            if not len(indices):
                continue
            per_seed = np.column_stack(
                [
                    np.mean(
                        np.column_stack(
                            [model.decision_function(matrix[indices]) for model in fold_models]
                        ),
                        axis=1,
                    )
                    for fold_models in seed_models
                ]
            )
            output[indices] = np.median(per_seed, axis=1)
        return output

    def postprocess_prediction(
        self, frame: pd.DataFrame, prediction: np.ndarray, apply_overrides: bool = True
    ) -> np.ndarray:
        categories = frame["category"].astype(str).to_numpy()
        prediction = np.asarray(prediction, dtype=np.int8).copy()
        for index, (category, description) in enumerate(zip(categories, frame["description"])):
            key = clean_text(description).casefold()
            if key:
                learned = self.description_labels.get((category, key))
                if learned is not None:
                    prediction[index] = learned
        if apply_overrides:
            prediction = apply_conservative_overrides(frame, prediction)
        return prediction

    def predict(self, frame: pd.DataFrame, apply_overrides: bool = True) -> tuple[np.ndarray, np.ndarray]:
        scores = self.decision_function(frame)
        categories = frame["category"].astype(str).to_numpy()
        prediction = np.zeros(len(frame), dtype=np.int8)
        for category, threshold in self.thresholds.items():
            mask = categories == category
            prediction[mask] = scores[mask] >= threshold
        prediction = self.postprocess_prediction(frame, prediction, apply_overrides=apply_overrides)
        return scores, prediction
