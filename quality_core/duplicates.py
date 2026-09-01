from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from quality_core.common import clean_text, normalize_exact, normalize_tokens


ImageKey = tuple[int, int]


def image_keys_from_zip(path: Path) -> dict[int, tuple[ImageKey, ...]]:
    """Read lossless byte signatures from ZIP metadata without decoding images."""
    result: dict[int, list[ImageKey]] = defaultdict(list)
    valid_suffixes = {".jpg", ".jpeg", ".png"}
    with ZipFile(path) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if member.suffix.casefold() not in valid_suffixes or len(member.parts) < 3:
                continue
            try:
                item_id = int(member.parts[-2])
            except ValueError:
                continue
            result[item_id].append((int(info.CRC), int(info.file_size)))
    return {item_id: tuple(sorted(set(keys))) for item_id, keys in result.items()}


def build_duplicate_keys(frame: pd.DataFrame, image_keys: dict[int, tuple[ImageKey, ...]]) -> dict[str, list[Hashable]]:
    names = frame["name"].fillna("").astype(str).tolist()
    descriptions = frame["description"].fillna("").astype(str).tolist()
    clean_names = [clean_text(value).casefold() for value in names]
    clean_descriptions = [clean_text(value).casefold() for value in descriptions]
    token_names = [normalize_tokens(value) for value in names]
    token_descriptions = [normalize_tokens(value) for value in descriptions]
    compact_names = [normalize_exact(value) for value in names]
    compact_descriptions = [normalize_exact(value) for value in descriptions]
    sets = [image_keys.get(int(item_id), ()) for item_id in frame["id"]]
    return {
        "name_clean": clean_names,
        "description_clean": clean_descriptions,
        "pair_tokens": list(zip(token_names, token_descriptions)),
        "name_tokens": token_names,
        "description_tokens": token_descriptions,
        "name_compact": compact_names,
        "description_compact": compact_descriptions,
        "image_set": sets,
        "image_any": sets,
    }


def _single_key_features(
    labels: np.ndarray,
    keys: Sequence[Hashable],
    train_index: np.ndarray,
    valid_index: np.ndarray,
    prior: float,
) -> np.ndarray:
    positive: dict[Hashable, int] = defaultdict(int)
    total: dict[Hashable, int] = defaultdict(int)
    for index in train_index:
        key = keys[int(index)]
        if key == "" or key == ():
            continue
        positive[key] += int(labels[index])
        total[key] += 1
    features = np.zeros((len(valid_index), 4), dtype=np.float32)
    features[:, 0] = prior
    for row, index in enumerate(valid_index):
        key = keys[int(index)]
        count = total.get(key, 0)
        if count:
            probability = positive[key] / count
            features[row] = probability, np.log1p(count), 1.0, abs(2 * probability - 1)
    return features


def _any_image_features(
    labels: np.ndarray,
    image_sets: Sequence[tuple[ImageKey, ...]],
    train_index: np.ndarray,
    valid_index: np.ndarray,
    prior: float,
) -> np.ndarray:
    inverted: dict[ImageKey, list[int]] = defaultdict(list)
    for index in train_index:
        for key in image_sets[int(index)]:
            inverted[key].append(int(index))
    features = np.zeros((len(valid_index), 4), dtype=np.float32)
    features[:, 0] = prior
    for row, index in enumerate(valid_index):
        matches: set[int] = set()
        for key in image_sets[int(index)]:
            matches.update(inverted.get(key, ()))
        if matches:
            match_index = np.fromiter(matches, dtype=np.int64)
            probability = float(labels[match_index].mean())
            features[row] = probability, np.log1p(len(matches)), 1.0, abs(2 * probability - 1)
    return features


def _id_neighbor_features(
    ids: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> np.ndarray:
    train_order = train_index[np.argsort(ids[train_index])]
    train_ids = ids[train_order]
    train_labels = labels[train_order]
    output = np.zeros((len(valid_index), 8), dtype=np.float32)
    for row, index in enumerate(valid_index):
        position = int(np.searchsorted(train_ids, ids[index]))
        left = max(0, position - 12)
        right = min(len(train_ids), position + 12)
        candidate_ids = train_ids[left:right]
        candidate_labels = train_labels[left:right]
        distance = np.abs(candidate_ids - ids[index])
        order = np.argsort(distance)
        for column, k in enumerate((1, 3, 5, 11)):
            selected = order[: min(k, len(order))]
            weights = 1.0 / np.maximum(distance[selected], 1)
            output[row, column] = np.average(candidate_labels[selected], weights=weights)
            output[row, column + 4] = float(distance[selected[0]])
    return output


def fold_duplicate_features(
    frame: pd.DataFrame,
    keys: dict[str, list[Hashable]],
    train_index: np.ndarray,
    valid_index: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy()
    ids = frame["id"].to_numpy(dtype=np.int64)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for category in sorted(np.unique(categories)):
        category_train = train_index[categories[train_index] == category]
        category_valid_positions = np.flatnonzero(categories[valid_index] == category)
        category_valid = valid_index[category_valid_positions]
        prior = float(labels[category_train].mean())
        category_blocks: list[np.ndarray] = []
        category_names: list[str] = []
        for key_name, values in keys.items():
            if key_name == "image_any":
                block = _any_image_features(labels, values, category_train, category_valid, prior)
            else:
                block = _single_key_features(labels, values, category_train, category_valid, prior)
            category_blocks.append(block)
            category_names.extend(
                [f"{key_name}_probability", f"{key_name}_log_count", f"{key_name}_available", f"{key_name}_purity"]
            )
        category_blocks.append(_id_neighbor_features(ids, labels, category_train, category_valid))
        category_names.extend([f"id_vote_{k}" for k in (1, 3, 5, 11)] + [f"id_distance_{k}" for k in (1, 3, 5, 11)])
        assembled = np.column_stack(category_blocks)
        if not blocks:
            width = assembled.shape[1]
            full = np.zeros((len(valid_index), width), dtype=np.float32)
            blocks.append(full)
            names = category_names
        blocks[0][category_valid_positions] = assembled
    return blocks[0], names
