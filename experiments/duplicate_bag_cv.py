from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from quality_core.common import apply_category_thresholds, best_f1_threshold, category_f1
from quality_core.duplicates import build_duplicate_keys, fold_duplicate_features, image_keys_from_zip
from quality_core.rule_features import conservative_override_masks, normalized_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--images-zip", type=Path, default=Path("data/images.zip"))
    parser.add_argument(
        "--base-oof", type=Path, default=Path("outputs/raw_sparse_hybrid_7seed_oof.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/duplicate_bag_features.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data).fillna("")
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    strata = np.char.add(np.char.add(categories, "__"), labels.astype(str))
    archive = np.load(args.base_oof, allow_pickle=True)
    if not np.array_equal(archive["id"], frame["id"].to_numpy()):
        raise ValueError("base OOF IDs do not match")
    seeds = archive["seeds"].astype(int).tolist()
    base_score = archive["score_median"].astype(np.float64)
    thresholds = {
        category: best_f1_threshold(
            labels[categories == category], base_score[categories == category]
        )[0]
        for category in sorted(np.unique(categories))
    }
    base_prediction = apply_category_thresholds(base_score, categories, thresholds)
    override_masks = conservative_override_masks(frame, fields=normalized_fields(frame))

    def apply_cached_overrides(prediction: np.ndarray) -> np.ndarray:
        output = prediction.copy()
        output[override_masks["supplement_positive"]] = 1
        output[override_masks["supplement_negative"]] = 0
        output[override_masks["flammable_positive"]] = 1
        output[override_masks["flammable_negative"]] = 0
        return output

    keys = build_duplicate_keys(frame, image_keys_from_zip(args.images_zip))
    feature_cube: np.ndarray | None = None
    feature_names: list[str] = []
    for seed_position, seed in enumerate(seeds):
        splits = list(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(frame, strata)
        )
        for fold, (train_index, valid_index) in enumerate(splits):
            features, feature_names = fold_duplicate_features(frame, keys, train_index, valid_index)
            if feature_cube is None:
                feature_cube = np.zeros(
                    (len(frame), len(seeds), features.shape[1]), dtype=np.float32
                )
            feature_cube[valid_index, seed_position] = features
        print(f"seed={seed} complete", flush=True)
    assert feature_cube is not None

    candidates: list[dict[str, object]] = []
    key_names = [name for name in keys if name != "image_any"] + ["image_any"]
    for key_name in key_names:
        probability_column = feature_names.index(f"{key_name}_probability")
        count_column = feature_names.index(f"{key_name}_log_count")
        available_column = feature_names.index(f"{key_name}_available")
        available = feature_cube[:, :, available_column] > 0.5
        probabilities = feature_cube[:, :, probability_column]
        counts = np.expm1(feature_cube[:, :, count_column])
        positive_votes = np.sum(available & (probabilities >= 0.999), axis=1)
        negative_votes = np.sum(available & (probabilities <= 0.001), axis=1)
        max_count = np.max(counts, axis=1)
        for min_count in (1, 2, 3):
            for required_votes in (1, 2, 4, 6):
                positive_mask = (positive_votes >= required_votes) & (negative_votes == 0) & (
                    max_count >= min_count
                )
                negative_mask = (negative_votes >= required_votes) & (positive_votes == 0) & (
                    max_count >= min_count
                )
                prediction = base_prediction.copy()
                prediction[positive_mask] = 1
                prediction[negative_mask] = 0
                overridden = apply_cached_overrides(prediction)
                score, parts = category_f1(labels, overridden, categories)
                candidates.append(
                    {
                        "key": key_name,
                        "min_count": min_count,
                        "required_votes": required_votes,
                        "positive_rows": int(positive_mask.sum()),
                        "negative_rows": int(negative_mask.sum()),
                        "changed": int(np.sum(overridden != base_prediction)),
                        "score": score,
                        "parts": parts,
                    }
                )

    base_override = apply_cached_overrides(base_prediction)
    base_result = category_f1(labels, base_override, categories)
    ranking = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    report = {
        "base": {"score": base_result[0], "parts": base_result[1]},
        "top": ranking[:30],
        "feature_names": feature_names,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        id=frame["id"].to_numpy(),
        label=labels,
        category=categories,
        seeds=np.asarray(seeds),
        features=feature_cube,
        feature_names=np.asarray(feature_names),
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )


if __name__ == "__main__":
    main()
