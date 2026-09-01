from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from quality_core.rule_features import conservative_override_masks, normalized_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--sparse-oof", type=Path, default=Path("outputs/sparse_bag_oof.npz"))
    parser.add_argument("--dense-oof", type=Path, required=True)
    parser.add_argument(
        "--duplicate-features", type=Path, default=Path("outputs/duplicate_bag_features.npz")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/fusion_search.json"))
    parser.add_argument("--config-output", type=Path, default=Path("artifacts/fusion.json"))
    return parser.parse_args()


def fixed_prediction_masks(
    frame: pd.DataFrame, duplicate_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    duplicate = np.load(duplicate_path, allow_pickle=False)
    names = duplicate["feature_names"].tolist()
    features = duplicate["features"]
    probability = features[:, :, names.index("description_clean_probability")]
    available = features[:, :, names.index("description_clean_available")] > 0.5
    count = np.expm1(features[:, :, names.index("description_clean_log_count")])
    positive_votes = np.sum(available & (probability >= 0.999), axis=1)
    negative_votes = np.sum(available & (probability <= 0.001), axis=1)
    max_count = np.max(count, axis=1)
    positive = (positive_votes >= 2) & (negative_votes == 0) & (max_count >= 1)
    negative = (negative_votes >= 2) & (positive_votes == 0) & (max_count >= 1)

    rule_masks = conservative_override_masks(frame, fields=normalized_fields(frame))
    forced = positive | negative
    value = positive.astype(np.int8)
    for name, target in (
        ("supplement_positive", 1),
        ("supplement_negative", 0),
        ("flammable_positive", 1),
        ("flammable_negative", 0),
    ):
        mask = rule_masks[name]
        forced[mask] = True
        value[mask] = target
    return forced, value


def best_forced_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    forced: np.ndarray,
    forced_value: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    free_index = np.flatnonzero(~forced)
    fixed_positive = forced & (forced_value == 1)
    fixed_tp = int(np.sum(labels[fixed_positive] == 1))
    fixed_predicted = int(fixed_positive.sum())
    total_positive = int(labels.sum())
    order = free_index[np.argsort(-scores[free_index], kind="stable")]
    sorted_labels = labels[order]
    cumulative_tp = np.r_[0, np.cumsum(sorted_labels)]
    predicted_count = np.arange(len(order) + 1)
    tp = fixed_tp + cumulative_tp
    denominator = total_positive + fixed_predicted + predicted_count
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    valid = np.ones(len(order) + 1, dtype=bool)
    if len(order) > 1:
        valid[1:-1] = scores[order[:-1]] != scores[order[1:]]
    candidate = np.flatnonzero(valid)
    k = int(candidate[np.argmax(f1[candidate])])
    if not len(order):
        threshold = 0.0
    elif k == 0:
        threshold = float(np.nextafter(scores[order[0]], np.inf))
    elif k == len(order):
        threshold = float(np.nextafter(scores[order[-1]], -np.inf))
    else:
        threshold = float((scores[order[k - 1]] + scores[order[k]]) / 2)
    prediction = (scores >= threshold).astype(np.int8)
    prediction[forced] = forced_value[forced]
    return threshold, float(f1_score(labels, prediction)), prediction


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.data).fillna("")
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    sparse = np.load(args.sparse_oof, allow_pickle=False)
    dense = np.load(args.dense_oof, allow_pickle=False)
    expected_ids = frame["id"].to_numpy()
    if not np.array_equal(sparse["id"], expected_ids) or not np.array_equal(dense["id"], expected_ids):
        raise ValueError("OOF IDs do not match data")
    sparse_score = sparse["score"].astype(np.float64)
    excluded = {"id", "label", "category", "text"}
    dense_names = [
        name
        for name in dense.files
        if name not in excluded
        and not name.startswith("blend_")
        and (name == "dense_score" or name.startswith(("svc_", "lr_", "rbf_")))
        and dense[name].shape == (len(frame),)
        and dense[name].dtype.kind == "f"
    ]
    if not dense_names:
        raise ValueError("dense OOF archive contains no deployable row-level score")
    forced, forced_value = fixed_prediction_masks(frame, args.duplicate_features)
    weights = (0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0)
    report: dict[str, object] = {"categories": {}, "dense_candidates": dense_names}
    final_config: dict[str, object] = {
        "embedding_batch_size": 128,
        "embedding_max_length": 4096,
        "embedding_max_pixels": 262_144,
        "comment_batch_size": 128,
        "comment_max_new_tokens": 40,
        "categories": {},
    }

    for category in sorted(np.unique(categories)):
        mask = categories == category
        local_labels = labels[mask]
        sparse_local = sparse_score[mask]
        sparse_center = float(np.mean(sparse_local))
        sparse_scale = float(np.std(sparse_local) + 1e-12)
        sparse_standard = (sparse_local - sparse_center) / sparse_scale
        candidates: list[dict[str, object]] = []
        for dense_name in dense_names:
            dense_local = dense[dense_name].astype(np.float64)[mask]
            dense_center = float(np.mean(dense_local))
            dense_scale = float(np.std(dense_local) + 1e-12)
            dense_standard = (dense_local - dense_center) / dense_scale
            for sparse_weight in weights:
                dense_weight = 1.0 - sparse_weight
                score = sparse_weight * sparse_standard + dense_weight * dense_standard
                threshold, metric, _ = best_forced_threshold(
                    local_labels,
                    score,
                    forced[mask],
                    forced_value[mask],
                )
                candidates.append(
                    {
                        "dense_name": dense_name,
                        "sparse_weight": sparse_weight,
                        "dense_weight": dense_weight,
                        "sparse_center": sparse_center,
                        "sparse_scale": sparse_scale,
                        "dense_center": dense_center,
                        "dense_scale": dense_scale,
                        "threshold": threshold,
                        "f1": metric,
                    }
                )
        candidates.sort(key=lambda item: float(item["f1"]), reverse=True)
        report["categories"][category] = candidates[:40]
        final_config["categories"][category] = candidates[0]

    chosen_predictions = np.zeros(len(frame), dtype=np.int8)
    for category, values in final_config["categories"].items():
        mask = categories == category
        sparse_standard = (sparse_score[mask] - values["sparse_center"]) / values[
            "sparse_scale"
        ]
        dense_values = dense[values["dense_name"]].astype(np.float64)[mask]
        dense_standard = (dense_values - values["dense_center"]) / values["dense_scale"]
        score = values["sparse_weight"] * sparse_standard + values["dense_weight"] * dense_standard
        chosen_predictions[mask] = score >= values["threshold"]
    chosen_predictions[forced] = forced_value[forced]
    per_category = {
        category: float(f1_score(labels[categories == category], chosen_predictions[categories == category]))
        for category in sorted(np.unique(categories))
    }
    report["chosen"] = final_config
    report["chosen_parts"] = per_category
    report["chosen_score"] = float(np.mean(list(per_category.values())))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    args.config_output.write_text(
        json.dumps(final_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
