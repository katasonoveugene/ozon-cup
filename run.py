from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import joblib
import numpy as np
import pandas as pd

from quality_core.comments import format_results, generate_comments
from quality_core.embedding_runtime import extract_embeddings


EXPECTED_CATEGORIES = {"БАД", "Легковоспламеняющиеся"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--test_data_path",
        "--test-data-path",
        dest="test_data_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        "--output-path",
        dest="output_path",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def validate_input(frame: pd.DataFrame) -> None:
    required = {"id", "name", "description", "category"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("input must contain at least one row")
    if frame["id"].duplicated().any():
        raise ValueError("input IDs must be unique")
    unknown = set(frame["category"].astype(str)).difference(EXPECTED_CATEGORIES)
    if unknown:
        raise ValueError(f"unsupported categories: {sorted(unknown)}")


def fuse_scores(
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
    categories: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(sparse_scores) == len(dense_scores) == len(categories)):
        raise ValueError("fusion input lengths do not match")
    if not np.isfinite(sparse_scores).all() or not np.isfinite(dense_scores).all():
        raise ValueError("fusion inputs contain non-finite scores")
    fused = np.zeros(len(sparse_scores), dtype=np.float64)
    predictions = np.zeros(len(sparse_scores), dtype=np.int8)
    category_config = config["categories"]
    for category in sorted(EXPECTED_CATEGORIES):
        mask = categories == category
        values = category_config[category]
        for scale_name in ("sparse_scale", "dense_scale"):
            scale = float(values[scale_name])
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"invalid {scale_name} for category {category}")
        sparse_standard = (sparse_scores[mask] - values["sparse_center"]) / values[
            "sparse_scale"
        ]
        dense_standard = (dense_scores[mask] - values["dense_center"]) / values[
            "dense_scale"
        ]
        fused[mask] = (
            values["sparse_weight"] * sparse_standard
            + values["dense_weight"] * dense_standard
        )
        predictions[mask] = fused[mask] >= values["threshold"]
    return fused, predictions


def validate_output(frame: pd.DataFrame, output: pd.DataFrame) -> None:
    if output.columns.tolist() != ["id", "result"]:
        raise RuntimeError("output columns are invalid")
    if output["id"].tolist() != frame["id"].tolist():
        raise RuntimeError("output IDs or order changed")
    for value in output["result"].astype(str):
        if not value.startswith("<комментарий>") or "<вердикт>" not in value:
            raise RuntimeError("output markup is invalid")
        comment, verdict = value[len("<комментарий>") :].split("<вердикт>", maxsplit=1)
        if not 50 <= len(comment) <= 300:
            raise RuntimeError("comment length is outside [50, 300]")
        if verdict not in {"бан", "не бан"}:
            raise RuntimeError("verdict is invalid")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    frame = pd.read_csv(args.test_data_path).fillna("")
    validate_input(frame)
    images_root = args.test_data_path.parent / "images"
    if not images_root.is_dir():
        raise FileNotFoundError(f"images directory not found: {images_root}")
    shared_models = Path(os.environ.get("SHARED_MODELS_PATH", "/shared_models"))

    sparse_model = joblib.load(root / "artifacts" / "sparse_bag.joblib")
    dense_model = joblib.load(root / "artifacts" / "dense_bag.joblib")
    fusion_config = json.loads(
        (root / "artifacts" / "fusion.json").read_text(encoding="utf-8")
    )
    sparse_scores = sparse_model.decision_function(frame)
    embeddings = extract_embeddings(
        frame=frame,
        images_root=images_root,
        model_path=shared_models / "Qwen" / "Qwen3-VL-Embedding-2B",
        project_root=root,
        batch_size=int(fusion_config.get("embedding_batch_size", 64)),
        max_length=int(fusion_config.get("embedding_max_length", 4096)),
        max_pixels=int(fusion_config.get("embedding_max_pixels", 262_144)),
    )
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    dense_scores = dense_model.decision_function(embeddings, categories)
    _, predictions = fuse_scores(
        sparse_scores, dense_scores, categories, fusion_config
    )
    predictions = sparse_model.postprocess_prediction(frame, predictions, apply_overrides=True)

    comments = generate_comments(
        frame,
        predictions,
        model_path=shared_models / "Qwen" / "Qwen3.5-4B",
        batch_size=int(fusion_config.get("comment_batch_size", 64)),
        max_new_tokens=int(fusion_config.get("comment_max_new_tokens", 40)),
    )
    output = pd.DataFrame(
        {"id": frame["id"].to_numpy(), "result": format_results(comments, predictions)}
    )
    validate_output(frame, output)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print(f"saved={args.output_path} rows={len(output)}", flush=True)


if __name__ == "__main__":
    main()
