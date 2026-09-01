from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC

from quality_core.common import apply_category_thresholds, best_f1_threshold, category_f1
from quality_core.dense_model import DenseBagModel, normalize_rows


DEFAULT_SEEDS = (42, 3407, 1337, 2025, 20260829, 20260701, 777)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dense_bag.joblib"))
    parser.add_argument("--oof-output", type=Path, default=Path("outputs/dense_bag_oof.npz"))
    parser.add_argument("--report", type=Path, default=Path("outputs/dense_bag_report.json"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--jobs", type=int, default=5)
    return parser.parse_args()


def parse_spec(name: str) -> tuple[str, float, float | str, object]:
    parts = name.split("_")
    c_values = {
        "003": 0.03,
        "01": 0.1,
        "03": 0.3,
        "1": 1.0,
        "3": 3.0,
        "10": 10.0,
        "30": 30.0,
    }
    if len(parts) == 3 and parts[0] in {"svc", "lr"}:
        if parts[1] not in c_values or parts[2] not in {"plain", "bal"}:
            raise ValueError(f"invalid dense head specification: {name}")
        return (
            parts[0],
            c_values[parts[1]],
            "scale",
            "balanced" if parts[2] == "bal" else None,
        )
    if len(parts) == 4 and parts[0] == "rbf":
        c_token = parts[1].removeprefix("c")
        gamma_values = {"g07": 0.7, "g1": 1.0, "g15": 1.5}
        if (
            c_token not in c_values
            or parts[2] not in gamma_values
            or parts[3] not in {"plain", "bal"}
        ):
            raise ValueError(f"invalid dense head specification: {name}")
        return (
            "rbf",
            c_values[c_token],
            gamma_values[parts[2]],
            "balanced" if parts[3] == "bal" else None,
        )
    raise ValueError(f"unsupported deployable dense head: {name}")


def main() -> None:
    args = parse_args()
    started = time.time()
    frame = pd.read_csv(args.data)
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    category_values = sorted(np.unique(categories))
    strata = np.char.add(np.char.add(categories, "__"), labels.astype(str))
    vectors = normalize_rows(np.load(args.embeddings, allow_pickle=False))
    if vectors.shape[0] != len(frame):
        raise ValueError("embedding row count does not match data")
    manifest_path = args.embeddings.with_suffix(args.embeddings.suffix + ".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("ids") != frame["id"].astype(int).tolist():
            raise ValueError("embedding manifest IDs do not match data")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    chosen = selection["chosen"]["categories"]
    specs = {category: str(chosen[category]["dense_name"]) for category in category_values}
    parsed_specs = {category: parse_spec(spec) for category, spec in specs.items()}
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    splits_by_seed = {
        seed: list(
            StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed).split(
                vectors, strata
            )
        )
        for seed in seeds
    }

    def fit_partition(seed: int, fold: int, category: str):
        train_index, valid_index = splits_by_seed[seed][fold]
        fit_index = train_index[categories[train_index] == category]
        score_index = valid_index[categories[valid_index] == category]
        kind, c_value, gamma, class_weight = parsed_specs[category]
        if kind == "svc":
            model = LinearSVC(
                C=c_value,
                class_weight=class_weight,
                dual="auto",
                max_iter=20_000,
                random_state=seed + fold,
            )
        elif kind == "lr":
            model = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                max_iter=5_000,
                random_state=seed + fold,
            )
        else:
            model = SVC(
                C=c_value,
                gamma=gamma,
                class_weight=class_weight,
                cache_size=1024,
                random_state=seed + fold,
            )
        model.fit(vectors[fit_index], labels[fit_index])
        score = model.decision_function(vectors[score_index]).astype(np.float32)
        return seed, fold, category, score_index, score, model

    fitted = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(fit_partition)(seed, fold, category)
        for seed in seeds
        for fold in range(args.folds)
        for category in category_values
    )
    seed_position = {seed: position for position, seed in enumerate(seeds)}
    seed_oof = np.zeros((len(frame), len(seeds)), dtype=np.float32)
    models = {
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
    metric, parts = category_f1(labels, prediction, categories)
    config = {
        "seeds": seeds,
        "folds": args.folds,
        "aggregation": "mean folds within seed, median across seeds",
        "category_specs": specs,
    }
    artifact = DenseBagModel(models=models, thresholds=thresholds, seeds=seeds, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    report = {
        "rows": len(frame),
        "embedding_shape": list(vectors.shape),
        "score": metric,
        "parts": parts,
        "thresholds": thresholds,
        "artifact_bytes": args.output.stat().st_size,
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
        dense_score=ensemble_oof,
        prediction=prediction,
        threshold_category=np.asarray(category_values),
        threshold_value=np.asarray([thresholds[category] for category in category_values]),
        config_json=np.asarray(json.dumps(config, ensure_ascii=False)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
