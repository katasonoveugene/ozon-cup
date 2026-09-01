from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from quality_core.common import apply_category_thresholds, best_f1_threshold, category_f1
from quality_core.dense_model import normalize_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/data.csv"))
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--cache-mib", type=float, default=1536.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    frame = pd.read_csv(args.data)
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy(dtype=str)
    vectors = normalize_rows(np.load(args.embeddings, allow_pickle=False))
    if len(vectors) != len(frame):
        raise ValueError("embedding row count does not match data")
    manifest_path = args.embeddings.with_suffix(args.embeddings.suffix + ".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("ids") != frame["id"].astype(int).tolist():
            raise ValueError("embedding manifest IDs do not match data")

    strata = np.char.add(np.char.add(categories, "__"), labels.astype(str))
    folds = list(
        StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(
            vectors, strata
        )
    )
    specs = [
        (f"rbf_c{c_name}_g{gamma_name}_{weight_name}", c_value, gamma, class_weight)
        for c_name, c_value in (("3", 3.0), ("10", 10.0), ("30", 30.0))
        for gamma_name, gamma in (("07", 0.7), ("1", 1.0), ("15", 1.5))
        for weight_name, class_weight in (("plain", None), ("bal", "balanced"))
    ]
    category_values = sorted(np.unique(categories))

    def fit_one(spec_position: int, fold: int, category: str):
        name, c_value, gamma, class_weight = specs[spec_position]
        train_index, valid_index = folds[fold]
        fit_index = train_index[categories[train_index] == category]
        score_index = valid_index[categories[valid_index] == category]
        model = SVC(
            C=c_value,
            gamma=gamma,
            class_weight=class_weight,
            cache_size=args.cache_mib,
        )
        model.fit(vectors[fit_index], labels[fit_index])
        score = model.decision_function(vectors[score_index]).astype(np.float32)
        return spec_position, fold, category, score_index, score, int(model.n_support_.sum())

    fitted = Parallel(n_jobs=args.jobs, prefer="threads")(
        delayed(fit_one)(spec_position, fold, category)
        for spec_position in range(len(specs))
        for fold in range(args.folds)
        for category in category_values
    )
    oof = {name: np.zeros(len(frame), dtype=np.float32) for name, *_ in specs}
    support_counts: dict[str, dict[str, list[int]]] = {
        name: {category: [] for category in category_values} for name, *_ in specs
    }
    for spec_position, fold, category, score_index, score, support_count in fitted:
        name = specs[spec_position][0]
        oof[name][score_index] = score
        support_counts[name][category].append(support_count)

    results: list[dict[str, object]] = []
    for name, *_ in specs:
        thresholds = {
            category: best_f1_threshold(
                labels[categories == category], oof[name][categories == category]
            )[0]
            for category in category_values
        }
        prediction = apply_category_thresholds(oof[name], categories, thresholds)
        score, parts = category_f1(labels, prediction, categories)
        results.append(
            {
                "name": name,
                "score": score,
                "parts": parts,
                "thresholds": thresholds,
                "mean_support": {
                    category: float(np.mean(support_counts[name][category]))
                    for category in category_values
                },
            }
        )
    results.sort(key=lambda item: float(item["score"]), reverse=True)
    report = {
        "rows": len(frame),
        "embedding_shape": list(vectors.shape),
        "seed": args.seed,
        "folds": args.folds,
        "elapsed_seconds": time.time() - started,
        "ranking": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        id=frame["id"].to_numpy(),
        label=labels,
        category=categories,
        **oof,
        report_json=np.asarray(json.dumps(report, ensure_ascii=False)),
    )


if __name__ == "__main__":
    main()
