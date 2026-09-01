from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays: list[np.ndarray] = []
    ids: list[int] = []
    manifests: list[dict[str, object]] = []
    expected_start = 0
    for path in args.inputs:
        array = np.load(path, allow_pickle=False)
        manifest_path = path.with_suffix(path.suffix + ".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_ids = [int(value) for value in manifest["ids"]]
        start = int(manifest.get("start_index", expected_start))
        if start != expected_start:
            raise ValueError(f"non-contiguous shard {path}: expected {expected_start}, found {start}")
        if len(shard_ids) != len(array):
            raise ValueError(f"manifest length mismatch for {path}")
        arrays.append(array)
        ids.extend(shard_ids)
        manifests.append(manifest)
        expected_start += len(array)
    merged = np.concatenate(arrays, axis=0)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate IDs across shards")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, merged, allow_pickle=False)
    output_manifest = {
        "shape": list(merged.shape),
        "dtype": str(merged.dtype),
        "ids": ids,
        "sources": [str(path) for path in args.inputs],
        "source_manifests": manifests,
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"saved": str(args.output), "shape": list(merged.shape)}))


if __name__ == "__main__":
    main()
