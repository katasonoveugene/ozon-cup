from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

INSTRUCTIONS = {
    "БАД": (
        "Represent this product listing for binary compliance classification. Focus on whether text or packaging explicitly marks it "
        "as a dietary supplement, and distinguish sports nutrition or explicit non-supplement statements."
    ),
    "Легковоспламеняющиеся": (
        "Represent this product listing for binary compliance classification. Focus on standalone ignition sources, contained or "
        "included fuel, flammable substances or combustible gases, while distinguishing appliances, built-in sources and absent contents."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("mixed", "text", "image"), default="mixed")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def image_paths(root: Path, item_id: int) -> list[str]:
    directory = root / str(item_id)
    if not directory.is_dir():
        return []
    return [
        str(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.casefold() in VALID_IMAGE_SUFFIXES
    ]


def listing_text(row: object) -> str:
    name = "" if pd.isna(row.name) else str(row.name)
    description = "" if pd.isna(row.description) else str(row.description)
    return f"Product title: {name}\nDeclared category: {row.category}\nProduct description: {description}"


def build_input(row: object, images_root: Path, mode: str) -> dict[str, object]:
    value: dict[str, object] = {"instruction": INSTRUCTIONS[str(row.category)]}
    if mode in {"mixed", "text"}:
        value["text"] = listing_text(row)
    if mode in {"mixed", "image"}:
        value["image"] = image_paths(images_root, int(row.id))
    return value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    sys.path.insert(0, str(args.vendor.resolve()))
    from qvle.qwen3_vl_embedding import Qwen3VLEmbedder

    frame = pd.read_csv(args.data)
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    end_index = len(frame) if args.end_index is None else min(args.end_index, len(frame))
    if end_index < args.start_index:
        raise ValueError("--end-index must not be smaller than --start-index")
    frame = frame.iloc[args.start_index:end_index].copy()
    if args.limit:
        frame = frame.iloc[: args.limit].copy()
    frame = frame.reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    model_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = Qwen3VLEmbedder(
        model_name_or_path=str(args.model),
        max_length=args.max_length,
        min_pixels=4_096,
        max_pixels=args.max_pixels,
        torch_dtype=model_dtype,
        attn_implementation="sdpa",
    )
    print(
        json.dumps(
            {
                "rows": len(frame),
                "mode": args.mode,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "max_pixels": args.max_pixels,
                "dtype": args.dtype,
                "model_load_seconds": time.time() - started,
                "gpu": torch.cuda.get_device_name(0),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    batches: list[np.ndarray] = []
    for start in range(0, len(frame), args.batch_size):
        end = min(start + args.batch_size, len(frame))
        records = [build_input(row, args.images, args.mode) for row in frame.iloc[start:end].itertuples(index=False)]
        vectors = model.process(records, normalize=True)
        batches.append(vectors.to(dtype=torch.float16).cpu().numpy())
        if len(batches) % args.log_every == 0 or end == len(frame):
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "complete": end,
                        "rows_per_second": end / elapsed,
                        "elapsed_seconds": elapsed,
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / (1024**3),
                    }
                ),
                flush=True,
            )

    embeddings = np.concatenate(batches, axis=0)
    np.save(args.output, embeddings, allow_pickle=False)
    manifest = {
        "data": str(args.data),
        "images": str(args.images),
        "model": str(args.model),
        "mode": args.mode,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_pixels": args.max_pixels,
        "dtype": args.dtype,
        "start_index": args.start_index,
        "end_index": end_index,
        "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype),
        "elapsed_seconds": time.time() - started,
        "ids": frame["id"].astype(int).tolist(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"saved": str(args.output), "shape": list(embeddings.shape)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
