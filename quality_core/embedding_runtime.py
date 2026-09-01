from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


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


def build_input(row: object, images_root: Path) -> dict[str, object]:
    return {
        "instruction": INSTRUCTIONS[str(row.category)],
        "text": listing_text(row),
        "image": image_paths(images_root, int(row.id)),
    }


def extract_embeddings(
    frame: pd.DataFrame,
    images_root: Path,
    model_path: Path,
    project_root: Path,
    batch_size: int = 64,
    max_length: int = 4096,
    max_pixels: int = 262_144,
) -> np.ndarray:
    import torch

    third_party = project_root / "third_party"
    if str(third_party) not in sys.path:
        sys.path.insert(0, str(third_party))
    from qvle.qwen3_vl_embedding import Qwen3VLEmbedder

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multimodal inference")
    started = time.time()
    model = Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        max_length=max_length,
        min_pixels=4_096,
        max_pixels=max_pixels,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )

    def checked_process(records: list[dict[str, object]]) -> np.ndarray:
        """Isolate malformed inputs and reduce oversized batches without reordering rows."""
        try:
            vectors = model.process(records, normalize=True)
            if vectors.ndim != 2 or vectors.shape[0] != len(records):
                raise RuntimeError(
                    f"embedding shape {tuple(vectors.shape)} does not match batch size {len(records)}"
                )
            if not torch.isfinite(vectors).all():
                raise RuntimeError("embedding batch contains non-finite values")
            return vectors.to(dtype=torch.float16).cpu().numpy()
        except Exception as error:
            failure_type = type(error).__name__
            failure_message = str(error)
            was_oom = isinstance(error, torch.cuda.OutOfMemoryError)

        # Retry only after leaving the exception scope. Keeping the traceback alive while
        # recursing retains failed-forward CUDA tensors, so empty_cache() cannot release them.
        gc.collect()
        if was_oom:
            torch.cuda.empty_cache()
        if len(records) > 1:
            midpoint = len(records) // 2
            print(
                f"embedding batch retry size={len(records)} reason={failure_type}",
                flush=True,
            )
            left = checked_process(records[:midpoint])
            right = checked_process(records[midpoint:])
            return np.concatenate([left, right], axis=0)
        record = records[0]
        if record.get("image"):
            print(
                f"embedding row retry without images reason={failure_type}",
                flush=True,
            )
            text_only = dict(record)
            text_only["image"] = []
            return checked_process([text_only])
        raise RuntimeError(
            f"embedding failed for a text-only row: {failure_type}: {failure_message}"
        )

    batches: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        records = [
            build_input(row, images_root)
            for row in frame.iloc[start:end].itertuples(index=False)
        ]
        batches.append(checked_process(records))
        if end == len(frame) or end % (batch_size * 10) == 0:
            print(
                f"embedded={end}/{len(frame)} elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    embeddings = np.concatenate(batches, axis=0)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings
