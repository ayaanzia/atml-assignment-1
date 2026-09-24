#!/usr/bin/env python3
"""Append unique AdaIN candidates only to review buckets below quota."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
TASK_DIR = APP_DIR.parent
REPO_DIR = TASK_DIR.parent
sys.path.insert(0, str(REPO_DIR))

from task1.analysis.cue_conflict_calibration import cue_conflict_id
from task1.cue_review_app.workflow import read_candidates, read_ratings, write_csv_atomic
from task1.utils import metrics as mx
from task1.utils.adain import AdaINStyleTransfer
from task1.utils.config import DEVICE, STL10_CLASSES, load_json, make_rng
from task1.utils.data import STL10Wrapped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=TASK_DIR / "results/cue_conflict_candidates/candidates.csv")
    parser.add_argument("--ratings", type=Path, default=TASK_DIR / "results/cue_conflict_candidates/review_ratings.json")
    parser.add_argument("--quota", type=int, default=20)
    parser.add_argument("--add", type=int, default=40, help="New unique candidates per deficient bucket")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vgg", type=Path, default=TASK_DIR / "models/vgg_normalised.pth")
    parser.add_argument("--decoder", type=Path, default=TASK_DIR / "models/decoder.pth")
    return parser.parse_args()


def approval_counts(candidates: list[dict], ratings: dict) -> Counter:
    counts = Counter()
    for row in candidates:
        if ratings.get(row["cue_id"], {}).get("rating") == "approve":
            counts[row["bucket_key"]] += 1
    return counts


def load_eval_pools(candidates: list[dict]) -> dict[str, list[tuple[torch.Tensor, str]]]:
    subset = load_json(TASK_DIR / "results/eval_subset_ids.json")
    test_ds = STL10Wrapped("test", download=False)
    wanted = {row["content_class"] for row in candidates} | {row["texture_class"] for row in candidates}
    pools = {name: [] for name in wanted}
    for index in subset["indices"]:
        image, label, image_id = test_ds[index]
        name = STL10_CLASSES[int(label)]
        if name in pools:
            pools[name].append((image, image_id))
    empty = [name for name, values in pools.items() if not values]
    if empty:
        raise RuntimeError(f"No evaluation images loaded for: {empty}")
    return pools


def choose_new_pairs(content_pool, style_pool, direction, existing_ids, count, seed_name):
    available = []
    for content in content_pool:
        for style in style_pool:
            probe = {"content_id": content[1], "style_id": style[1], "direction": direction}
            if cue_conflict_id(probe) not in existing_ids:
                available.append((content, style))
    if len(available) < count:
        raise RuntimeError(f"Only {len(available)} unused pairs remain; requested {count}")
    rng = make_rng(seed_name)
    indices = rng.choice(len(available), size=count, replace=False)
    return [available[int(index)] for index in indices]


def generate_bucket(model, template, pairs, image_dir, batch_size):
    generated = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        content = torch.stack([item[0][0] for item in batch])
        style = torch.stack([item[1][0] for item in batch])
        with torch.no_grad():
            stylized = model.style_transfer(content, style, alpha=1.0).cpu()
        for output, ((content_image, content_id), (_, style_id)) in zip(stylized, batch):
            output_np = output.permute(1, 2, 0).numpy()
            check = mx.is_valid_cue_conflict(content_image.permute(1, 2, 0).numpy(), output_np)
            record = {
                "pair": template["pair"], "direction": template["direction"],
                "bucket_key": template["bucket_key"], "content_class": template["content_class"],
                "texture_class": template["texture_class"],
                "content_class_idx": template["content_class_idx"],
                "texture_class_idx": template["texture_class_idx"],
                "content_id": content_id, "style_id": style_id,
                "ssim": check["ssim_to_content"], "pixel_std": check["pixel_std"],
                "degenerate": check["degenerate"],
            }
            cue_id = cue_conflict_id(record)
            image_name = f"{cue_id}.png"
            Image.fromarray((output_np.clip(0, 1) * 255).round().astype(np.uint8)).save(image_dir / image_name)
            generated.append({"cue_id": cue_id, "image_path": f"images/{image_name}", **record})
    return generated


def main():
    args = parse_args()
    candidates = read_candidates(args.manifest)
    ratings = read_ratings(args.ratings)
    counts = approval_counts(candidates, ratings)
    buckets = sorted({row["bucket_key"] for row in candidates})
    deficient = [bucket for bucket in buckets if counts[bucket] < args.quota]
    if not deficient:
        print("Every bucket already meets the approval quota; nothing to generate.")
        return
    print("Deficient buckets:")
    for bucket in deficient:
        print(f"  {bucket}: {counts[bucket]}/{args.quota} approved")
    print(f"Generating {args.add} new unique candidates for each deficient bucket on {DEVICE}.")

    pools = load_eval_pools(candidates)
    model = AdaINStyleTransfer(str(args.vgg), str(args.decoder)).to(DEVICE)
    by_bucket = defaultdict(list)
    for row in candidates:
        by_bucket[row["bucket_key"]].append(row)
    existing_ids = {row["cue_id"] for row in candidates}
    image_dir = args.manifest.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    additions = []
    for bucket in deficient:
        template = by_bucket[bucket][0]
        pairs = choose_new_pairs(
            pools[template["content_class"]], pools[template["texture_class"]],
            template["direction"], existing_ids, args.add,
            f"cue_conflict_supplement:{bucket}:{len(by_bucket[bucket])}",
        )
        rows = generate_bucket(model, template, pairs, image_dir, args.batch_size)
        additions.extend(rows)
        existing_ids.update(row["cue_id"] for row in rows)
        print(f"  appended {len(rows)} to {bucket}")

    fieldnames = list(candidates[0])
    normalized = [{key: row.get(key, "") for key in fieldnames} for row in additions]
    write_csv_atomic(args.manifest, candidates + normalized)
    print(f"Manifest now contains {len(candidates) + len(normalized)} unique candidates.")
    print("Restart the review GUI, keep Status=Unreviewed, and continue reviewing.")


if __name__ == "__main__":
    main()
