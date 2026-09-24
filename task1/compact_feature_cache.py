#!/usr/bin/env python3
"""Compact legacy per-image feature files that retained full-batch storage."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=Path(__file__).resolve().parent / "results/cache",
    )
    parser.add_argument("--prefix", default="feat__clip_vit_b_32_quickgelu__")
    parser.add_argument("--minimum-bytes", type=int, default=1_000_000)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(args.cache_dir.glob(f"{args.prefix}*__image_*.pt"))
    oversized = [path for path in paths if path.stat().st_size >= args.minimum_bytes]
    print(f"Found {len(oversized)} oversized files out of {len(paths)} matching files.")
    for index, path in enumerate(oversized, 1):
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        compact = {
            "features": bundle["features"].clone(),
            "labels": list(bundle["labels"]),
            "image_ids": list(bundle["image_ids"]),
        }
        temporary = path.with_suffix(path.suffix + ".compact")
        torch.save(compact, temporary)
        os.replace(temporary, path)
        if index % 500 == 0 or index == len(oversized):
            print(f"Compacted {index}/{len(oversized)}")


if __name__ == "__main__":
    main()
