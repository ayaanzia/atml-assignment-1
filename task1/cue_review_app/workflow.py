"""Persistence and balanced-selection helpers for blind cue-conflict review."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


RATINGS = {"approve", "reject"}


def read_candidates(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"cue_id", "bucket_key", "image_path"}
    if not rows:
        raise ValueError(f"No candidates found in {path}")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Candidate manifest is missing columns: {sorted(missing)}")
    ids = [row["cue_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate cue_id values must be unique")
    return rows


def read_ratings(path: Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Ratings file must contain a JSON object")
    return data


def write_json_atomic(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_rating(path: Path, cue_id: str, rating: str, rated_at: str) -> dict[str, dict]:
    if rating not in RATINGS:
        raise ValueError(f"Unknown rating: {rating}")
    ratings = read_ratings(path)
    ratings[cue_id] = {"rating": rating, "rated_at": rated_at}
    write_json_atomic(path, ratings)
    return ratings


def remove_rating(path: Path, cue_id: str) -> dict[str, dict]:
    ratings = read_ratings(path)
    ratings.pop(cue_id, None)
    write_json_atomic(path, ratings)
    return ratings


def review_summary(candidates: list[dict], ratings: dict[str, dict]) -> dict:
    buckets = sorted({row["bucket_key"] for row in candidates})
    counts = {bucket: Counter() for bucket in buckets}
    for row in candidates:
        value = ratings.get(row["cue_id"], {}).get("rating", "unreviewed")
        counts[row["bucket_key"]][value] += 1
    return {
        "total": len(candidates),
        "reviewed": sum(1 for row in candidates if row["cue_id"] in ratings),
        "buckets": {bucket: dict(counts[bucket]) for bucket in buckets},
    }


def _stable_rank(cue_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{cue_id}".encode()).hexdigest()


def balanced_approved_rows(
    candidates: list[dict], ratings: dict[str, dict], quota: int = 20, seed: int = 6304
) -> list[dict]:
    """Return exactly ``quota`` approved rows per bucket using a stable rank."""
    approved = defaultdict(list)
    for row in candidates:
        if ratings.get(row["cue_id"], {}).get("rating") == "approve":
            approved[row["bucket_key"]].append(row)
    buckets = sorted({row["bucket_key"] for row in candidates})
    short = {bucket: len(approved[bucket]) for bucket in buckets if len(approved[bucket]) < quota}
    if short:
        detail = ", ".join(f"{bucket}={count}/{quota}" for bucket, count in short.items())
        raise ValueError(f"Not enough approved images: {detail}")
    selected = []
    for bucket in buckets:
        ranked = sorted(approved[bucket], key=lambda row: _stable_rank(row["cue_id"], seed))
        selected.extend(ranked[:quota])
    return selected


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty selection")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
