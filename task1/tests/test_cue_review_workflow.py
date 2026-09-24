import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cue_review_app"))
from workflow import balanced_approved_rows, read_candidates, remove_rating, save_rating


def test_balanced_selection_is_exact_and_stable():
    candidates = [
        {"cue_id": f"{bucket}-{i}", "bucket_key": bucket, "image_path": f"{bucket}-{i}.png"}
        for bucket in ("a", "b") for i in range(4)
    ]
    ratings = {row["cue_id"]: {"rating": "approve"} for row in candidates}
    first = balanced_approved_rows(candidates, ratings, quota=3, seed=6304)
    second = balanced_approved_rows(list(reversed(candidates)), ratings, quota=3, seed=6304)
    assert [row["cue_id"] for row in first] == [row["cue_id"] for row in second]
    assert len(first) == 6


def test_balanced_selection_rejects_short_bucket():
    candidates = [{"cue_id": "a-1", "bucket_key": "a", "image_path": "a.png"}]
    with pytest.raises(ValueError, match="a=1/2"):
        balanced_approved_rows(candidates, {"a-1": {"rating": "approve"}}, quota=2)


def test_ratings_are_resumable(tmp_path):
    path = tmp_path / "ratings.json"
    save_rating(path, "one", "approve", "now")
    assert json.loads(path.read_text())["one"]["rating"] == "approve"
    remove_rating(path, "one")
    assert json.loads(path.read_text()) == {}


def test_candidate_ids_must_be_unique(tmp_path):
    path = tmp_path / "candidates.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cue_id", "bucket_key", "image_path"])
        writer.writeheader()
        writer.writerows([
            {"cue_id": "same", "bucket_key": "a", "image_path": "1.png"},
            {"cue_id": "same", "bucket_key": "a", "image_path": "2.png"},
        ])
    with pytest.raises(ValueError, match="unique"):
        read_candidates(path)
