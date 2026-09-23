import json

import numpy as np
import torch

from task1.analysis.cue_conflict_calibration import (
    ReviewSession,
    cue_conflict_id,
    load_manual_ratings,
    sample_for_review,
    sweep_thresholds,
)


def _record(bucket: int, item: int, ssim: float) -> dict:
    return {
        "pair": f"pair-{bucket}",
        "direction": "A_shape_B_texture",
        "content_id": f"content-{bucket}-{item}",
        "style_id": f"style-{bucket}-{item}",
        "ssim": ssim,
        "image": torch.zeros(3, 4, 4),
    }


def test_stable_id_and_stratified_sampling_are_order_independent():
    records = [_record(bucket, item, 0.1 + item / 100) for bucket in range(10) for item in range(20)]
    forward = sample_for_review(records, sample_frac=0.10)
    reverse = sample_for_review(list(reversed(records)), sample_frac=0.10)
    assert len(forward) == 20
    assert [item["review_id"] for item in forward] == [item["review_id"] for item in reverse]
    assert cue_conflict_id(records[0]) == "cueconflict-content-0-0-style-0-0-A_shape_B_texture"


def test_duplicate_ratings_are_rejected(tmp_path):
    path = tmp_path / "ratings.jsonl"
    row = {"review_id": "same", "rating": "pass"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    try:
        load_manual_ratings(path)
    except ValueError as error:
        assert "Duplicate review_id" in str(error)
    else:
        raise AssertionError("duplicate review ids must fail loudly")


def test_sweep_uses_fail_as_positive_and_stricter_tie_break(tmp_path):
    sampled = [
        {"review_id": "low", "ssim": 0.25, "degenerate": False},
        {"review_id": "high", "ssim": 0.75, "degenerate": False},
    ]
    frame = sweep_thresholds(
        sampled,
        {"low": "fail", "high": "pass"},
        np.asarray([0.3, 0.4]),
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    winner = frame.loc[frame.selected].iloc[0]
    assert winner.threshold == 0.4
    assert winner.fail_precision == 1.0
    assert winner.fail_recall == 1.0
    assert bool(winner.tie_break_invoked)
    assert (tmp_path / "cue_conflict_threshold_sweep.csv").exists()
    assert (tmp_path / "cue_conflict_threshold_sweep.png").exists()


def test_review_session_persists_and_resumes_without_duplicates(tmp_path):
    records = [
        {
            "review_id": f"review-{index}",
            "bucket_key": "pair:A_shape_B_texture",
            "ssim": 0.2 + index / 10,
            "image": torch.zeros(3, 4, 4),
        }
        for index in range(2)
    ]
    ratings_path = tmp_path / "ratings.jsonl"
    first = ReviewSession(records, np.asarray([0.2]), ratings_path, on_complete=lambda *_: None)
    assert first.current["review_id"] == "review-0"
    first._rate("pass")
    assert first.current["review_id"] == "review-1"

    resumed = ReviewSession(records, np.asarray([0.2]), ratings_path, on_complete=lambda *_: None)
    assert resumed.current["review_id"] == "review-1"
    resumed._rate("fail")
    rows = [json.loads(line) for line in ratings_path.read_text().splitlines()]
    assert [row["review_id"] for row in rows] == ["review-0", "review-1"]


def test_single_manual_class_falls_back_to_f1(tmp_path, capsys):
    sampled = [{"review_id": "only", "ssim": 0.1, "degenerate": False}]
    frame = sweep_thresholds(
        sampled,
        {"only": "fail"},
        np.asarray([0.2, 0.3]),
        results_dir=tmp_path,
        figures_dir=tmp_path,
    )
    assert frame.iloc[0].ranking_metric == "fail_f1"
    assert "kappa is undefined" in capsys.readouterr().out
