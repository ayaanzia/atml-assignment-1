"""Human calibration tooling for the cue-conflict rejection threshold.

This module deliberately lives outside ``utils.metrics``: manual review and
threshold selection are experiment calibration steps, not report-time metrics.
The notebook owns the images; this module owns stable identifiers, reproducible
sampling, the resumable review UI, and the threshold sweep artifacts.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

try:  # repository-root imports (tests/scripts)
    from task1.utils.config import FIGURES_DIR, RESULTS_DIR, make_rng
except ModuleNotFoundError:  # notebook launched with task1/ as its working dir
    from utils.config import FIGURES_DIR, RESULTS_DIR, make_rng


def cue_conflict_id(record: Mapping) -> str:
    """Return the stable cache/review id for one stylized image."""
    required = ("content_id", "style_id", "direction")
    missing = [key for key in required if key not in record]
    if missing:
        raise KeyError(f"Cue-conflict record is missing stable-id fields: {missing}")
    return f"cueconflict-{record['content_id']}-{record['style_id']}-{record['direction']}"


def bucket_key(record: Mapping) -> str:
    """Return the required ``pair:direction`` calibration stratum."""
    if "bucket_key" in record:
        return str(record["bucket_key"])
    return f"{record['pair']}:{record['direction']}"


def _ssim(record: Mapping) -> float:
    if "ssim" in record:
        return float(record["ssim"])
    return float(record["ssim_to_content"])


def sample_for_review(
    accepted_records: list[dict],
    sample_frac: float = 0.10,
    seed_name: str = "cue_conflict_threshold_audit",
) -> list[dict]:
    """Draw a reproducible sample within every pair/direction bucket.

    Exact duplicate content/style/direction triples represent the same decoded
    image, so they are deduplicated before sampling. The returned ``review_id``
    never depends on list position and therefore survives re-thresholding.
    """
    if not 0 < sample_frac <= 1:
        raise ValueError("sample_frac must be in (0, 1]")

    unique: dict[str, dict] = {}
    for record in accepted_records:
        review_id = cue_conflict_id(record)
        prior = unique.get(review_id)
        if prior is not None and (
            bucket_key(prior) != bucket_key(record) or not np.isclose(_ssim(prior), _ssim(record))
        ):
            raise ValueError(f"Stable id collision with inconsistent metadata: {review_id}")
        unique.setdefault(review_id, record)

    by_bucket: dict[str, list[dict]] = {}
    for review_id, record in unique.items():
        by_bucket.setdefault(bucket_key(record), []).append(
            {
                "review_id": review_id,
                "bucket_key": bucket_key(record),
                "ssim": _ssim(record),
                "degenerate": bool(record.get("degenerate", False)),
                "image": record["image"],
            }
        )

    rng = make_rng(seed_name)
    sampled: list[dict] = []
    for key in sorted(by_bucket):
        records = sorted(by_bucket[key], key=lambda item: item["review_id"])
        count = max(1, int(round(len(records) * sample_frac)))
        indices = np.sort(rng.choice(len(records), size=min(count, len(records)), replace=False))
        sampled.extend(records[int(index)] for index in indices)
    return sampled


def plot_ssim_distribution(
    records: Sequence[Mapping],
    output_path: Path = FIGURES_DIR / "cue_conflict_ssim_distribution.png",
) -> dict[str, float]:
    """Save the pre-calibration SSIM histogram and return summary statistics."""
    values = np.asarray([_ssim(record) for record in records], dtype=float)
    if values.size == 0:
        raise ValueError("Cannot plot an empty SSIM distribution")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    axis.hist(values, bins=min(25, max(8, int(np.sqrt(values.size)))), color="#4C78A8", edgecolor="white")
    axis.axvline(0.20, color="#E45756", linestyle="--", label="old threshold = 0.20")
    axis.set(xlabel="SSIM to content", ylabel="Count", title="Cue-conflict SSIM distribution before calibration")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def load_manual_ratings(path: Path) -> dict[str, str]:
    """Load a JSONL ratings file, rejecting duplicate or malformed entries."""
    path = Path(path)
    if not path.exists():
        return {}
    ratings: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            review_id = item["review_id"]
            rating = item["rating"]
            if rating not in {"pass", "fail"}:
                raise ValueError(f"Invalid rating on line {line_number}: {rating!r}")
            if review_id in ratings:
                raise ValueError(f"Duplicate review_id on line {line_number}: {review_id}")
            ratings[review_id] = rating
    return ratings


def sweep_thresholds(
    sampled_records: list[dict],
    manual_ratings: dict[str, str],
    thresholds: np.ndarray,
    *,
    results_dir: Path = RESULTS_DIR,
    figures_dir: Path = FIGURES_DIR,
    tie_epsilon: float = 1e-9,
) -> pd.DataFrame:
    """Evaluate SSIM thresholds against manual ratings, with ``fail`` positive.

    Rows are ranked by Cohen's kappa, with ties broken toward the stricter
    (higher) threshold. If all manual ratings are identical, kappa is undefined;
    the function announces this and ranks by fail-class F1 instead.
    """
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0 or not np.isfinite(thresholds).all():
        raise ValueError("thresholds must be a non-empty finite 1D array")
    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError("thresholds must not contain duplicates")

    by_id = {record["review_id"]: record for record in sampled_records}
    missing = sorted(set(by_id) - set(manual_ratings))
    extra = sorted(set(manual_ratings) - set(by_id))
    if missing:
        raise ValueError(f"Manual ratings are incomplete; missing {len(missing)} review ids")
    if extra:
        raise ValueError(f"Ratings file contains {len(extra)} ids outside this sample")

    review_ids = sorted(by_id)
    truth = np.asarray([manual_ratings[review_id] for review_id in review_ids])
    single_manual_class = len(np.unique(truth)) == 1
    rows = []
    for threshold in thresholds:
        predicted = np.asarray([
            "fail" if by_id[review_id].get("degenerate", False) or by_id[review_id]["ssim"] < threshold else "pass"
            for review_id in review_ids
        ])
        agreement = float((predicted == truth).mean() * 100.0)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, predicted, labels=["fail"], average="binary", pos_label="fail", zero_division=0
        )
        kappa = float("nan") if single_manual_class else float(cohen_kappa_score(truth, predicted))
        rows.append({
            "threshold": float(threshold),
            "agreement_pct": agreement,
            "cohen_kappa": kappa,
            "fail_precision": float(precision),
            "fail_recall": float(recall),
            "fail_f1": float(f1),
            "manual_pass": int((truth == "pass").sum()),
            "manual_fail": int((truth == "fail").sum()),
            "predicted_fail": int((predicted == "fail").sum()),
        })

    frame = pd.DataFrame(rows)
    rank_column = "fail_f1" if single_manual_class else "cohen_kappa"
    best_value = float(frame[rank_column].max())
    tied = np.isclose(frame[rank_column], best_value, atol=tie_epsilon, rtol=0)
    winning_threshold = float(frame.loc[tied, "threshold"].max())
    tie_break_invoked = int(tied.sum()) > 1
    frame["selected"] = np.isclose(frame["threshold"], winning_threshold, atol=tie_epsilon, rtol=0)
    frame["ranking_metric"] = rank_column
    frame["tie_break_invoked"] = tie_break_invoked
    frame["sweep_min"] = float(thresholds.min())
    frame["sweep_max"] = float(thresholds.max())
    frame["sweep_count"] = int(thresholds.size)
    frame = frame.sort_values(
        [rank_column, "threshold"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)

    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_dir / "cue_conflict_threshold_sweep.csv", index=False)

    plot_frame = frame.sort_values("threshold")
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    if single_manual_class:
        axis.plot(plot_frame.threshold, plot_frame.fail_f1, marker="o", label="fail-class F1")
        axis.set_ylabel("Fail-class F1 (kappa undefined)")
    else:
        axis.plot(plot_frame.threshold, plot_frame.cohen_kappa, marker="o", label="Cohen's kappa")
        axis.set_ylabel("Cohen's kappa")
    axis.axvline(winning_threshold, color="#E45756", linestyle="--", label=f"selected = {winning_threshold:.3f}")
    axis.set_xlabel("SSIM threshold")
    axis.set_title("Manual cue-conflict threshold calibration")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figures_dir / "cue_conflict_threshold_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    winner = frame.loc[frame.selected].iloc[0]
    if single_manual_class:
        print("All manual ratings have one class; Cohen's kappa is undefined. Ranking by fail-class F1.")
    print(
        f"Selected threshold {winning_threshold:.3f}: "
        f"kappa={winner.cohen_kappa:.4f}, fail-F1={winner.fail_f1:.4f}. "
        f"Stricter-threshold tie-break {'invoked' if tie_break_invoked else 'not invoked'}."
    )
    return frame


def _tensor_to_png_bytes(image: torch.Tensor) -> bytes:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


class ReviewSession:
    """Resumable ipywidgets UI for one-at-a-time pass/fail ratings."""

    def __init__(
        self,
        sampled_records: list[dict],
        thresholds: np.ndarray,
        ratings_path: Path = RESULTS_DIR / "cue_conflict_manual_ratings.jsonl",
        *,
        on_complete: Callable[[list[dict], dict[str, str], np.ndarray], object] | None = None,
    ):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError("ReviewSession requires ipywidgets and IPython") from error

        review_ids = [record["review_id"] for record in sampled_records]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("sampled_records contains duplicate review_ids")
        self.records = sampled_records
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.ratings_path = Path(ratings_path)
        self.ratings_path.parent.mkdir(parents=True, exist_ok=True)
        self.ratings = load_manual_ratings(self.ratings_path)
        unknown = set(self.ratings) - set(review_ids)
        if unknown:
            raise ValueError(f"Ratings file contains ids not present in this sample: {sorted(unknown)[:3]}")
        self.on_complete = on_complete or (
            lambda records, ratings, values: sweep_thresholds(records, ratings, values)
        )
        self._widgets = widgets
        self._display = display
        self.progress = widgets.HTML()
        self.image = widgets.Image(format="png", width=448)
        self.note = widgets.Textarea(placeholder="Optional note", layout=widgets.Layout(width="448px", height="70px"))
        self.pass_button = widgets.Button(description="Pass", button_style="success")
        self.fail_button = widgets.Button(description="Fail", button_style="danger")
        self.pass_button.on_click(lambda _: self._rate("pass"))
        self.fail_button.on_click(lambda _: self._rate("fail"))
        self.output = widgets.Output()
        self.container = widgets.VBox([
            self.progress,
            self.image,
            self.note,
            widgets.HBox([self.pass_button, self.fail_button]),
            self.output,
        ])
        self._advance()

    def show(self):
        """Display the widget and return it for notebook composition/tests."""
        self._display(self.container)
        return self.container

    def _next_record(self) -> dict | None:
        return next((record for record in self.records if record["review_id"] not in self.ratings), None)

    def _advance(self) -> None:
        record = self._next_record()
        completed = len(self.ratings)
        total = len(self.records)
        if record is None:
            self.progress.value = f"<b>Complete: {completed} of {total} rated.</b>"
            self.image.value = b""
            self.note.value = ""
            self.pass_button.disabled = True
            self.fail_button.disabled = True
            with self.output:
                self.output.clear_output(wait=True)
                self.on_complete(self.records, self.ratings, self.thresholds)
            return
        self.current = record
        self.progress.value = (
            f"<b>{completed + 1} of {total}</b> &mdash; {record['bucket_key']} &mdash; "
            f"SSIM {record['ssim']:.3f}"
        )
        self.image.value = _tensor_to_png_bytes(record["image"])
        self.note.value = ""

    def _rate(self, rating: str) -> None:
        review_id = self.current["review_id"]
        if review_id in self.ratings:
            return
        item = {
            "review_id": review_id,
            "bucket_key": self.current["bucket_key"],
            "ssim": float(self.current["ssim"]),
            "rating": rating,
            "note": self.note.value.strip(),
            "rated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        with self.ratings_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.ratings[review_id] = rating
        self._advance()
