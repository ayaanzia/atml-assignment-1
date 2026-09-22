"""Validate saved Task 2/3 artifacts without training or dataset inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_state(state: dict | None) -> bool:
    if state is None:
        return True
    return all(bool(torch.isfinite(value).all()) for value in state.values()
               if isinstance(value, torch.Tensor))


def task2_validation() -> dict:
    results = ROOT / "task2" / "results"
    table = pd.read_csv(results / "main_comparison_table.csv").set_index("method")
    checks = {}
    warnings = []
    for method in ("source_only", "dan", "dann", "cdan"):
        checkpoint_path = results / "checkpoints" / f"{method}_seed6304.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        row = table.loc[method]
        source_match = math.isclose(float(checkpoint["mean_source_val_macro_f1"]),
                                    float(row.mean_source_f1), abs_tol=1e-12)
        checks[method] = {
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "all_model_tensors_finite": finite_state(checkpoint["model_state"]),
            "all_discriminator_tensors_finite": finite_state(checkpoint.get("discriminator_state")),
            "checkpoint_source_f1_matches_table": source_match,
        }
        curve = pd.read_csv(results / "training_curves" / f"{method}.csv")
        numeric = curve.select_dtypes("number")
        checks[method]["all_curve_values_finite"] = bool(numeric.map(math.isfinite).all().all())
        checks[method]["maximum_recorded_loss"] = float(curve[["classification_loss", "adaptation_loss"]].max().max())
        if float(row.mean_source_acc) < 0.3 and float(row.target_acc) < 0.1:
            warnings.append(f"{method}: collapsed accuracy profile")
        if checks[method]["maximum_recorded_loss"] > 1000:
            warnings.append(f"{method}: recorded loss exceeds 1000")
    return {"checks": checks, "warnings": warnings,
            "scope": "saved-state finiteness and cross-file consistency; not a reproducibility rerun"}


def task3_validation() -> dict:
    results = ROOT / "task3" / "results"
    table = pd.read_csv(results / "main_comparison_table.csv").set_index("method")
    checkpoint_paths = {
        "DAN-DG": results / "checkpoints" / "dan_dg_seed6304.pt",
        "SAM": results / "checkpoints" / "sam_seed6304.pt",
    }
    checks = {}
    warnings = []
    for method, checkpoint_path in checkpoint_paths.items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        row = table.loc[method]
        checks[method] = {
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "all_model_tensors_finite": finite_state(checkpoint["model_state"]),
            "checkpoint_source_f1_matches_table": math.isclose(
                float(checkpoint["mean_source_val_macro_f1"]), float(row.mean_source_f1), abs_tol=1e-12),
        }
        if float(row.mean_source_acc) < 0.3 and float(row.sketch_acc) < 0.1:
            warnings.append(f"{method}: collapsed accuracy profile")

    metadata = json.loads((results / "erm_checkpoint_metadata.json").read_text())
    source_checkpoint = ROOT / "task2" / "results" / "checkpoints" / "source_only_seed6304.pt"
    checks["ERM"] = {
        "task2_source_checkpoint_sha256": sha256(source_checkpoint),
        "hash_matches_recorded_metadata": sha256(source_checkpoint) == metadata["sha256"],
        "recorded_as_unchanged": bool(metadata["unchanged"]),
    }
    return {"checks": checks, "warnings": warnings,
            "scope": "saved-state finiteness and cross-file consistency; not a reproducibility rerun"}


def main() -> None:
    reports = {"task2": task2_validation(), "task3": task3_validation(),
               "model_training_or_dataset_inference_performed": False}
    for task in ("task2", "task3"):
        output = ROOT / task / "results" / "artifact_validation.json"
        output.write_text(json.dumps(reports[task], indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
