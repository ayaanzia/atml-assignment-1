# Implementation Plan: Cue-Conflict Threshold Calibration + Notebook Fixes

**Target codebase:** `task1/` (notebook `task1.ipynb` + `utils/{config,data,metrics,transforms,backbones,adain}.py`)
**Goal:** Replace the fixed `SSIM_MIN_THRESHOLD = 0.20` rejection rule with a threshold chosen by
human-labeled calibration, propagate the resulting (possibly different) `accepted_records` through
every downstream step, fix the missing t-SNE cue-conflict panel, and add a caveat/calibration for
the CLIP-linear confidence anomaly. Phases are ordered by dependency — do not skip ahead.

---

## Phase 0 — Housekeeping (no dependencies, do first)

1. Delete the stale "this notebook could not be executed here" disclaimer paragraph in the markdown
   cell following the imports cell (currently claims cells "could not be executed," which contradicts
   the populated outputs throughout the rest of the notebook).

**Definition of done:** that markdown cell either removed or rewritten to describe the actual
run environment (GPU used, packages available, etc.).

---

## Phase 1 — Manual review tool

Create a new module `task1/analysis/cue_conflict_calibration.py`. Keep this logic out of
`metrics.py` — it's calibration tooling, not a metric used at report time.

### 1.1 Stratified sampling

```python
def sample_for_review(
    accepted_records: list[dict],
    sample_frac: float = 0.10,
    seed_name: str = "cue_conflict_threshold_audit",
) -> list[dict]:
    """
    Stratified sample across the 10 (pair, direction) buckets, ~sample_frac
    of each bucket, using make_rng(seed_name) for reproducibility.
    Returns a list of dicts: {review_id, bucket_key, ssim, image (tensor)}.
    review_id must be a STABLE identifier derived from content_id + style_id +
    direction (NOT a positional index into accepted_records — that index will
    change if accepted_records is regenerated after re-thresholding, and
    ratings keyed by position would silently become mislabeled).
    """
```

Also write a quick histogram of all 220 `ssim_to_content` values (already stored per-record) to
`results/figures/cue_conflict_ssim_distribution.png` before doing anything else — use it to decide
whether the 0.20→0.50 / 0.05-step sweep range still makes sense, or should be narrowed/shifted to
where the distribution actually has spread. Log the chosen sweep range in the results file from 1.2.

### 1.2 Review UI (ipywidgets, in-notebook)

Add a new notebook section "8b. Cue-Conflict Rejection Threshold Calibration" (insert after the
existing Step 3 cue-conflict section, before Step 4 Translation). Implement a `ReviewSession` class:

```python
class ReviewSession:
    """
    ipywidgets-based one-image-at-a-time rating tool.

    UX:
      - Shows the stylized image (the actual decoded RGB image, not just metadata).
      - Shows bucket_key and a running "N of M rated" progress label.
      - Two buttons: "Pass" (valid cue-conflict) / "Fail" (reject).
      - Optional short free-text note field (for later qualitative writeup /
        rubric refinement), not required to submit.
      - On every click: appends one line to results/cue_conflict_manual_ratings.jsonl
        immediately (do not batch in memory only) so a kernel crash/restart
        loses at most the in-flight image, and advances to the next unrated image.
      - On construction: loads any existing ratings file and skips review_ids
        already rated, so the session is resumable across kernel restarts.
      - When all sampled images are rated, displays a completion message and
        automatically calls the sweep function from Phase 2.
    """
```

Rating record schema (JSONL, one object per line):
```json
{"review_id": "...", "bucket_key": "cat-truck:A_shape_B_texture", "ssim": 0.31,
 "rating": "pass" | "fail", "note": "", "rated_at": "2026-09-22T12:00:00Z"}
```

**Rubric (write this into a markdown cell directly above the review widget, not just in code
comments — it needs to be visible in the final notebook for reproducibility/grading):**
State explicitly what counts as "fail" before rating starts, e.g.: content shape no longer
recognizable as its class; style texture entirely absent (indistinguishable from an unstyled
photo); image is muddy/degenerate enough that neither cue is interpretable. Agree on this wording
with the user before implementing the widget — don't invent it unilaterally.

**Definition of done:** running the review cell twice in a row (simulating an interrupted session)
resumes correctly and never re-shows an already-rated image; the JSONL file has one line per
rating with no duplicates.

---

## Phase 2 — Automated threshold selection

Still in `cue_conflict_calibration.py`:

```python
def sweep_thresholds(
    sampled_records: list[dict],      # from sample_for_review, with .ssim
    manual_ratings: dict[str, str],   # review_id -> "pass"/"fail"
    thresholds: np.ndarray,           # from the histogram-informed range chosen in 1.1
) -> pd.DataFrame:
    """
    For each threshold t:
      predicted = "pass" if (ssim >= t and not degenerate) else "fail"
      compute: agreement_pct, cohen_kappa, precision, recall, f1
        (precision/recall/f1 computed with "fail" as the positive class,
         since that's the rare, informative class — do NOT default to
         "pass" as positive or a trivial accept-everything threshold will
         score deceptively well on raw agreement alone).
    Returns one row per threshold, sorted by kappa descending.
    Ties (kappa within a small epsilon, e.g. 1e-9, of the max) are broken by
    preferring the STRICTER (higher) threshold — pre-register this rule here
    in code, don't decide it after seeing results.
    Saves the full table to results/cue_conflict_threshold_sweep.csv and a
    kappa-vs-threshold plot to results/figures/cue_conflict_threshold_sweep.png.
    """
```

Handle the degenerate case where every sampled image gets the same manual rating (kappa undefined,
`0/0`): fall back to reporting raw F1 only and flag this explicitly in the printed output — don't
silently divide by zero or silently fall back without telling the user.

**Definition of done:** running this on the completed ratings file prints the winning threshold,
its kappa/F1, and confirms the tie-break rule was or wasn't invoked; the CSV and plot exist.

---

## Phase 3 — Regenerate the cue-conflict dataset with the chosen threshold

1. Update `SSIM_MIN_THRESHOLD` in `metrics.py` to the value chosen in Phase 2. Keep the sweep
   result artifacts (CSV/plot) as the justification, referenced in the README.
2. **Cache invalidation fix (do this before rerunning, not after):** `data.py`'s feature cache
   keys cue-conflict features by the positional id `f"cueconflict-{i}"`. If `accepted_records` is
   regenerated with a different threshold, index `i` will point to a different underlying image
   than before, but stale cache files under `feat__*__cue_conflict__*` will still exist and will
   be silently reused (`load_cached_features` only checks existence, not content). Fix this by
   changing the id construction in the notebook's `CueConflictDataset` (and everywhere else that
   builds `f"cueconflict-{i}"`) to a content-derived id, e.g.:
   `f"cueconflict-{content_id}-{style_id}-{direction}"`
   This makes ids stable across regenerations and prevents silent cache collisions. Alternatively,
   simply delete all `results/cache/feat__*__cue_conflict__*.pt` files before rerunning — the
   content-derived id is the more robust long-term fix and should be preferred.
3. Bump `TARGET_PER_BUCKET` from 22 to ~30 as a buffer before rerunning, in case the calibrated
   threshold rejects more images than before and a bucket would otherwise fall short of ≥200 total
   / reasonable per-bucket balance.
4. Rerun the cue-conflict generation cells (pair sampling → AdaIN stylization → rejection check)
   with the new threshold and buffer. Confirm: total accepted ≥ 200, buckets still reasonably
   balanced, and log the new accept/reject counts per bucket (same format as before).

**Definition of done:** `results/cue_conflict_rejection_log.json` reflects non-trivial rejection
counts (i.e., not all buckets showing `rejected: 0` unless the calibration genuinely justified
that), and total accepted ≥ 200.

---

## Phase 4 — Propagate the new `accepted_records`

Rerun, in order, everything downstream of cue-conflict generation:
1. Shape/texture summary (`shape_bias_summary` per model) — regenerate the table.
2. Qualitative gallery — regenerate; consider deliberately including one example that the *old*
   threshold accepted but the *new* one would reject, as a labeled failure case for the report.
3. Cue-conflict representation stability ($I_T$) — rerun; pairing logic (content_id → clean
   feature) does not need to change, just re-execute against the refreshed `accepted_records`.

**Definition of done:** all three outputs reflect the Phase 3 `accepted_records`, not the original
220.

---

## Phase 5 — t-SNE cue-conflict panel + legend fix

Do this after Phase 4, so it plots the calibrated images, not the pre-calibration ones.

1. Add the cue-conflict branch to the t-SNE plotting loop, reusing the same content-image pairing
   already used for $I_T$:
   ```python
   if cond == "cue_conflict":
       cond_ids = [<content-derived ids from Phase 3.2>]
       cond_feats = get_cached_feats(backbone_name, cond, cond_ids)
       content_ids = [rec["content_id"] for rec in accepted_records]
       ref_feats = get_cached_feats(backbone_name, "clean_eval", content_ids)
       ref_labels = [rec["content_class_idx"] for rec in accepted_records]
       plot_tsne_panel(backbone_name, cond, ref_feats, cond_feats, ref_labels)
       continue
   ```
   Note `content_ids` will contain duplicates (content images are sampled with replacement) —
   this is expected, not a bug to fix.
2. Add `ax.legend(fontsize=6, markerscale=0.7)` inside `plot_tsne_panel` — currently `label=...` is
   set on some scatter calls but `legend()` is never invoked, so no class-color key renders in any
   of the existing panels either. Fix once, benefits all panels.

**Definition of done:** 4 backbones × 4 conditions... actually 3 backbones × 4 conditions = 12
t-SNE panels exist (previously 9), each with a visible legend.

---

## Phase 6 — CLIP-linear confidence caveat (independent — can run in parallel with Phases 1–5)

Pick one:
- **(a) Minimal:** no code change. Add one sentence to the report/README noting that
  `clip_vit_b_32_linear`'s raw mean-max-softmax-confidence isn't comparable to the other three
  backbones' because its input features are L2-normalized (per spec) and the linear head has no
  logit-scale analogous to CLIP's own `logit_scale.exp()` used in zero-shot — small logit
  magnitudes from a weight-decay-regularized fit on unit-norm features produce a naturally flatter
  softmax despite high accuracy.
- **(b) Supplementary calibration:** add a `temperature_scale(logits, labels)` helper that fits a
  single scalar $T$ by minimizing NLL on the validation logits/labels already available from
  `train_linear_head`'s history, apply it only when computing the reported confidence metric
  (never touching the trained head or its accuracy/predictions), and report both raw and
  calibrated confidence side by side in the clean-baseline table.

**Definition of done:** report/README contains the explanatory note either way; if (b), both
numbers appear in the baseline table with a one-line method note.

---

## Phase 7 — Final full re-run + verification

1. Restart kernel, run `task1.ipynb` top-to-bottom end to end. This is the real test of the
   Phase 3 cache-invalidation fix — if it's wrong, stale cue-conflict features will silently
   reappear here.
2. Confirm the final summary table (Step 12) and any interpretive text reflect the calibrated
   numbers, not the pre-calibration ones.
3. Save the calibration artifacts (`cue_conflict_ssim_distribution.png`,
   `cue_conflict_threshold_sweep.csv`, `cue_conflict_threshold_sweep.png`,
   `cue_conflict_manual_ratings.jsonl`) under `results/` so every reported number traces to a
   file, per the assignment's reproducibility requirement.

---

## Open decisions the agent should confirm with the user before implementing (not assume)

- Exact wording of the pass/fail rubric (Phase 1.2).
- Whether to do Phase 6(a) or 6(b).
- Whether the sweep range from Phase 1.1's histogram inspection should override the originally
  proposed 0.20→0.50 / 0.05-step range.
