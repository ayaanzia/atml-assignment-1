import sys, os
sys.path.insert(0, os.path.abspath("."))

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image

from utils.config import (
    SEED, DEVICE, USE_CUDA, IMG_SIZE, STL10_CLASSES,
    RESULTS_DIR, CACHE_DIR, FIGURES_DIR,
    seed_everything, make_rng, autocast_ctx, save_json, load_json,
)
from utils import backbones as bb
from utils import transforms as tr
from utils import data as dm
from utils import metrics as mx
from utils.adain import AdaINStyleTransfer
from analysis import cue_conflict_calibration as cuecal

seed_everything(SEED)
print(f"Device: {DEVICE} | CUDA: {USE_CUDA} | classes: {STL10_CLASSES}")

backbone_dict = bb.build_backbones(device=DEVICE)
resnet, vit, clip = backbone_dict["resnet50"], backbone_dict["vit_b_16"], backbone_dict["clip_vit_b_32"]
for name, model in backbone_dict.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params/1e6:.1f}M params, feature_dim={model.feature_dim}, frozen={not any(p.requires_grad for p in model.parameters())}")

                                      
print([f"a photo of a {c}." for c in STL10_CLASSES])

train_ds, test_ds = dm.get_datasets(download=True)
print(f"STL-10 train: {len(train_ds)}, test: {len(test_ds)}")

train_subset, val_subset = dm.stratified_split(train_ds, train_frac=0.8, seed_name="stratified_split")
print(f"Stratified split -> train: {len(train_subset)}, val: {len(val_subset)}")

train_loader_raw = dm.make_loader(train_subset, batch_size=128, shuffle=False)
val_loader_raw = dm.make_loader(val_subset, batch_size=128, shuffle=False)

                                                                                    
linear_heads = {}
head_histories = {}

for name, model in backbone_dict.items():
    t0 = time.time()
    train_feat = dm.extract_features(model, train_loader_raw, condition="clean_train")
    val_feat = dm.extract_features(model, val_loader_raw, condition="clean_val")
    print(f"[{name}] feature extraction: {time.time()-t0:.1f}s "
          f"(train={train_feat['features'].shape}, val={val_feat['features'].shape})")

    head, history = dm.train_linear_head(
        train_feats=train_feat["features"], train_labels=train_feat["labels"],
        val_feats=val_feat["features"], val_labels=val_feat["labels"],
        in_features=model.feature_dim, num_classes=len(STL10_CLASSES),
        seed=SEED,
    )
    linear_heads[name] = head
    head_histories[name] = history
    print(f"[{name}] best val acc = {history['best_val_acc']:.4f} "
          f"(stopped after {history['epochs_run']} epochs)")

save_json({k: {kk: vv for kk, vv in v.items() if kk != 'train_loss'} for k, v in head_histories.items()},
          RESULTS_DIR / "linear_head_training_summary.json")

eval_subset = dm.select_eval_subset(test_ds, total=500, seed_name="eval_subset")
save_json(eval_subset, RESULTS_DIR / "eval_subset_ids.json")

print(f"Selected {eval_subset['total_selected']} images "
      f"(target {eval_subset['total_requested']}, {eval_subset['per_class_target']}/class)")
if eval_subset["shortfall_log"]:
    print("Shortfalls:", eval_subset["shortfall_log"])
else:
    print("No shortfalls — every class reached its target count.")

eval_image_ids = eval_subset["image_ids"]
eval_labels = eval_subset["labels"]
eval_indices = eval_subset["indices"]

from torch.utils.data import Subset

eval_ds_clean = Subset(test_ds, eval_indices)
eval_loader_clean = dm.make_loader(eval_ds_clean, batch_size=100, shuffle=False)

@torch.no_grad()
def run_head_over_loader(model, head, loader, condition, cache=True):
    feat_bundle = dm.extract_features(model, loader, condition=condition, cache=cache)
    feats = feat_bundle["features"].to(DEVICE)
    labels = feat_bundle["labels"]
    ids = feat_bundle["image_ids"]
    head.eval()
    with autocast_ctx():
        logits = head(feats)
    preds = logits.argmax(dim=-1).cpu().numpy().tolist()
    return {
        "preds": preds, "labels": labels, "image_ids": ids,
        "logits": logits.float().cpu(), "features": feat_bundle["features"],
    }


@torch.no_grad()
def run_clip_zero_shot_over_loader(clip_model, loader, condition):
    all_preds, all_labels, all_ids, all_conf = [], [], [], []
    for imgs, lbls, ids in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        pred, conf = clip_model.zero_shot_predict(imgs)
        all_preds.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())
        all_labels.extend([int(x) for x in lbls])
        all_ids.extend(list(ids))
    return {"preds": all_preds, "labels": all_labels, "image_ids": all_ids, "confidences": all_conf}

clean_results = {}
for name in ["resnet50", "vit_b_16"]:
    r = run_head_over_loader(backbone_dict[name], linear_heads[name], eval_loader_clean, condition="clean_eval")
    clean_results[name] = r

                                                                
clean_results["clip_vit_b_32_linear"] = run_head_over_loader(
    backbone_dict["clip_vit_b_32"], linear_heads["clip_vit_b_32"], eval_loader_clean, condition="clean_eval")
clean_results["clip_vit_b_32_zeroshot"] = run_clip_zero_shot_over_loader(
    backbone_dict["clip_vit_b_32"], eval_loader_clean, condition="clean_eval")

rows = []
for name, r in clean_results.items():
    acc = mx.accuracy(r["preds"], r["labels"])
    f1 = mx.macro_f1(r["preds"], r["labels"])
    if "logits" in r:
        conf = mx.mean_max_softmax_confidence(r["logits"])
    else:
        conf = float(np.mean(r["confidences"]))
    rows.append({"model": name, "accuracy": acc, "macro_f1": f1, "mean_max_confidence": conf})

clean_baseline_df = pd.DataFrame(rows)
clean_baseline_df.to_csv(RESULTS_DIR / "clean_baseline.csv", index=False)
clean_baseline_df

                                                                                            
clean_preds_by_model = {name: dict(zip(r["image_ids"], r["preds"])) for name, r in clean_results.items()}
save_json(clean_preds_by_model, RESULTS_DIR / "clean_predictions_per_image.json")

                                                                               
train_labels_all = np.array(train_ds.base.labels)
images_by_class = {}
loader_full_train = dm.make_loader(train_subset, batch_size=256, shuffle=False)
                                                                  
imgs_accum = {c: [] for c in range(len(STL10_CLASSES))}
for imgs, lbls, ids in loader_full_train:
    for img, lbl in zip(imgs, lbls):
        c = int(lbl)
        if len(imgs_accum[c]) < 200:                                                    
            imgs_accum[c].append(img)
images_by_class = {c: torch.stack(v) for c, v in imgs_accum.items() if len(v) > 0}

class_lab_stats = tr.compute_class_lab_stats(images_by_class)
color_swap_mapping = tr.make_class_derangement(len(STL10_CLASSES), seed_name="color_swap_pairing")
assert all(k != v for k, v in color_swap_mapping.items()), "derangement must have no fixed points"

save_json({STL10_CLASSES[k]: STL10_CLASSES[v] for k, v in color_swap_mapping.items()},
          RESULTS_DIR / "color_swap_mapping.json")
print("Color-swap donor mapping:", {STL10_CLASSES[k]: STL10_CLASSES[v] for k, v in color_swap_mapping.items()})

class GrayscaleDataset(torch.utils.data.Dataset):
    def __init__(self, base_subset):
        self.base = base_subset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, lbl, iid = self.base[i]
        return tr.to_grayscale3(img.unsqueeze(0)).squeeze(0), lbl, iid


class ColorSwapDataset(torch.utils.data.Dataset):
    def __init__(self, base_subset, mapping, class_lab_stats):
        self.base = base_subset
        self.mapping = mapping
        self.class_lab_stats = class_lab_stats
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, lbl, iid = self.base[i]
        out = tr.class_swap_color_transfer(img.unsqueeze(0), int(lbl), self.mapping, self.class_lab_stats)
        return out.squeeze(0), lbl, iid


gray_loader = dm.make_loader(GrayscaleDataset(eval_ds_clean), batch_size=100, shuffle=False)
swap_loader = dm.make_loader(ColorSwapDataset(eval_ds_clean, color_swap_mapping, class_lab_stats), batch_size=100, shuffle=False)

def evaluate_condition(condition_name, loader, include_clip_zeroshot=True):
    results = {}
    for name in ["resnet50", "vit_b_16"]:
        results[name] = run_head_over_loader(backbone_dict[name], linear_heads[name], loader, condition=condition_name)
    results["clip_vit_b_32_linear"] = run_head_over_loader(
        backbone_dict["clip_vit_b_32"], linear_heads["clip_vit_b_32"], loader, condition=condition_name)
    if include_clip_zeroshot:
        results["clip_vit_b_32_zeroshot"] = run_clip_zero_shot_over_loader(
            backbone_dict["clip_vit_b_32"], loader, condition=condition_name)
    return results


def summarize_vs_clean(condition_name, condition_results, clean_results):
    rows = []
    for name, r in condition_results.items():
        acc = mx.accuracy(r["preds"], r["labels"])
        clean_acc = mx.accuracy(clean_results[name]["preds"], clean_results[name]["labels"])
                                                                     
        clean_map = dict(zip(clean_results[name]["image_ids"], clean_results[name]["preds"]))
        aligned_clean_preds = [clean_map[i] for i in r["image_ids"]]
        consistency = mx.prediction_consistency(aligned_clean_preds, r["preds"])
        rows.append({
            "condition": condition_name, "model": name,
            "accuracy": acc, "accuracy_delta_vs_clean": acc - clean_acc,
            "consistency_vs_clean": consistency,
        })
    return rows

gray_results = evaluate_condition("grayscale", gray_loader)
swap_results = evaluate_condition("color_swap", swap_loader)

color_bias_rows = (
    summarize_vs_clean("grayscale", gray_results, clean_results)
    + summarize_vs_clean("color_swap", swap_results, clean_results)
)
color_bias_df = pd.DataFrame(color_bias_rows)
color_bias_df.to_csv(RESULTS_DIR / "color_bias_results.csv", index=False)
color_bias_df

                                                                        
                                                                         
                                                                       
VGG_WEIGHTS_PATH = "models/vgg_normalised.pth"                    
DECODER_WEIGHTS_PATH = "models/decoder.pth"                        

adain_model = AdaINStyleTransfer(
    vgg_weights_path=VGG_WEIGHTS_PATH,
    decoder_weights_path=DECODER_WEIGHTS_PATH,
).to(DEVICE)

CUE_CONFLICT_PAIRS = [
    ("cat", "truck"),
    ("bird", "ship"),
    ("dog", "car"),
    ("horse", "airplane"),
    ("monkey", "deer"),
]
TARGET_PER_BUCKET = 30                                                                    
class_to_idx = {c: i for i, c in enumerate(STL10_CLASSES)}

                                                                         
                                                                       
                                                                    
eval_imgs_by_class = {c: [] for c in range(len(STL10_CLASSES))}
for imgs, lbls, ids in eval_loader_clean:
    for img, lbl, iid in zip(imgs, lbls, ids):
        eval_imgs_by_class[int(lbl)].append((img, iid))

for c, items in eval_imgs_by_class.items():
    print(STL10_CLASSES[c], len(items))

rng_cue = make_rng("cue_conflict_sampling")
candidate_records = []                                                                                
candidate_dir = RESULTS_DIR / "cue_conflict_candidates"
candidate_image_dir = candidate_dir / "images"
candidate_image_dir.mkdir(parents=True, exist_ok=True)

def sample_pairs(pool_a, pool_b, n, rng):
                                                                 
    combinations = [(a, b) for a in pool_a for b in pool_b]
    if n > len(combinations):
        raise ValueError(f"Requested {n} pairs from only {len(combinations)} unique combinations")
    indices = rng.choice(len(combinations), size=n, replace=False)
    return [combinations[i] for i in indices]

for (cls_a, cls_b) in CUE_CONFLICT_PAIRS:
    ia, ib = class_to_idx[cls_a], class_to_idx[cls_b]
    for direction, (content_cls, style_cls) in [("A_shape_B_texture", (ia, ib)), ("B_shape_A_texture", (ib, ia))]:
        bucket_key = f"{cls_a}-{cls_b}:{direction}"
        pairs = sample_pairs(eval_imgs_by_class[content_cls], eval_imgs_by_class[style_cls], TARGET_PER_BUCKET, rng_cue)
        for (content_img, content_id), (style_img, style_id) in pairs:
            with torch.no_grad():
                stylized = adain_model.style_transfer(content_img.unsqueeze(0), style_img.unsqueeze(0), alpha=1.0)
            content_np = content_img.permute(1, 2, 0).cpu().numpy()
            stylized_np = stylized.squeeze(0).permute(1, 2, 0).cpu().numpy()
            check = mx.is_valid_cue_conflict(content_np, stylized_np)
            record = {
                "pair": f"{cls_a}-{cls_b}", "direction": direction, "bucket_key": bucket_key,
                "content_class": STL10_CLASSES[content_cls], "texture_class": STL10_CLASSES[style_cls],
                "content_class_idx": content_cls, "texture_class_idx": style_cls,
                "content_id": content_id, "style_id": style_id,
                "image": stylized.squeeze(0).cpu(), "ssim": check["ssim_to_content"],
                "pixel_std": check["pixel_std"], "degenerate": check["degenerate"],
            }
            candidate_records.append(record)

                                                                    
candidate_rows = []
for rec in candidate_records:
    cue_id = cuecal.cue_conflict_id(rec)
    image_name = f"{cue_id}.png"
    image_np = (rec["image"].permute(1, 2, 0).numpy().clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(image_np).save(candidate_image_dir / image_name)
    candidate_rows.append({
        "cue_id": cue_id, "image_path": f"images/{image_name}",
        **{k: v for k, v in rec.items() if k != "image"},
    })
candidate_df = pd.DataFrame(candidate_rows)
candidate_manifest_path = candidate_dir / "candidates.csv"
if candidate_manifest_path.exists():
                                                                                   
    existing_candidate_df = pd.read_csv(candidate_manifest_path)
    candidate_df = pd.concat([existing_candidate_df, candidate_df], ignore_index=True)
    candidate_df = candidate_df.drop_duplicates(subset="cue_id", keep="first")
assert candidate_df["cue_id"].is_unique, "Candidate IDs must be unique before review"
candidate_df.to_csv(candidate_manifest_path, index=False)
print(f"Preserved/wrote {len(candidate_df)} blind-review candidates in {candidate_dir}")

                                                                           
approved_path = RESULTS_DIR / "cue_conflict_approved_manifest.csv"
if not approved_path.exists():
    raise RuntimeError(
        "Review candidates with `python task1/cue_review_app/app.py`, export the "
        "balanced approved manifest, then rerun this cell and the cells below."
    )
approved_df = pd.read_csv(approved_path)
approved_ids = approved_df["cue_id"].tolist()
candidate_ids = set(candidate_df["cue_id"])
missing_ids = sorted(set(approved_ids).difference(candidate_ids))
assert not missing_ids, f"Approved manifest contains unknown/stale candidates: {missing_ids[:3]}"
accepted_records = []
for row in approved_df.to_dict(orient="records"):
    image_path = candidate_dir / row.pop("image_path")
    row.pop("cue_id")
    image_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    row["image"] = torch.from_numpy(image_np).permute(2, 0, 1).contiguous()
    accepted_records.append(row)
assert len(accepted_records) == 200 and len(set(approved_ids)) == 200
assert approved_df.groupby("bucket_key").size().eq(20).all()
approved_df.to_csv(RESULTS_DIR / "cue_conflict_manifest.csv", index=False)
generated_counts = candidate_df.groupby("bucket_key").size().to_dict()
selection_log = {bucket: {"accepted": 20, "rejected_or_unselected": int(generated_counts[bucket]) - 20,
                          "generated": int(generated_counts[bucket])}
                 for bucket in approved_df["bucket_key"].unique()}
save_json(selection_log, RESULTS_DIR / "cue_conflict_rejection_log.json")
print("Loaded 200 manually approved, exactly balanced cue-conflict images.")

class CueConflictDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records
    def __len__(self):
        return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
                                                                          
                                                                          
                                                                    
                                                                        
                                                             
        return r["image"], r["content_class_idx"], cuecal.cue_conflict_id(r)

cue_ds = CueConflictDataset(accepted_records)
cue_loader = dm.make_loader(cue_ds, batch_size=100, shuffle=False)
cue_results_by_model = evaluate_condition("cue_conflict", cue_loader)

shape_texture_rows = []
for name, r in cue_results_by_model.items():
    id_to_pred = dict(zip(r["image_ids"], r["preds"]))
    decisions_by_pair = {}
    for i, rec in enumerate(accepted_records):
        iid = cuecal.cue_conflict_id(rec)
        pred = id_to_pred[iid]
        decision = mx.classify_shape_texture(pred, rec["content_class_idx"], rec["texture_class_idx"])
        decisions_by_pair.setdefault(rec["pair"], []).append(decision)
        decisions_by_pair.setdefault("__all__", []).append(decision)
    for pair_key, decisions in decisions_by_pair.items():
        summary = mx.shape_bias_summary(decisions)
        shape_texture_rows.append({"model": name, "pair": pair_key, **summary})

shape_texture_df = pd.DataFrame(shape_texture_rows)
shape_texture_df.to_csv(RESULTS_DIR / "shape_texture_results.csv", index=False)
shape_texture_df[shape_texture_df["pair"] == "__all__"]

def show_cue_conflict_gallery(records, results_by_model, n=6):
                                                                          
                                                                           
                                                  
    pred_maps = {name: dict(zip(r["image_ids"], r["preds"])) for name, r in results_by_model.items()}
    categorized = []
    for i, rec in enumerate(records):
        iid = cuecal.cue_conflict_id(rec)
        decisions = {name: mx.classify_shape_texture(preds[iid], rec["content_class_idx"], rec["texture_class_idx"])
                     for name, preds in pred_maps.items()}
        values = list(decisions.values())
        if all(value == "shape" for value in values): category = "all shape"
        elif all(value == "texture" for value in values): category = "all texture"
        elif all(value == "other" for value in values): category = "all other"
        elif decisions["clip_vit_b_32_linear"] != decisions["clip_vit_b_32_zeroshot"]: category = "CLIP head/zero-shot disagree"
        elif decisions["resnet50"] == "texture" and decisions["vit_b_16"] == "shape": category = "ResNet texture / ViT shape"
        else: category = "mixed decisions"
        categorized.append((category, iid, i))
    priority = ["all shape", "all texture", "ResNet texture / ViT shape",
                "CLIP head/zero-shot disagree", "all other", "mixed decisions"]
    pick, used = [], set()
    for category in priority:
        matches = sorted((iid, i) for cat, iid, i in categorized if cat == category and i not in used)
        if matches and len(pick) < n:
            iid, i = matches[0]; pick.append((category, i)); used.add(i)
    for category, iid, i in sorted(categorized, key=lambda item: item[1]):
        if len(pick) >= n: break
        if i not in used: pick.append((category, i)); used.add(i)
    fig, axes = plt.subplots(1, len(pick), figsize=(3 * len(pick), 3.5))
    if len(pick) == 1:
        axes = [axes]
    for ax, (category, i) in zip(axes, pick):
        rec = records[i]
        img = rec["image"].permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.axis("off")
        iid = cuecal.cue_conflict_id(rec)
        title_lines = [category, f"shape={rec['content_class']} / tex={rec['texture_class']}"]
        for name, r in results_by_model.items():
            pred_idx = dict(zip(r["image_ids"], r["preds"]))[iid]
            title_lines.append(f"{name}: {STL10_CLASSES[pred_idx]}")
        ax.set_title("\n".join(title_lines), fontsize=7)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cue_conflict_gallery.png", dpi=150)
    plt.show()

show_cue_conflict_gallery(accepted_records, cue_results_by_model)

calibration_config_path = RESULTS_DIR / "cue_conflict_calibration_config.json"
completed_sweep_path = RESULTS_DIR / "cue_conflict_threshold_sweep.csv"
if completed_sweep_path.exists() and calibration_config_path.exists():
                                                                        
                                                              
    calibration_config = load_json(calibration_config_path)
    ssim_summary = calibration_config["ssim_summary"]
    SWEEP_MIN = calibration_config["sweep_min"]
    SWEEP_MAX = calibration_config["sweep_max"]
    SWEEP_STEP = calibration_config["sweep_step"]
    SWEEP_THRESHOLDS = np.asarray(calibration_config["thresholds"], dtype=float)
    print("Using completed calibration config:", calibration_config)
else:
    ssim_summary = cuecal.plot_ssim_distribution(candidate_records)
    print("SSIM distribution summary:", ssim_summary)
                                                                          
                                                                           
    SWEEP_MIN = 0.20
    SWEEP_MAX = 0.50
    SWEEP_STEP = 0.05
    SWEEP_THRESHOLDS = np.round(
        np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_STEP / 2, SWEEP_STEP), 10
    )
    save_json(
        {
            "ssim_summary": ssim_summary,
            "sweep_min": SWEEP_MIN,
            "sweep_max": SWEEP_MAX,
            "sweep_step": SWEEP_STEP,
            "thresholds": SWEEP_THRESHOLDS.tolist(),
            "range_decision": "Retained the preregistered 0.20--0.50 range: the distribution spans this interval and its lower tail is decision-relevant.",
            "tie_break": "kappa ties within 1e-9 select the stricter (higher) threshold",
            "positive_class": "fail",
        },
        calibration_config_path,
    )
print("Sweep thresholds:", SWEEP_THRESHOLDS)

sweep_path = RESULTS_DIR / "cue_conflict_threshold_sweep.csv"
if sweep_path.exists():
                                                                     
                                                               
    completed_sweep = pd.read_csv(sweep_path)
    selected = completed_sweep.loc[completed_sweep["selected"].astype(str).str.lower() == "true"]
    print("Calibration already complete; selected row:")
    display(selected)
else:
    calibration_sample = cuecal.sample_for_review(
        candidate_records,
        sample_frac=0.10,
        seed_name="cue_conflict_threshold_audit",
    )
    print(
        f"Manual review sample: {len(calibration_sample)} unique images across "
        f"{len({record['bucket_key'] for record in calibration_sample})} buckets"
    )
    review_session = cuecal.ReviewSession(
        calibration_sample,
        thresholds=SWEEP_THRESHOLDS,
        ratings_path=RESULTS_DIR / "cue_conflict_manual_ratings.jsonl",
    )
    review_session.show()

translation_rows = []
delta_values = [0, 8, 16, 32]

class TranslateDataset(torch.utils.data.Dataset):
    def __init__(self, base_subset, delta, direction):
        self.base = base_subset
        self.delta = delta
        self.direction = direction
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, lbl, iid = self.base[i]
        if self.delta == 0:
            out = img
        else:
            out = tr.translate_reflect(img.unsqueeze(0), self.delta, self.direction).squeeze(0)
        return out, lbl, iid

for delta in delta_values:
    directions = [tr.DIRECTIONS[0]] if delta == 0 else list(tr.DIRECTIONS)
    per_direction_metrics = {name: {"acc": [], "cons": []} for name in
                              ["resnet50", "vit_b_16", "clip_vit_b_32_linear", "clip_vit_b_32_zeroshot"]}
    for direction in directions:
        loader = dm.make_loader(TranslateDataset(eval_ds_clean, delta, direction), batch_size=100, shuffle=False)
        cond_name = f"translate_d{delta}_{direction}"
        results = evaluate_condition(cond_name, loader)
        for name, r in results.items():
            acc = mx.accuracy(r["preds"], r["labels"])
            clean_map = dict(zip(clean_results[name]["image_ids"], clean_results[name]["preds"]))
            aligned_clean_preds = [clean_map[i] for i in r["image_ids"]]
            cons = mx.prediction_consistency(aligned_clean_preds, r["preds"])
            per_direction_metrics[name]["acc"].append(acc)
            per_direction_metrics[name]["cons"].append(cons)
    for name, m in per_direction_metrics.items():
        translation_rows.append({
            "delta": delta, "model": name,
            "accuracy": float(np.mean(m["acc"])), "consistency_vs_clean": float(np.mean(m["cons"])),
        })

translation_df = pd.DataFrame(translation_rows)
translation_df.to_csv(RESULTS_DIR / "translation_results.csv", index=False)
translation_df

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in translation_df["model"].unique():
    sub = translation_df[translation_df["model"] == name].sort_values("delta")
    axes[0].plot(sub["delta"], sub["accuracy"], marker="o", label=name)
    axes[1].plot(sub["delta"], sub["consistency_vs_clean"], marker="o", label=name)
axes[0].set_xlabel("Translation δ (px)"); axes[0].set_ylabel("Accuracy"); axes[0].set_title("Accuracy vs. δ")
axes[1].set_xlabel("Translation δ (px)"); axes[1].set_ylabel("Consistency vs. clean"); axes[1].set_title("Consistency vs. δ")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(FIGURES_DIR / "translation_curves.png", dpi=150)
plt.show()

patch_permutations = {iid: tr.make_patch_permutation(iid, grid=4).tolist() for iid in eval_image_ids}
save_json(patch_permutations, RESULTS_DIR / "patch_permutations.json")

class PatchShuffleDataset(torch.utils.data.Dataset):
    def __init__(self, base_subset):
        self.base = base_subset
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, lbl, iid = self.base[i]
        perm = np.array(patch_permutations[iid])
        out = tr.patch_shuffle(img, perm, grid=4)
        return out, lbl, iid

shuffle_loader = dm.make_loader(PatchShuffleDataset(eval_ds_clean), batch_size=100, shuffle=False)
shuffle_results = evaluate_condition("patch_shuffle", shuffle_loader)
patch_shuffle_rows = summarize_vs_clean("patch_shuffle", shuffle_results, clean_results)
patch_shuffle_df = pd.DataFrame(patch_shuffle_rows)
patch_shuffle_df.to_csv(RESULTS_DIR / "patch_shuffle_results.csv", index=False)
patch_shuffle_df

REPR_CONDITIONS = ["grayscale", "cue_conflict", "translate_d32_up", "patch_shuffle"]
                                                                    
                                                                 
                                                                          
                                                                        
                                                                          
                                                                         
                                                              

repr_backbones = {"resnet50": resnet, "vit_b_16": vit, "clip_vit_b_32": clip}

def get_cached_feats(backbone_name, condition, image_ids):
    cache_name = repr_backbones[backbone_name].cache_name
    cached = dm.load_cached_features(cache_name, condition, image_ids)
    assert cached is not None, f"Expected cached features for {backbone_name}/{condition}; run the corresponding step above first."
    id_to_row = {iid: i for i, iid in enumerate(cached["image_ids"])}
    order = [id_to_row[iid] for iid in image_ids]
    return cached["features"][order]

I_T_rows = []
for backbone_name in repr_backbones:
    clean_feats = get_cached_feats(backbone_name, "clean_eval", eval_image_ids)
    for cond in REPR_CONDITIONS:
        if cond == "cue_conflict":
            cond_ids = [cuecal.cue_conflict_id(rec) for rec in accepted_records]
            cond_feats = get_cached_feats(backbone_name, cond, cond_ids)
                                                                        
                                                                          
                                                                       
                                                                   
            content_ids = [rec["content_id"] for rec in accepted_records]
            ref_feats = get_cached_feats(backbone_name, "clean_eval", content_ids)
        else:
            cond_feats = get_cached_feats(backbone_name, cond, eval_image_ids)
            ref_feats = clean_feats
        i_t = mx.representation_stability(ref_feats, cond_feats)
        I_T_rows.append({"backbone": backbone_name, "condition": cond, "I_T": i_t})

I_T_df = pd.DataFrame(I_T_rows)
I_T_df.to_csv(RESULTS_DIR / "representation_stability.csv", index=False)
I_T_df.pivot(index="backbone", columns="condition", values="I_T")

def plot_tsne_panel(backbone_name, condition, clean_feats, cond_feats, clean_labels):
    combined = torch.cat([clean_feats, cond_feats], dim=0).numpy()
    proj = mx.fit_2d_projection(combined, method="tsne", seed_name="tsne", perplexity=30)
    n = clean_feats.shape[0]
    proj_clean, proj_cond = proj[:n], proj[n:]

    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab10")
    for c in range(len(STL10_CLASSES)):
        mask = np.array(clean_labels) == c
        ax.scatter(proj_clean[mask, 0], proj_clean[mask, 1], color=cmap(c), marker="o", s=18,
                   label=STL10_CLASSES[c],
                   alpha=0.8)
        ax.scatter(proj_cond[mask, 0], proj_cond[mask, 1], color=cmap(c), marker="x", s=18, alpha=0.8)
    ax.legend(fontsize=6, markerscale=0.7)
    ax.set_title(f"{backbone_name} — clean (o) vs {condition} (x)\n[independent fit; coords not comparable across panels]", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"tsne_{backbone_name}_{condition}.png", dpi=150)
    plt.show()

for backbone_name in repr_backbones:
    clean_feats = get_cached_feats(backbone_name, "clean_eval", eval_image_ids)
    for cond in REPR_CONDITIONS:
        if cond == "cue_conflict":
            cond_ids = [cuecal.cue_conflict_id(rec) for rec in accepted_records]
            cond_feats = get_cached_feats(backbone_name, cond, cond_ids)
            content_ids = [rec["content_id"] for rec in accepted_records]
            ref_feats = get_cached_feats(backbone_name, "clean_eval", content_ids)
            ref_labels = [rec["content_class_idx"] for rec in accepted_records]
            plot_tsne_panel(backbone_name, cond, ref_feats, cond_feats, ref_labels)
            continue
        cond_feats = get_cached_feats(backbone_name, cond, eval_image_ids)
        plot_tsne_panel(backbone_name, cond, clean_feats, cond_feats, eval_labels)

summary_rows = []
summary_rows += [{"condition": "clean", "model": r["model"], "accuracy": r["accuracy"], "consistency_vs_clean": 1.0}
                  for r in clean_baseline_df.to_dict("records")]
summary_rows += color_bias_df[["condition", "model", "accuracy", "consistency_vs_clean"]].to_dict("records")
summary_rows += patch_shuffle_df[["condition", "model", "accuracy", "consistency_vs_clean"]].to_dict("records")
summary_rows += [{"condition": f"translate_d{d}", "model": m, "accuracy": a, "consistency_vs_clean": c}
                  for d, m, a, c in translation_df[["delta", "model", "accuracy", "consistency_vs_clean"]].values]

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "all_conditions_summary.csv", index=False)
summary_df.pivot_table(index="model", columns="condition", values="accuracy")
