from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .thresholds import calibrate_threshold


def _numpy(x):
    return x.detach().float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def auroc_known_unknown(known_scores, unknown_scores) -> float:
    known, unknown = _numpy(known_scores), _numpy(unknown_scores)
    labels = np.r_[np.zeros(len(known)), np.ones(len(unknown))]
    return float(roc_auc_score(labels, np.r_[known, unknown]))


def evaluate_score(validation, known_test, near, far):
    threshold = calibrate_threshold(validation)
    all_unknown = torch.cat((near, far))
    known_acceptance = float((_numpy(known_test) <= threshold).mean())
    near_rejection = float((_numpy(near) > threshold).mean())
    far_rejection = float((_numpy(far) > threshold).mean())
    all_rejection = float((_numpy(all_unknown) > threshold).mean())
    return {
        "threshold": threshold,
        "near_auroc": auroc_known_unknown(known_test, near),
        "far_auroc": auroc_known_unknown(known_test, far),
        "all_auroc": auroc_known_unknown(known_test, all_unknown),
        "known_test_acceptance": known_acceptance,
        "near_rejection": near_rejection,
        "far_rejection": far_rejection,
        "all_rejection": all_rejection,
        "near_fpr_at_95_tpr": 1.0 - near_rejection,
        "far_fpr_at_95_tpr": 1.0 - far_rejection,
        "all_fpr_at_95_tpr": 1.0 - all_rejection,
    }
