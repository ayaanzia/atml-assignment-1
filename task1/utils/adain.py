"""
Adaptive Instance Normalization (AdaIN) style transfer, following Huang &
Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance
Normalization" (ICCV 2017).

This module defines the standard architecture (VGG-19 encoder truncated at
relu4_1, AdaIN layer, mirrored decoder). Pretrained decoder/encoder weights
are NOT bundled with torchvision, so this wrapper supports loading a public
pretrained checkpoint (e.g. the widely-used naoto0804/pytorch-AdaIN weights,
`vgg_normalised.pth` + `decoder.pth`) from a local path. If no checkpoint is
available, the class can still be instantiated for interface/testing
purposes but `style_transfer` will raise, since an untrained decoder would
produce meaningless output (using it would silently violate the "explain
what your transformation does" requirement, so we fail loudly instead of
returning garbage stylizations).

Usage in the notebook:
    adain = AdaINStyleTransfer(vgg_weights_path=..., decoder_weights_path=...)
    stylized = adain.style_transfer(content_0_1, style_0_1, alpha=1.0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import DEVICE
except ImportError:
    from config import DEVICE

# ---------------------------------------------------------------------------
# Full VGG-19 "normalised" architecture (the variant used by the original
# AdaIN paper/repo, with reflection padding baked in before each conv).
# Public pretrained checkpoints (e.g. naoto0804/pytorch-AdaIN's
# vgg_normalized.pth) contain weights for the FULL network through relu5_4
# (53 sequential modules), not just the relu4_1 prefix the encoder actually
# uses — so this must be defined in full for load_state_dict to match all
# keys; the encoder is then built by truncating this at relu4_1 (index 30).
# ---------------------------------------------------------------------------
_VGG_ENCODER_LAYERS = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(3, 64, (3, 3)), nn.ReLU(),      # relu1_1 (idx 3)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),     # relu1_2 (idx 6)
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 128, (3, 3)), nn.ReLU(),    # relu2_1 (idx 10)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),   # relu2_2 (idx 13)
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 256, (3, 3)), nn.ReLU(),   # relu3_1 (idx 17)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),   # relu3_2 (idx 20)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),   # relu3_3 (idx 23)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),   # relu3_4 (idx 26)
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 512, (3, 3)), nn.ReLU(),   # relu4_1 (idx 30) <- encoder truncates here
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu4_2 (idx 33)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu4_3 (idx 36)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu4_4 (idx 39)
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_1 (idx 43)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_2 (idx 46)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_3 (idx 49)
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 512, (3, 3)), nn.ReLU(),   # relu5_4 (idx 52)
)
_RELU4_1_INDEX = 30  # inclusive end-slice for encoder truncation (unchanged)

_DECODER = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)), nn.Conv2d(64, 3, (3, 3)),
)


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Adaptive Instance Normalization: align content feature's per-channel
    mean/std to the style feature's, per-instance."""
    n, c = content_feat.shape[:2]
    c_mean = content_feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    c_std = content_feat.view(n, c, -1).std(dim=2).view(n, c, 1, 1) + eps
    s_mean = style_feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    s_std = style_feat.view(n, c, -1).std(dim=2).view(n, c, 1, 1) + eps
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean


class AdaINStyleTransfer(nn.Module):
    IMG_MEAN = (0.485, 0.456, 0.406)
    IMG_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        vgg_weights_path: Optional[str] = None,
        decoder_weights_path: Optional[str] = None,
    ):
        super().__init__()
        self.encoder = nn.Sequential(*list(_VGG_ENCODER_LAYERS.children())[: _RELU4_1_INDEX + 1])
        self.decoder = _DECODER
        self._weights_loaded = False
        if vgg_weights_path and Path(vgg_weights_path).exists():
            full_vgg = _VGG_ENCODER_LAYERS
            full_vgg.load_state_dict(torch.load(vgg_weights_path, map_location="cpu"))
            self.encoder = nn.Sequential(*list(full_vgg.children())[: _RELU4_1_INDEX + 1])
            self._weights_loaded = True
        if decoder_weights_path and Path(decoder_weights_path).exists():
            self.decoder.load_state_dict(torch.load(decoder_weights_path, map_location="cpu"))
        else:
            self._weights_loaded = False

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def _norm(self, x):
        mean = torch.tensor(self.IMG_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMG_STD, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _denorm(self, x):
        mean = torch.tensor(self.IMG_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMG_STD, device=x.device).view(1, 3, 1, 1)
        return (x * std + mean).clamp(0, 1)

    @torch.no_grad()
    def style_transfer(
        self,
        content_0_1: torch.Tensor,
        style_0_1: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        content_0_1, style_0_1: Tensor[N, 3, H, W] in [0, 1] (already resized
        to a common resolution, e.g. 224x224 or a working stylization size).
        alpha in [0, 1] interpolates between content-preserving (0) and full
        style transfer (1); use alpha=1.0 for maximal shape/texture
        cue-conflict per the spec.

        Raises RuntimeError if no pretrained decoder weights were loaded,
        since an untrained decoder cannot produce a meaningful stylization
        (fail loudly rather than silently returning noise).
        """
        if not self._weights_loaded:
            raise RuntimeError(
                "AdaIN decoder weights were not loaded. Download a public "
                "pretrained checkpoint (e.g. naoto0804/pytorch-AdaIN's "
                "vgg_normalised.pth and decoder.pth) and pass their paths to "
                "AdaINStyleTransfer(...). Refusing to stylize with an "
                "untrained decoder."
            )
        content_0_1 = content_0_1.to(DEVICE)
        style_0_1 = style_0_1.to(DEVICE)
        c_feat = self.encoder(self._norm(content_0_1))
        s_feat = self.encoder(self._norm(style_0_1))
        t = adain(c_feat, s_feat)
        t = alpha * t + (1 - alpha) * c_feat
        out = self.decoder(t)
        return self._denorm(out)
