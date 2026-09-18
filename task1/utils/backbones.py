"""
Frozen backbone wrappers: ResNet-50 (IMAGENET1K_V2), ViT-B/16 (IMAGENET1K_V1),
and OpenCLIP ViT-B/32 (pretrained='openai').

Each wrapper exposes a uniform interface:
    .normalize(images_0_1)         -> backbone-specific normalized tensor
    .forward_features(images_0_1)  -> Tensor[N, D] frozen representation
    .to(device) / .eval()          -> standard nn.Module semantics (frozen)

`images_0_1` is always a [N, 3, 224, 224] float tensor in [0, 1], already
resized/cropped and with any intervention applied — normalization is the
*last* step, applied inside the wrapper, never baked into cached images.
This is what guarantees pixel-identical inputs across backbones (Sec. 4 of
the spec).

CLIP additionally exposes `.zero_shot_logits` / `.zero_shot_predict` using the
prompt template "a photo of a {class}." and CLIP's learned logit scale.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .config import AMP_DTYPE, DEVICE, STL10_CLASSES, USE_CUDA, autocast_ctx


def _normalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


class FrozenBackbone(nn.Module):
    """Common base: freezes params, forces eval mode, provides feature dim."""

    name: str
    feature_dim: int

    def __init__(self):
        super().__init__()

    def _freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Hard override: this backbone must never leave eval mode (so that
        # BatchNorm running stats etc. are never perturbed), even if some
        # calling code calls .train() by mistake.
        return super().train(False)

    def normalize(self, images_0_1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def forward_features(self, images_0_1: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ResNet-50
# ---------------------------------------------------------------------------
class ResNet50Backbone(FrozenBackbone):
    name = "resnet50"
    feature_dim = 2048
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        net = torchvision.models.resnet50(weights=weights)
        # Everything up to (but excluding) the final `fc` layer, keeping the
        # adaptive-avg-pool so output is already [N, 2048, 1, 1].
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
            net.avgpool,
        )
        self._freeze()
        if USE_CUDA:
            self.stem = self.stem.to(memory_format=torch.channels_last)

    def normalize(self, images_0_1: torch.Tensor) -> torch.Tensor:
        return _normalize(images_0_1, self.IMAGENET_MEAN, self.IMAGENET_STD)

    @torch.no_grad()
    def forward_features(self, images_0_1: torch.Tensor) -> torch.Tensor:
        x = self.normalize(images_0_1).to(DEVICE)
        if USE_CUDA:
            x = x.to(memory_format=torch.channels_last)
        with autocast_ctx():
            feat = self.stem(x)
        feat = feat.flatten(1)  # [N, 2048]
        return feat.float()


# ---------------------------------------------------------------------------
# ViT-B/16
# ---------------------------------------------------------------------------
class ViTB16Backbone(FrozenBackbone):
    name = "vit_b_16"
    feature_dim = 768
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        self.net = torchvision.models.vit_b_16(weights=weights)
        self._freeze()

    def normalize(self, images_0_1: torch.Tensor) -> torch.Tensor:
        return _normalize(images_0_1, self.IMAGENET_MEAN, self.IMAGENET_STD)

    @torch.no_grad()
    def forward_features(self, images_0_1: torch.Tensor) -> torch.Tensor:
        x = self.normalize(images_0_1).to(DEVICE)
        with autocast_ctx():
            # Replicate torchvision's internal _process_input + encoder,
            # stopping before the classification head, so we get the raw
            # [CLS] token embedding (pre-logit).
            x = self.net._process_input(x)
            n = x.shape[0]
            cls_token = self.net.class_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = self.net.encoder(x)
            cls_out = x[:, 0]
        return cls_out.float()


# ---------------------------------------------------------------------------
# OpenCLIP ViT-B/32
# ---------------------------------------------------------------------------
class CLIPBackbone(FrozenBackbone):
    name = "clip_vit_b_32"
    feature_dim = 512  # ViT-B/32 openai image embedding dim

    def __init__(self, class_names: List[str] = STL10_CLASSES):
        super().__init__()
        import open_clip

        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # OpenAI CLIP's own normalization constants (NOT ImageNet stats).
        self.CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
        self.CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
        self._freeze()
        self.class_names = list(class_names)
        self._text_features = None  # lazily computed, cached

    def normalize(self, images_0_1: torch.Tensor) -> torch.Tensor:
        return _normalize(images_0_1, self.CLIP_MEAN, self.CLIP_STD)

    @torch.no_grad()
    def forward_features(self, images_0_1: torch.Tensor) -> torch.Tensor:
        """L2-normalized CLIP image embedding, as specified."""
        x = self.normalize(images_0_1).to(DEVICE)
        with autocast_ctx():
            feat = self.model.encode_image(x)
        feat = feat.float()
        feat = F.normalize(feat, dim=-1)
        return feat

    @torch.no_grad()
    def _get_text_features(self) -> torch.Tensor:
        if self._text_features is None:
            prompts = [f"a photo of a {c}." for c in self.class_names]
            tokens = self.tokenizer(prompts).to(DEVICE)
            with autocast_ctx():
                txt = self.model.encode_text(tokens)
            txt = F.normalize(txt.float(), dim=-1)
            self._text_features = txt
        return self._text_features

    @torch.no_grad()
    def zero_shot_logits(self, images_0_1: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between image and text (class prompt) embeddings,
        scaled by CLIP's learned logit scale (exp(logit_scale)). Returns raw
        (pre-softmax) logits of shape [N, num_classes].
        """
        img_feat = self.forward_features(images_0_1)  # already L2-normalized
        txt_feat = self._get_text_features()
        logit_scale = self.model.logit_scale.exp().float()
        logits = logit_scale * img_feat @ txt_feat.t()
        return logits

    @torch.no_grad()
    def zero_shot_predict(self, images_0_1: torch.Tensor):
        """Returns (pred_class_idx [N], confidence [N]) via softmax over the
        scaled cosine-similarity logits."""
        logits = self.zero_shot_logits(images_0_1)
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        return pred, conf


def build_backbones(device=DEVICE):
    """Instantiate all three frozen backbones and move to device."""
    resnet = ResNet50Backbone().to(device)
    vit = ViTB16Backbone().to(device)
    clip = CLIPBackbone().to(device)
    return {"resnet50": resnet, "vit_b_16": vit, "clip_vit_b_32": clip}
