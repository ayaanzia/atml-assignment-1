import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import data


def test_per_image_cache_does_not_retain_batch_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    features = torch.randn(1000, 512)
    data.save_cached_features("test", "clean", ["a", "b"], features[:2], [0, 1])
    paths = sorted(tmp_path.glob("*.pt"))
    assert len(paths) == 2
    assert all(path.stat().st_size < 20_000 for path in paths)
    assert torch.load(paths[0], weights_only=False)["features"].shape == (1, 512)
