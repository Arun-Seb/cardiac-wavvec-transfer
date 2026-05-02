"""
test_model.py
─────────────
Unit tests for the Wav2Vec classification head.
Run with: pytest tests/
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import Wav2VecHead

EMBED_DIM  = 1536
N_CLASSES  = 5
BATCH_SIZE = 8


def make_batch(B=BATCH_SIZE):
    return torch.randn(B, EMBED_DIM)


class TestWav2VecHead:

    def test_output_shape(self):
        head = Wav2VecHead(EMBED_DIM, N_CLASSES)
        out  = head(make_batch())
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_single_sample(self):
        head = Wav2VecHead(EMBED_DIM, N_CLASSES)
        head.eval()
        out  = head(torch.randn(1, EMBED_DIM))
        assert out.shape == (1, N_CLASSES)

    def test_no_nan(self):
        head = Wav2VecHead(EMBED_DIM, N_CLASSES)
        out  = head(make_batch())
        assert not torch.isnan(out).any()

    def test_no_inf(self):
        head = Wav2VecHead(EMBED_DIM, N_CLASSES)
        out  = head(make_batch())
        assert not torch.isinf(out).any()

    def test_param_count(self):
        head  = Wav2VecHead(EMBED_DIM, N_CLASSES)
        count = head.param_count()
        # LayerNorm: 1536*2=3072, Linear: 1536*5+5=7685, total=10757
        assert count == 10_757, f"Unexpected param count: {count}"

    def test_trainable_params(self):
        head = Wav2VecHead(EMBED_DIM, N_CLASSES)
        for name, p in head.named_parameters():
            assert p.requires_grad, f"{name} should be trainable"

    def test_gradients_flow(self):
        head = Wav2VecHead(EMBED_DIM, N_CLASSES)
        out  = head(make_batch())
        out.sum().backward()
        for name, p in head.named_parameters():
            assert p.grad is not None, f"No gradient: {name}"

    def test_eval_mode_deterministic(self):
        head = Wav2VecHead(EMBED_DIM, N_CLASSES)
        head.eval()
        x = make_batch()
        with torch.no_grad():
            out1 = head(x)
            out2 = head(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_different_n_classes(self):
        for n in [2, 3, 10]:
            head = Wav2VecHead(EMBED_DIM, n)
            out  = head(make_batch())
            assert out.shape == (BATCH_SIZE, n)

    def test_different_embed_dim(self):
        head = Wav2VecHead(embed_dim=768, n_classes=N_CLASSES)
        out  = head(torch.randn(BATCH_SIZE, 768))
        assert out.shape == (BATCH_SIZE, N_CLASSES)

    def test_layer_norm_applied(self):
        """LayerNorm should normalise — zero-mean input stays controlled."""
        head  = Wav2VecHead(EMBED_DIM, N_CLASSES)
        large = torch.ones(1, EMBED_DIM) * 1000.0
        head.eval()
        with torch.no_grad():
            out = head(large)
        assert not torch.isnan(out).any(), "LayerNorm should prevent NaN on large input"
