"""
model.py
────────
Lightweight classification head for frozen Wav2Vec 2.0 embeddings.

The full model is:
    Wav2Vec 2.0 backbone (94M params, FROZEN)
         ↓
    Mean + Std pooling → 1536-dim
         ↓
    Wav2VecHead (10,757 params, TRAINABLE)
         ↓
    N-class logits
"""

import torch
import torch.nn as nn


class Wav2VecHead(nn.Module):
    """
    Lightweight linear classification head.

    Sits on top of frozen Wav2Vec 2.0 embeddings.
    Only 10,757 parameters — 0.011% of the full model.

    Architecture:
        LayerNorm(embed_dim)   — stabilise embedding distribution
        Dropout(dropout)       — prevent overfitting (585 samples)
        Linear(embed_dim → n)  — map to class logits

    Why LayerNorm before the linear layer?
        Wav2Vec embeddings can have large variance across samples.
        LayerNorm centres and scales each embedding independently,
        making the linear layer's job much easier and training
        more stable.

    Args:
        embed_dim : input embedding dimension (default 1536 = 768*2)
        n_classes : number of output classes
        dropout   : dropout rate (default 0.3)
    """
    def __init__(self, embed_dim: int = 1536,
                 n_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        return self.linear(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_backbone(model_name: str = "facebook/wav2vec2-base",
                  device: str = "cpu"):
    """
    Load and freeze Wav2Vec 2.0 backbone.

    Returns:
        (processor, backbone) — both ready to use
    """
    import sys
    sys.modules.setdefault("torchvision", None)
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    backbone  = Wav2Vec2Model.from_pretrained(model_name).to(device)

    # Freeze ALL backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    frozen = sum(p.numel() for p in backbone.parameters())
    print(f"✅ Wav2Vec backbone loaded — {frozen:,} params FROZEN")
    return processor, backbone
