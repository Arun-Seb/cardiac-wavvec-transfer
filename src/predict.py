"""
predict.py
──────────
Single-file inference for the Wav2Vec transfer learning classifier.
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from pathlib import Path

from src.dataset import SR, DURATION, extract_embedding


def predict(filepath: str, head, processor,
            backbone, le_classes: list,
            device: str = "cpu") -> tuple[str, dict]:
    """
    Predict cardiac condition from any .wav file.

    Args:
        filepath   : path to .wav file
        head       : trained Wav2VecHead
        processor  : Wav2Vec2FeatureExtractor
        backbone   : frozen Wav2Vec2Model
        le_classes : list of class name strings
        device     : 'cpu' or 'cuda'

    Returns:
        (predicted_class, probability_dict)
    """
    y, _ = librosa.load(filepath, sr=SR, duration=DURATION)
    target = SR * DURATION
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    y = (y / (np.abs(y).max() + 1e-8)).astype(np.float32)

    emb   = extract_embedding(y, processor, backbone, device)
    emb_t = torch.tensor(emb).unsqueeze(0).to(device)

    head.eval()
    with torch.no_grad():
        logits = head(emb_t)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred       = le_classes[np.argmax(probs)]
    probs_dict = dict(zip(le_classes, probs.tolist()))

    print(f"\n🩺  File   : {Path(filepath).name}")
    print(f"    Result : {pred.upper()}")
    print()
    for cls, p in sorted(probs_dict.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"    {cls:<14} {bar} {p:.1%}")

    return pred, probs_dict


def load_checkpoint(checkpoint_path: str,
                    device: str = "cpu"):
    """
    Restore head + backbone from a saved checkpoint.

    Usage:
        head, processor, backbone, classes = load_checkpoint("wav2vec_head.pt")
        pred, probs = predict("heartbeat.wav", head, processor, backbone, classes)
    """
    from src.model import Wav2VecHead, load_backbone

    ckpt       = torch.load(checkpoint_path, map_location=device)
    cfg        = ckpt["config"]
    le_classes = list(ckpt["le_classes"])

    processor, backbone = load_backbone(cfg["w2v_model"], device)

    head = Wav2VecHead(
        embed_dim = cfg["embed_dim"],
        n_classes = cfg["n_classes"],
    ).to(device)
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    print(f"✅ Checkpoint loaded from {checkpoint_path}")
    return head, processor, backbone, le_classes
