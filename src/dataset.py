"""
dataset.py
──────────
Audio loading and Wav2Vec 2.0 embedding extraction
for the transfer learning cardiac classifier.
"""

import os
import numpy as np
import librosa
from pathlib import Path
from torch.utils.data import Dataset
import torch

SR       = 16000   # Wav2Vec 2.0 requires 16kHz
DURATION = 4       # pad/trim all clips to 4 seconds
VALID    = {"normal","murmur","extrastole","artifact","extrahls"}


# ── Dataset discovery ────────────────────────────────────────
def find_dataset(base: Path) -> dict:
    paths = {}
    for root, dirs, files in os.walk(base):
        root = Path(root)
        if len(root.relative_to(base).parts) > 4:
            continue
        for name in files:
            if name == "set_a.csv" and "set_a_csv" not in paths:
                paths["set_a_csv"] = root / name
            if name == "set_b.csv" and "set_b_csv" not in paths:
                paths["set_b_csv"] = root / name
        for d in dirs:
            if d == "set_a" and "set_a_dir" not in paths:
                paths["set_a_dir"] = root / d
            if d == "set_b" and "set_b_dir" not in paths:
                paths["set_b_dir"] = root / d
    missing = [k for k in ("set_a_csv","set_b_csv","set_a_dir","set_b_dir")
               if k not in paths]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}")
    return paths


def load_metadata(paths: dict):
    import pandas as pd
    df_a = pd.read_csv(paths["set_a_csv"])
    df_a["dataset"] = "A"
    df_a.columns    = [c.lower().strip() for c in df_a.columns]
    rows = []
    for f in paths["set_b_dir"].iterdir():
        if f.suffix != ".wav":
            continue
        prefix = f.name.split("_")[0].lower()
        if prefix in VALID:
            rows.append({"fname": f.name, "label": prefix, "dataset": "B"})
    df_b = pd.DataFrame(rows)
    df   = pd.concat([df_a, df_b], ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[~df["label"].isin(["nan","unlabeled",""])]
    return df


# ── Audio loading ─────────────────────────────────────────────
def load_audio(fname: str, dataset: str, paths: dict) -> np.ndarray | None:
    """
    Load audio, resample to 16kHz, pad/trim to DURATION seconds,
    and normalise to [-1, 1] as required by Wav2Vec 2.0.
    """
    folder = paths["set_a_dir"] if dataset == "A" else paths["set_b_dir"]
    path   = folder / Path(fname).name
    if not path.exists():
        return None
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        target = SR * DURATION
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        y = y / (np.abs(y).max() + 1e-8)   # normalise to [-1, 1]
        return y.astype(np.float32)
    except Exception:
        return None


# ── Wav2Vec embedding extraction ─────────────────────────────
def extract_embedding(audio: np.ndarray,
                      processor, backbone,
                      device: str = "cpu") -> np.ndarray:
    """
    Extract 1536-dim embedding from raw audio using frozen Wav2Vec.

    Strategy: Mean + Std pooling over time axis
        mean(T, 768) → (768,)  captures average acoustic content
        std(T, 768)  → (768,)  captures temporal variation
        cat          → (1536,) richer than mean alone

    Args:
        audio     : normalised float32 array (SR * DURATION,)
        processor : Wav2Vec2FeatureExtractor
        backbone  : Wav2Vec2Model (frozen)
        device    : 'cpu' or 'cuda'

    Returns:
        numpy float32 array of shape (1536,)
    """
    import torch
    inp = processor(audio, sampling_rate=SR,
                    return_tensors="pt", padding=True)
    with torch.no_grad():
        out    = backbone(inp.input_values.to(device))
        hidden = out.last_hidden_state.squeeze(0)   # (T, 768)
        emb    = torch.cat([hidden.mean(0),
                             hidden.std(0)])         # (1536,)
    return emb.cpu().numpy().astype(np.float32)


# ── PyTorch Dataset ──────────────────────────────────────────
class EmbeddingDataset(Dataset):
    """
    Dataset over pre-extracted Wav2Vec embeddings.
    Returns (embedding_tensor, label_tensor) per sample.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
