# ============================================================
#  MULTIMODAL CARDIAC CLASSIFIER — OPTION 3
#  Fine-tune Wav2Vec 2.0 Classification Head (Frozen Backbone)
#
#  Architecture:
#    Raw audio → Wav2Vec 2.0 (FROZEN, 90M params)
#             → Mean + Std pooling → 1536-dim
#             → Linear head (TRAINABLE, ~7,685 params)
#             → 5-class Softmax
#
#  Why this works:
#    Wav2Vec was pretrained on 960h of speech audio.
#    Heartbeat sounds share low-level acoustic properties
#    with speech — rhythm, periodicity, spectral texture.
#    We freeze the backbone and only train the tiny
#    classification head → fast on CPU, low overfitting risk.
#
#  Dataset: https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds
#  Paste into ONE Jupyter cell and run.
# ============================================================

# ── STEP 1: Install dependencies ────────────────────────────
import sys, subprocess
pkgs = [
    "librosa", "scikit-learn", "pandas", "matplotlib",
    "seaborn", "numpy", "soundfile", "torch",
    "transformers==4.35.2",
]
subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs + ["-q"])
print("✅ Packages installed")

# ── STEP 2: Imports ─────────────────────────────────────────
import os, sys
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.modules.setdefault("torchvision", None)
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

import warnings
warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Imports done  |  Device: {DEVICE}")

# ── STEP 3: Config ───────────────────────────────────────────
DATASET_DIR  = Path(r"D:\Heart Beat Sound")
SR           = 16000       # Wav2Vec requires 16kHz
DURATION     = 4           # seconds
W2V_DIM      = 768         # Wav2Vec hidden size
EMBED_DIM    = W2V_DIM * 2 # mean + std pooling → 1536
BATCH_SIZE   = 8           # small batch — raw audio is large
EPOCHS       = 30
LR           = 1e-3        # high LR is fine — only head trains
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── STEP 4: Auto-detect dataset ─────────────────────────────
def find_dataset(base):
    paths = {}
    for root, dirs, files in os.walk(base):
        root = Path(root)
        if len(root.relative_to(base).parts) > 4: continue
        for name in files:
            if name == "set_a.csv" and "set_a_csv" not in paths: paths["set_a_csv"] = root/name
            if name == "set_b.csv" and "set_b_csv" not in paths: paths["set_b_csv"] = root/name
        for d in dirs:
            if d == "set_a" and "set_a_dir" not in paths: paths["set_a_dir"] = root/d
            if d == "set_b" and "set_b_dir" not in paths: paths["set_b_dir"] = root/d
    return paths

PATHS = find_dataset(DATASET_DIR)
print("✅ Dataset found")

# ── STEP 5: Load metadata ────────────────────────────────────
VALID = {"normal","murmur","extrastole","artifact","extrahls"}

def load_metadata():
    df_a = pd.read_csv(PATHS["set_a_csv"])
    df_a["dataset"] = "A"
    df_a.columns    = [c.lower().strip() for c in df_a.columns]
    rows = []
    for f in PATHS["set_b_dir"].iterdir():
        if f.suffix != ".wav": continue
        prefix = f.name.split("_")[0].lower()
        if prefix in VALID:
            rows.append({"fname": f.name, "label": prefix, "dataset": "B"})
    df = pd.concat([df_a, pd.DataFrame(rows)], ignore_index=True)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[~df["label"].isin(["nan","unlabeled",""])]
    print("📊 Label distribution:")
    print(df["label"].value_counts().to_string())
    return df

df = load_metadata()

# ── STEP 6: Audio loader ─────────────────────────────────────
def load_audio(fname, dataset):
    folder = PATHS["set_a_dir"] if dataset=="A" else PATHS["set_b_dir"]
    path   = folder / Path(fname).name
    if not path.exists(): return None
    try:
        y, _ = librosa.load(path, sr=SR, duration=DURATION)
        target = SR * DURATION
        y = np.pad(y, (0, max(0, target-len(y))))[:target]
        # Normalise to [-1, 1] — Wav2Vec expects normalised audio
        y = y / (np.abs(y).max() + 1e-8)
        return y.astype(np.float32)
    except: return None

# ── STEP 7: Load Wav2Vec 2.0 backbone (FROZEN) ───────────────
# The backbone is pretrained on 960h LibriSpeech speech audio.
# We use it as a FROZEN feature extractor — no weights update.
# Only the tiny classification head trains.
#
# Why freeze?
#   - 90M params would overfit badly on 585 samples
#   - CPU training of 90M params = hours per epoch
#   - Frozen backbone = only 7,685 params train = minutes
#   - Pretrained representations already capture useful
#     acoustic structure (rhythm, spectral texture)
print("\n⏳ Loading Wav2Vec 2.0 backbone (frozen)...")
W2V_NAME      = "facebook/wav2vec2-base"
w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(W2V_NAME)
w2v_backbone  = Wav2Vec2Model.from_pretrained(W2V_NAME).to(DEVICE)

# Freeze ALL backbone parameters
for param in w2v_backbone.parameters():
    param.requires_grad = False

w2v_backbone.eval()
frozen_params  = sum(p.numel() for p in w2v_backbone.parameters())
print(f"✅ Wav2Vec loaded — {frozen_params:,} params FROZEN")

# ── STEP 8: Pre-extract Wav2Vec embeddings ───────────────────
# Since the backbone is frozen, we extract embeddings ONCE
# and cache them — no need to run the transformer every epoch.
# This is what makes Option 3 fast on CPU.
print("\n⏳ Pre-extracting Wav2Vec embeddings (one-time, ~20 min)...")
print("   (cached after this — training is then very fast)")

embeddings, labels_raw = [], []
skipped = 0

for i, (_, row) in enumerate(df.iterrows(), 1):
    if i % 20 == 0 or i == len(df):
        pct = i/len(df)*100
        print(f"  [{'█'*int(pct/5)+'░'*(20-int(pct/5))}] {pct:.0f}%  ({i}/{len(df)})", end="\r")

    audio = load_audio(row["fname"], row["dataset"])
    if audio is None: skipped += 1; continue

    try:
        inp = w2v_processor(audio, sampling_rate=SR,
                            return_tensors="pt", padding=True)
        with torch.no_grad():
            out = w2v_backbone(inp.input_values.to(DEVICE))

        hidden = out.last_hidden_state.squeeze(0)    # (T, 768)
        # Mean + Std pooling → richer than mean alone
        mean_vec = hidden.mean(dim=0)                # (768,)
        std_vec  = hidden.std(dim=0)                 # (768,)
        emb      = torch.cat([mean_vec, std_vec])    # (1536,)
        embeddings.append(emb.cpu().numpy().astype(np.float32))
        labels_raw.append(row["label"])
    except: skipped += 1

embeddings = np.array(embeddings, dtype=np.float32)
print(f"\n✅ Embeddings: {embeddings.shape}  |  skipped {skipped}")

# ── STEP 9: Encode labels ────────────────────────────────────
le        = LabelEncoder()
y_enc     = le.fit_transform(np.array(labels_raw))
N_CLASSES = len(le.classes_)
print(f"Classes ({N_CLASSES}): {le.classes_}")

# Train / Val / Test split
X_tv,  X_test,  y_tv,  y_test  = train_test_split(
    embeddings, y_enc, test_size=0.15,
    random_state=RANDOM_STATE, stratify=y_enc)

X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.15,
    random_state=RANDOM_STATE, stratify=y_tv)

print(f"Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

# ── STEP 10: PyTorch Dataset ─────────────────────────────────
class EmbeddingDataset(Dataset):
    """Simple dataset over pre-extracted embeddings."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loader(X, y, shuffle=True):
    return DataLoader(EmbeddingDataset(X, y),
                      batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   shuffle=False)
test_loader  = make_loader(X_test,  y_test,  shuffle=False)

# ── STEP 11: Classification head ─────────────────────────────
#
#  Architecture:
#  ┌─────────────────────────────────────────────────────┐
#  │  Wav2Vec 2.0 backbone (90M params, FROZEN)          │
#  │  facebook/wav2vec2-base                             │
#  │  Pretrained on LibriSpeech 960h                     │
#  └──────────────────────┬──────────────────────────────┘
#                         │  (pre-extracted, cached)
#                         ▼
#  ┌─────────────────────────────────────────────────────┐
#  │  Mean + Std pooling → 1536-dim embedding            │
#  └──────────────────────┬──────────────────────────────┘
#                         ▼
#  ┌─────────────────────────────────────────────────────┐
#  │  Classification Head (7,685 params, TRAINABLE)      │
#  │  LayerNorm(1536)                                    │
#  │  Dropout(0.3)                                       │
#  │  Linear(1536 → 5)                                  │
#  │  → Softmax                                          │
#  └─────────────────────────────────────────────────────┘

class Wav2VecHead(nn.Module):
    """
    Lightweight classification head on top of frozen Wav2Vec embeddings.

    LayerNorm stabilises the embedding distribution.
    Dropout prevents overfitting on the small dataset.
    Single linear layer maps 1536-dim → n_classes.

    Total trainable params: 1536*5 + 5 (bias) + 1536*2 (LN) = 7,685
    """
    def __init__(self, embed_dim: int = EMBED_DIM,
                 n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout(x)
        return self.linear(x)

head = Wav2VecHead(embed_dim=EMBED_DIM, n_classes=N_CLASSES).to(DEVICE)
trainable = sum(p.numel() for p in head.parameters())
print(f"\n🧠 Classification head: {trainable:,} trainable parameters")
print(f"   Backbone (frozen)  : {frozen_params:,} parameters")
print(f"   Training ratio     : {trainable/frozen_params*100:.4f}% of total model")

# ── STEP 12: Class-weighted loss ─────────────────────────────
class_counts  = np.bincount(y_train, minlength=N_CLASSES)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * N_CLASSES
weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── STEP 13: Training loop ───────────────────────────────────
# Much faster than Option 2 — no CNN forward pass,
# just matrix multiply on cached 1536-dim vectors
def run_epoch(loader, train=True):
    head.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            if train: optimizer.zero_grad()
            out  = head(X_batch)
            loss = criterion(out, y_batch)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct    += (out.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
    return total_loss / total, correct / total

print(f"\n🏋 Training classification head for {EPOCHS} epochs...")
print("   (Fast — only 7,685 params training on cached embeddings)\n")

hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
best_val_acc = 0.0
best_state   = None
patience     = 10
no_improve   = 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    scheduler.step()

    hist["train_loss"].append(tr_loss)
    hist["val_loss"].append(vl_loss)
    hist["train_acc"].append(tr_acc)
    hist["val_acc"].append(vl_acc)

    flag = ""
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        best_state   = {k: v.cpu().clone() for k,v in head.state_dict().items()}
        no_improve   = 0
        flag         = " ← best"
    else:
        no_improve  += 1

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:>3}/{EPOCHS}  "
              f"Train: {tr_acc:.1%} ({tr_loss:.4f})  "
              f"Val: {vl_acc:.1%} ({vl_loss:.4f}){flag}")

    if no_improve >= patience:
        print(f"\n  Early stopping at epoch {epoch}")
        break

head.load_state_dict(best_state)
print(f"\n✅ Best val accuracy: {best_val_acc:.1%}")

# ── STEP 14: Training curves ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, len(hist["train_loss"]) + 1)

axes[0].plot(ep, hist["train_loss"], label="Train", color="#4C72B0", lw=2)
axes[0].plot(ep, hist["val_loss"],   label="Val",   color="#DD8452", lw=2)
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(ep, [x*100 for x in hist["train_acc"]], label="Train", color="#4C72B0", lw=2)
axes[1].plot(ep, [x*100 for x in hist["val_acc"]],   label="Val",   color="#DD8452", lw=2)
axes[1].axhline(best_val_acc*100, color="green", linestyle="--", alpha=0.7,
                label=f"Best val {best_val_acc:.1%}")
axes[1].set_title("Accuracy (%)"); axes[1].set_xlabel("Epoch")
axes[1].legend(); axes[1].grid(alpha=0.3)

fig.suptitle("Wav2Vec 2.0 (frozen) + Classification Head — Training Curves",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(DATASET_DIR / "wav2vec_training_curves.png", dpi=150)
plt.show()
print("Saved → wav2vec_training_curves.png")

# ── STEP 15: Test evaluation ─────────────────────────────────
head.eval()
all_preds, all_true = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        out = head(X_batch.to(DEVICE))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_true.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_true  = np.array(all_true)
test_acc  = (all_preds == all_true).mean()

print(f"\n{'='*55}")
print(f"  TEST RESULTS — Wav2Vec 2.0 + Linear Head")
print(f"{'='*55}")
print(f"  Test Accuracy: {test_acc:.1%}")
print(f"\n{classification_report(all_true, all_preds, target_names=le.classes_)}")

# ── STEP 16: Confusion matrix ────────────────────────────────
cm = confusion_matrix(all_true, all_preds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title(f"Confusion Matrix (counts)\nAcc: {test_acc:.1%}", fontsize=11)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
sns.heatmap(cm_norm, annot=True, fmt=".0f", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title(f"Confusion Matrix (normalised %)\nAcc: {test_acc:.1%}", fontsize=11)
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(DATASET_DIR / "wav2vec_confusion_matrix.png", dpi=150)
plt.show()
print("Saved → wav2vec_confusion_matrix.png")

# ── STEP 17: Compare all 3 options ───────────────────────────
# Summary comparison of all three multimodal approaches
print("\n📊 All Options Comparison:")
option_results = {
    "Option 1\nFeature Fusion\n(Handcrafted+ComParE\n+Wav2Vec → LightGBM)": 78.6,
    "Option 2\nCNN + ComParE\nFusion\n(PyTorch)": 76.1,
    "Option 3\nWav2Vec Head\n(Transfer Learning)": test_acc * 100,
}

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4C72B0","#DD8452","#55A868"]
bars   = ax.bar(option_results.keys(),
                option_results.values(),
                color=colors, alpha=0.85, width=0.5)
for bar, v in zip(bars, option_results.values()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("Multimodal Approach Comparison\n(All trained on CPU, PASCAL Heartbeat Sounds)",
             fontsize=12, fontweight="bold")
ax.axhline(65.9, color="red", linestyle="--", alpha=0.5,
           label="ComParE alone baseline (65.9%)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(DATASET_DIR / "all_options_comparison.png", dpi=150)
plt.show()
print("Saved → all_options_comparison.png")

# ── STEP 18: Embedding visualisation (t-SNE) ─────────────────
print("\n⏳ Visualising Wav2Vec embeddings with t-SNE...")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# PCA first for speed, then t-SNE
pca   = PCA(n_components=50, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(embeddings)
tsne  = TSNE(n_components=2, perplexity=30, max_iter=1000,
             random_state=RANDOM_STATE, verbose=0)
X_2d  = tsne.fit_transform(X_pca)

COLORS = {
    "normal"    : "#4C72B0",
    "murmur"    : "#DD8452",
    "extrastole": "#55A868",
    "artifact"  : "#C44E52",
    "extrahls"  : "#8172B2",
}

fig, ax = plt.subplots(figsize=(9, 7))
for label in le.classes_:
    mask = np.array(labels_raw) == label
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=COLORS[label], label=label,
               s=35, alpha=0.75, edgecolors="none")
ax.set_title("Wav2Vec 2.0 Embeddings — t-SNE Projection\n"
             "(How the frozen backbone separates cardiac conditions)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10, markerscale=1.5)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(DATASET_DIR / "wav2vec_tsne_embeddings.png", dpi=150)
plt.show()
print("Saved → wav2vec_tsne_embeddings.png")

# ── STEP 19: Predict a single .wav file ──────────────────────
def predict(filepath: str):
    """Predict cardiac condition from any .wav file."""
    y, _ = librosa.load(filepath, sr=SR, duration=DURATION)
    target = SR * DURATION
    y = np.pad(y, (0, max(0, target-len(y))))[:target]
    y = (y / (np.abs(y).max() + 1e-8)).astype(np.float32)

    # Extract Wav2Vec embedding
    inp = w2v_processor(y, sampling_rate=SR,
                        return_tensors="pt", padding=True)
    with torch.no_grad():
        out    = w2v_backbone(inp.input_values.to(DEVICE))
        hidden = out.last_hidden_state.squeeze(0)
        emb    = torch.cat([hidden.mean(0), hidden.std(0)])   # (1536,)

    # Classify
    head.eval()
    with torch.no_grad():
        logits = head(emb.unsqueeze(0))
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred = le.classes_[np.argmax(probs)]
    print(f"\n🩺  File   : {Path(filepath).name}")
    print(f"    Result : {pred.upper()}")
    print()
    for cls, p in sorted(zip(le.classes_, probs), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"    {cls:<14} {bar} {p:.1%}")
    return pred, dict(zip(le.classes_, probs.tolist()))

# Demo predictions
print("\n── Demo predictions ────────────────────────────────")
for cls in ["normal","murmur","artifact"]:
    rows = df[df["label"] == cls]
    if len(rows) == 0: continue
    row  = rows.iloc[0]
    path = (PATHS["set_a_dir"] if row["dataset"]=="A"
            else PATHS["set_b_dir"]) / Path(row["fname"]).name
    predict(str(path))

# ── STEP 20: Save checkpoint ─────────────────────────────────
save_path = DATASET_DIR / "wav2vec_head.pt"
torch.save({
    "head_state"  : head.state_dict(),
    "le_classes"  : le.classes_,
    "config"      : {
        "embed_dim" : EMBED_DIM,
        "n_classes" : N_CLASSES,
        "w2v_model" : W2V_NAME,
        "sr"        : SR,
        "duration"  : DURATION,
    }
}, save_path)
print(f"\n✅ Head saved → {save_path}")

# ── STEP 21: Final summary ───────────────────────────────────
print(f"\n{'='*60}")
print(f"  WAV2VEC 2.0 + LINEAR HEAD — COMPLETE")
print(f"{'='*60}")
print(f"\n  Strategy      : Transfer learning (frozen backbone)")
print(f"  Backbone      : facebook/wav2vec2-base ({frozen_params:,} params, FROZEN)")
print(f"  Head          : LayerNorm + Dropout + Linear ({trainable:,} params)")
print(f"  Training ratio: {trainable/frozen_params*100:.4f}% of total model")
print(f"\n  Test accuracy : {test_acc:.1%}")
print(f"  Best val acc  : {best_val_acc:.1%}")
print(f"\n  Multimodal comparison:")
print(f"    Option 1 — Feature fusion (LightGBM)   : 78.6%")
print(f"    Option 2 — CNN + ComParE (PyTorch)      : 76.1%")
print(f"    Option 3 — Wav2Vec head (this)          : {test_acc:.1%}")
print(f"\n  Output files saved to: {DATASET_DIR}")
for f in ["wav2vec_training_curves.png",
          "wav2vec_confusion_matrix.png",
          "wav2vec_tsne_embeddings.png",
          "all_options_comparison.png",
          "wav2vec_head.pt"]:
    print(f"    → {f}")
print(f"{'='*60}")
