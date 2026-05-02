# 🫀 Cardiac Sound — Wav2Vec Transfer Learning

Transfer learning for cardiac condition detection — a **94M-parameter Wav2Vec 2.0 backbone** (completely frozen) with a **tiny 10K-parameter linear head** trained on heartbeat audio. Achieves 67% accuracy by updating only **0.011% of the model** on CPU.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Wav2Vec2-yellow.svg)](https://huggingface.co/facebook/wav2vec2-base)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg)](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds)

---

## 🧠 Core Idea

Most fine-tuning approaches update all model weights. This project takes the extreme opposite: **freeze everything except the last layer**.

```
facebook/wav2vec2-base (94,371,712 params)
         FROZEN — zero gradient updates
              ↓
    Mean + Std pooling → 1,536-dim
              ↓
  ┌─────────────────────────────┐
  │  Classification Head        │  ← only this trains
  │  LayerNorm(1536)            │
  │  Dropout(0.3)               │  10,757 parameters
  │  Linear(1536 → 5 classes)  │  0.011% of total model
  └─────────────────────────────┘
              ↓
    Normal / Murmur / Extrastole
    Extrahls / Artifact
```

**Why freeze?**
- 94M params would massively overfit on 585 audio samples
- Full fine-tuning on CPU would take hours per epoch
- Frozen backbone means only 10,757 numbers update — **training takes minutes**
- Wav2Vec pretrained representations already capture useful acoustic structure (rhythm, spectral texture, periodicity)

---

## 📊 Results

### Test Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | **67.0%** |
| Best Val Accuracy | 58.7% |
| Macro F1 | 63% |
| Weighted F1 | 70% |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| artifact | 1.00 | 0.83 | **0.91** | 6 |
| extrahls | 0.60 | 1.00 | 0.75 | 3 |
| extrastole | 0.00 | 0.00 | 0.00 | 7 |
| murmur | 0.81 | 0.68 | **0.74** | 19 |
| normal | 0.78 | 0.72 | 0.75 | 53 |

### Multimodal Comparison (same dataset, all CPU)

| Approach | Accuracy | Trainable Params | Runtime |
|---|---|---|---|
| Option 1 — Feature fusion (LightGBM) | **78.6%** | N/A | ~5 min |
| Option 2 — CNN + ComParE fusion | 76.1% | 3.49M | ~45 min |
| **Option 3 — Wav2Vec head (this)** | **67.0%** | **10,757** | **~30 min** |

> Option 3 uses **324× fewer trainable parameters** than Option 2, with only a 9% accuracy trade-off. For production deployment, a 10K-param head is far easier to version, update, and serve.

---

## 🗂️ Repository Structure

```
cardiac-wavvec-transfer/
│
├── src/
│   ├── dataset.py      # Audio loading, Wav2Vec embedding extraction
│   ├── model.py        # Wav2VecHead classification head
│   ├── train.py        # Training loop, early stopping
│   └── predict.py      # Single-file inference
│
├── notebooks/
│   └── wav2vec_transfer.py    # Full pipeline — paste into Jupyter
│
├── results/            # Training curves, confusion matrix, t-SNE
├── data/samples/       # Put sample .wav files here
├── tests/
│   └── test_model.py   # Unit tests for head architecture
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Arun-Seb/cardiac-wavvec-transfer.git
cd cardiac-wavvec-transfer
pip install -r requirements.txt
```

Open `notebooks/wav2vec_transfer.py` in Jupyter and run all cells.

### Predict a single file

```python
from src.predict import predict, load_checkpoint

head, processor, backbone, le_classes = load_checkpoint("wav2vec_head.pt")
pred, probs = predict("heartbeat.wav", head, processor, backbone, le_classes)
# 🩺  Result: NORMAL
# normal    ████████████████████████████ 72.4%
# murmur    ████████ 20.1%
# ...
```

---

## 🔬 How It Works

### Step 1 — Pre-extract embeddings (one-time, ~20 min)

```python
# Wav2Vec processes raw audio directly — no mel spectrogram needed
inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

with torch.no_grad():          # backbone frozen — no grad tracking
    output = backbone(inputs.input_values)

hidden = output.last_hidden_state   # (T, 768) — T varies by clip length
mean   = hidden.mean(dim=0)         # (768,)   — temporal mean
std    = hidden.std(dim=0)          # (768,)   — temporal std
embed  = torch.cat([mean, std])     # (1536,)  — richer than mean alone
```

**Why Mean + Std pooling?** Mean captures the average acoustic content; Std captures how much it varies over time — important for detecting irregular patterns like murmurs.

### Step 2 — Train the head (fast, ~5-10 min)

```python
# Only the head has requires_grad=True
head = nn.Sequential(
    nn.LayerNorm(1536),   # stabilise embedding distribution
    nn.Dropout(0.3),      # prevent overfitting on 585 samples
    nn.Linear(1536, 5),   # map to 5 cardiac conditions
)
```

Since embeddings are pre-cached, each training epoch is just a matrix multiply — no audio loading, no transformer forward pass.

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Backbone | facebook/wav2vec2-base (frozen) |
| Embedding | Mean + Std pooling → 1,536-dim |
| Head | LayerNorm + Dropout(0.3) + Linear |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss (class-weighted) |
| Early stopping | patience=10, triggered at epoch 14 |
| Train/Val/Test | 72% / 13% / 15% |

---

## 💡 Key Findings

1. **0.011% of model parameters is enough** — a single linear layer on top of frozen Wav2Vec features achieves 67% accuracy, proving the pretrained representations are genuinely useful for cardiac audio.

2. **Artifact detection is near-perfect (F1=0.91)** — recording noise has a very distinctive acoustic signature that Wav2Vec captures well even without fine-tuning.

3. **Extrastole is the hardest class (F1=0.00)** — only 7 test samples, acoustically similar to normal. More data or data augmentation needed.

4. **Early stopping at epoch 14** — the head converges quickly. The backbone does all the heavy lifting; the head just learns to read the embedding.

5. **Accuracy ceiling without fine-tuning** — 67% suggests the Wav2Vec speech representations are useful but not perfectly aligned with cardiac audio. Partially unfreezing the last transformer layer would likely push this above 75%.

---

## 🔄 Related Repositories

- 👉 [cardiac-sound-classifier](https://github.com/Arun-Seb/cardiac-sound-classifier) — full ML benchmark (RF/XGBoost/LightGBM/SVM)
- 👉 [cardiac-clustering-analysis](https://github.com/Arun-Seb/cardiac-clustering-analysis) — unsupervised clustering
- 👉 [cardiac-multimodal](https://github.com/Arun-Seb/cardiac-multimodal) — CNN + ComParE fusion

---

## 📦 Dependencies

```
torch, torchaudio        — PyTorch backend
transformers==4.35.2     — Wav2Vec 2.0 model
librosa                  — audio loading and resampling
scikit-learn             — metrics, label encoding
pandas, numpy            — data handling
matplotlib, seaborn      — visualisation
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Arun** — [github.com/Arun-Seb](https://github.com/Arun-Seb)
