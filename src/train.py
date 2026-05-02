"""
train.py
────────
Training loop and evaluation for the Wav2Vec classification head.
Fast — operates on pre-cached embeddings, not raw audio.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def make_loss(y_train: np.ndarray, n_classes: int,
              device: str) -> nn.CrossEntropyLoss:
    """
    Class-weighted CrossEntropyLoss.
    Weights = 1 / class_frequency, normalised.
    Prevents the model favouring 'normal' (351 samples)
    over 'extrahls' (19 samples).
    """
    counts  = np.bincount(y_train, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    return nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device))


def run_epoch(model, loader: DataLoader,
              criterion, optimizer,
              device: str,
              train: bool = True) -> tuple[float, float]:
    """Run one training or evaluation epoch."""
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if train:
                optimizer.zero_grad()

            out  = model(X_batch)
            loss = criterion(out, y_batch)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (out.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

    return total_loss / total, correct / total


def train_head(head, train_loader, val_loader,
               criterion, optimizer, scheduler,
               epochs: int = 30,
               patience: int = 10,
               device: str = "cpu") -> dict:
    """
    Full training loop with early stopping.

    Because embeddings are pre-cached, each epoch is extremely fast —
    just a LayerNorm + Dropout + Linear forward/backward pass.

    Returns:
        history dict with per-epoch train/val loss and accuracy
    """
    hist = {"train_loss":[], "val_loss":[],
            "train_acc":[], "val_acc":[]}
    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(head, train_loader, criterion,
                                    optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(head, val_loader, criterion,
                                    optimizer, device, train=False)
        scheduler.step()

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone()
                            for k, v in head.state_dict().items()}
            no_improve   = 0
            flag         = " ← best"
        else:
            no_improve  += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"Train: {tr_acc:.1%} ({tr_loss:.4f})  "
                  f"Val: {vl_acc:.1%} ({vl_loss:.4f}){flag}")

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    head.load_state_dict(best_state)
    print(f"\n✅ Best val accuracy: {best_val_acc:.1%}")
    hist["best_val_acc"] = best_val_acc
    return hist


def evaluate(head, loader: DataLoader,
             device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over a DataLoader, return (preds, trues)."""
    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            out = head(X_batch.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y_batch.numpy())
    return np.array(preds), np.array(trues)
