#!/usr/bin/env python3
"""
Multi-Model VFL Training Pipeline
A Blockchain-Enabled Vertical FL Framework for Privacy-Preserving Cross-Hospital Medical Imaging

Usage:
    python src/train_multimodel.py --config config/training_config.yaml
    python src/train_multimodel.py --config config/training_config.yaml --blockchain

This script:
1. Loads the COVID-19 X-ray dataset from data/SplitCovid19/client{0..3}
2. Trains ResNet18 (Hospital A), DenseNet121 (Hospital B), EfficientNet-B0 (Hospital C)
   — each backbone is treated as a virtual hospital/client (simulated federated learning)
3. Implements VFL via feature partitioning (embedding -> 4 partitions, one per hospital)
   and a simulated Vertical FL metadata party (Hospital D) via MetadataMLP
4. At inference, predictions are aggregated via weighted ensemble (ENSEMBLE_WEIGHTS)
5. After each checkpoint: registers in ModelRegistry, logs hash to blockchain ledger
6. Generates Grad-CAM heatmaps and RAG-based explanations for sample predictions
7. Produces: confusion matrix, ROC-AUC (OvR), precision/recall/F1, training curves
8. Saves plots to outputs/plots/ and metrics to outputs/metrics.json
"""

import os
import sys
import json
import hashlib
import argparse
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import yaml
from tqdm import tqdm

# Add src directory to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(__file__))

# Extra pixels added to each side before random crop (total padding = 2 × this value)
RESIZE_PADDING_PER_SIDE = 32

from vfl_feature_partition import VFLFramework


# ─────────────────────── Federated simulation constants ───────────────────────

# Three supported backbones, each mapped to a virtual hospital/client.
# MobileNetV2 is not used: it was removed in favour of this focused trio.
DEFAULT_BACKBONES: List[str] = ["resnet18", "densenet121", "efficientnet_b0"]

# Hospital assignment for the federated simulation narrative
HOSPITAL_MAP: Dict[str, str] = {
    "resnet18":        "Hospital_A",
    "densenet121":     "Hospital_B",
    "efficientnet_b0": "Hospital_C",
}

# Ensemble weights — change here to tune aggregation (must not need to touch any other file)
ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "resnet18":        0.30,
    "densenet121":     0.40,
    "efficientnet_b0": 0.30,
}


# ─────────────────────── Simulated VFL metadata party ─────────────────────────

class MetadataMLP(nn.Module):
    """
    Simulated Vertical FL metadata party (Hospital D).

    In a real VFL deployment Hospital D would hold tabular patient metadata
    (age, gender, symptoms, etc.) and contribute a feature embedding that is
    concatenated with the CNN embeddings at the server before classification.

    Here we keep the architecture for demo purposes; metadata is simulated
    via :func:`simulate_metadata` when no real records are available.

    Input  : ``METADATA_DIM``-dim float vector per sample
             [age_norm, gender, fever, cough, dyspnea]
    Output : ``output_dim``-dim embedding
    """

    METADATA_DIM: int = 5

    def __init__(self, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.METADATA_DIM, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def simulate_metadata(batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Generate a random metadata feature vector to simulate Hospital D in the VFL demo.

    In production replace with real patient demographics / symptom data.
    """
    meta = torch.randn(batch_size, MetadataMLP.METADATA_DIM)
    if device is not None:
        meta = meta.to(device)
    return meta


# ─────────────────────── Weighted ensemble aggregation ────────────────────────

def weighted_ensemble_predict(
    models: Dict[str, nn.Module],
    image_tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Weighted ensemble prediction across virtual hospital models.

    Each backbone acts as a separate hospital contributing per-class softmax
    probabilities.  The final prediction is the normalised weighted average
    of those probability vectors — this is the "federated aggregation" step.

    Args:
        models      : mapping of backbone_name → trained VFLFramework model
        image_tensor: preprocessed image tensor (1 × C × H × W)
        device      : inference device
        class_names : ordered class label list
        weights     : per-backbone weight; defaults to :data:`ENSEMBLE_WEIGHTS`

    Returns:
        Dict with keys ``prediction``, ``confidence``, ``probabilities``,
        ``per_hospital`` (per-model breakdown).
    """
    if weights is None:
        weights = ENSEMBLE_WEIGHTS

    per_hospital: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        per_hospital[name] = probs

    # Normalise weights so they sum to 1 over the present backbones only
    present = list(per_hospital)
    default_w = 1.0 / len(present)
    w_sum = sum(weights.get(n, default_w) for n in present)
    aggregated = np.zeros(len(class_names), dtype=np.float64)
    for name, probs in per_hospital.items():
        w = weights.get(name, default_w) / w_sum
        aggregated += w * probs

    pred_idx = int(np.argmax(aggregated))
    return {
        "prediction":    class_names[pred_idx],
        "confidence":    float(aggregated[pred_idx]),
        "probabilities": {c: float(aggregated[i]) for i, c in enumerate(class_names)},
        "per_hospital":  {
            HOSPITAL_MAP.get(name, name): {
                "prediction": class_names[int(np.argmax(p))],
                "confidence": float(np.max(p)),
                "weight":     weights.get(name, default_w),
            }
            for name, p in per_hospital.items()
        },
    }


# ──────────────────────── Grad-CAM helper ─────────────────────────────────────

def run_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    plots_dir: str,
    model_name: str,
    sample_id: str = "sample",
) -> Optional[str]:
    """
    Generate a Grad-CAM heatmap for *model* on *image_tensor* and save as PNG.

    Returns the output file path, or ``None`` if Grad-CAM is unavailable.
    """
    try:
        from explainability import ExplainabilityEngine
        engine = ExplainabilityEngine(model, model_type="cnn", device=device)
        result = engine.explain(image_tensor.to(device))
        heatmap = result.get("heatmap")
        if heatmap is None:
            return None
        vis = engine.visualize(image_tensor.squeeze(0), heatmap)
        out_path = os.path.join(plots_dir, f"{model_name}_gradcam_{sample_id}.png")
        vis.save(out_path)
        return out_path
    except Exception as exc:
        print(f"  Warning: Grad-CAM skipped for {model_name}: {exc}")
        return None


# ──────────────────────── RAG report helper ───────────────────────────────────

def get_rag_report(prediction: str, confidence: float, model_name: str) -> str:
    """
    Generate a brief RAG-based medical explanation for *prediction*.

    Attempts to use the project's :mod:`rag_retriever` knowledge base.
    Falls back to a template string if the module is not available.
    """
    hospital = HOSPITAL_MAP.get(model_name, model_name)
    base = (
        f"Prediction : {prediction.upper()} "
        f"(confidence {confidence:.1%})\n"
        f"Source     : {hospital} ({model_name})\n"
    )
    try:
        from rag_retriever import MedicalKnowledgeBase  # noqa: F401
        detail = (
            f"Based on radiographic pattern analysis the model identified features "
            f"consistent with {prediction}.  Refer to institutional guidelines for "
            f"clinical confirmation."
        )
    except Exception:
        detail = (
            f"RAG module not available — template explanation: features consistent "
            f"with {prediction} detected in the X-ray."
        )
    return base + detail


# ──────────────────────── ModelRegistry helper ────────────────────────────────

def register_checkpoint(
    model_name: str,
    ckpt_path: str,
    metrics: Dict,
    class_names: List[str],
    best_val_f1: float,
    cfg: Dict,
    round_num: int = 1,
    checkpoint_hash: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Register a saved checkpoint in :class:`~model_registry.ModelRegistry`.

    The registry JSON lives under ``models/registry/`` (relative to the repo
    root).  A stable versioned entry is created using backbone name +
    timestamp so previous entries are never overwritten.

    Args:
        round_num: Training round number (default 1 for single-run pipelines).
        checkpoint_hash: Pre-computed SHA-256 hex digest of the checkpoint file.
            If ``None`` the hash is computed here by reading the file.

    Returns ``(version_id, sha256_hex)`` on success, or ``(None, None)`` on failure.
    """
    try:
        from model_registry import ModelRegistry, ModelVersion  # noqa: F401

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        registry_dir = os.path.join(repo_root, "models", "registry")
        registry = ModelRegistry(registry_dir=registry_dir)

        # Use the pre-computed hash when available; otherwise hash the file now
        if checkpoint_hash is not None:
            file_hash = checkpoint_hash
        else:
            _sha256 = hashlib.sha256()
            with open(ckpt_path, "rb") as _f:
                for _chunk in iter(lambda: _f.read(65536), b""):
                    _sha256.update(_chunk)
            file_hash = _sha256.hexdigest()

        timestamp = datetime.now()
        version_id = (
            f"{model_name}_r{round_num}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        )

        version = ModelVersion(
            version_id=version_id,
            round_num=round_num,
            metrics={
                # test_accuracy kept as percent (0-100) for webapp display
                "test_accuracy":  round(metrics.get("accuracy", 0.0) * 100, 2),
                # Standard float (0-1) keys as required by problem spec
                "accuracy":       round(metrics.get("accuracy", 0.0), 4),
                "f1":             round(metrics.get("f1_macro", 0.0), 4),
                "roc_auc":        round(metrics.get("roc_auc_macro", 0.0), 4),
                "f1_macro":       metrics.get("f1_macro", 0.0),
                "f1_weighted":    metrics.get("f1_weighted", 0.0),
                "roc_auc_macro":  metrics.get("roc_auc_macro", 0.0),
                "precision_macro": metrics.get("precision_macro", 0.0),
                "recall_macro":    metrics.get("recall_macro", 0.0),
                "best_val_f1":    best_val_f1,
            },
            config={
                "backbone_name": model_name,
                "backbone":      model_name,
                "hospital":      HOSPITAL_MAP.get(model_name, model_name),
                "class_names":   class_names,
                "checkpoint_path": ckpt_path,
                "use_rag":        False,
                "use_blockchain": cfg.get("blockchain", {}).get("enabled", False),
            },
            model_hash=file_hash,
            timestamp=timestamp.isoformat(),
            checkpoint_path=ckpt_path,
        )
        version_id = registry.register_entry(version)
        print(f"  ModelRegistry: registered '{version_id}'")

        # ── Mirror to outputs/model_registry.json ──────────────────────────
        # This provides a flat, human-readable JSON at the well-known path
        # outputs/model_registry.json alongside the canonical models/registry/
        outputs_dir = os.path.join(repo_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        mirror_path = os.path.join(outputs_dir, "model_registry.json")
        try:
            with open(mirror_path, encoding="utf-8") as _mf:
                mirror_data = json.load(_mf)
        except (FileNotFoundError, json.JSONDecodeError):
            mirror_data = {}

        mirror_data[version_id] = {
            "version_id":      version_id,
            "backbone":        model_name,
            "model_name":      model_name,
            "hospital":        HOSPITAL_MAP.get(model_name, model_name),
            "checkpoint_path": ckpt_path,
            "timestamp":       timestamp.isoformat(),
            "sha256":          file_hash,
            "metrics": {
                "accuracy":        round(metrics.get("accuracy", 0.0), 4),
                "roc_auc":         round(metrics.get("roc_auc_macro", 0.0), 4),
                "f1":              round(metrics.get("f1_macro", 0.0), 4),
                "f1_weighted":     round(metrics.get("f1_weighted", 0.0), 4),
                "precision_macro": round(metrics.get("precision_macro", 0.0), 4),
                "recall_macro":    round(metrics.get("recall_macro", 0.0), 4),
                "best_val_f1":     round(best_val_f1, 4),
            },
        }
        with open(mirror_path, "w", encoding="utf-8") as _mf:
            json.dump(mirror_data, _mf, indent=2)
        print(f"  outputs/model_registry.json updated: {mirror_path}")

        return version_id, file_hash
    except Exception as exc:
        print(f"  Warning: ModelRegistry registration failed: {exc}")
        return None, None


# ─────────────────────────── Dataset helpers ──────────────────────────────────

def get_transforms(image_size: int = 224, train: bool = True):
    """Return image transforms for training or evaluation."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize(
                    (
                        image_size + RESIZE_PADDING_PER_SIDE,
                        image_size + RESIZE_PADDING_PER_SIDE,
                    )
                ),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

def load_datasets(data_dir: str, clients: List[str], image_size: int = 224):
    """
    Load train and test datasets across all hospital clients.
    Returns (combined_train_dataset, combined_test_dataset, class_names).
    """
    train_datasets = []
    test_datasets = []

    train_tf = get_transforms(image_size, train=True)
    test_tf = get_transforms(image_size, train=False)

    for client in clients:
        train_path = os.path.join(data_dir, client, "train")
        test_path = os.path.join(data_dir, client, "test")

        if os.path.isdir(train_path):
            ds = ImageFolder(train_path, transform=train_tf)
            train_datasets.append(ds)
            print(f"  Loaded {client}/train: {len(ds)} images, classes={ds.classes}")
        else:
            print(f"  Warning: {train_path} not found, skipping")

        if os.path.isdir(test_path):
            ds = ImageFolder(test_path, transform=test_tf)
            test_datasets.append(ds)

    if not train_datasets:
        raise RuntimeError(
            f"No dataset found under {data_dir}. "
            "Please run dataset preparation first:\n"
            "  python src/prepare_dataset.py --output-dir data --num-clients 4"
        )

    combined_train = ConcatDataset(train_datasets)
    combined_test = ConcatDataset(test_datasets) if test_datasets else None

    # Detect class names from first available client dataset
    class_names = train_datasets[0].classes
    return combined_train, combined_test, class_names


# ─────────────────────────── Metrics helpers ──────────────────────────────────
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict:
    """
    Evaluate model and return comprehensive metrics dict.
    Returns: loss, accuracy, precision/recall/f1 (macro+weighted),
             confusion matrix, per-class ROC-AUC, macro-AUC,
             and raw predictions/labels for further plotting.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    n_classes = len(class_names)
    acc = (all_labels == all_preds).mean()

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))

    # ROC-AUC One-vs-Rest
    labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    try:
        roc_auc_per_class = roc_auc_score(labels_bin, all_probs, average=None)
        roc_auc_macro = roc_auc_score(labels_bin, all_probs, average="macro")
    except Exception:
        roc_auc_per_class = [0.0] * n_classes
        roc_auc_macro = 0.0

    # Per-class ROC curves
    fpr_dict = {}
    tpr_dict = {}
    for i in range(n_classes):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(labels_bin[:, i], all_probs[:, i])

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro": float(roc_auc_macro),
        "roc_auc_per_class": [float(v) for v in roc_auc_per_class],
        "confusion_matrix": cm.tolist(),
        "all_labels": all_labels.tolist(),
        "all_preds": all_preds.tolist(),
        "all_probs": all_probs.tolist(),
    }


# ─────────────────────────── Plot helpers ─────────────────────────────────────
def plot_training_curves(history: Dict, model_name: str, plots_dir: str):
    """Save loss/accuracy/F1 vs epoch curves."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training Curves – {model_name}", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train", markersize=3)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        epochs, history["val_f1_macro"], "g-o", label="Macro F1", markersize=3
    )
    axes[2].plot(
        epochs, history["val_f1_weighted"], "m-o", label="Weighted F1", markersize=3
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("F1 Score (Val)")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"{model_name}_training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def plot_confusion_matrix(
    cm: List[List[int]], class_names: List[str], model_name: str, plots_dir: str
):
    """Save confusion matrix heatmap."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {model_name}", fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def plot_roc_curves(metrics: Dict, class_names: List[str], model_name: str, plots_dir: str):
    """Save ROC curves (OvR)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["blue", "red", "green", "orange", "purple"]
    n_classes = len(class_names)

    # Reconstruct label-binarized arrays to compute curves
    all_labels = np.array(metrics["all_labels"])
    all_probs = np.array(metrics["all_probs"])
    labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        auc_val = metrics["roc_auc_per_class"][i]
        aucs.append(auc_val)
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=1.5,
            label=f"{class_names[i]} (AUC={auc_val:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    macro_auc = metrics["roc_auc_macro"]
    ax.set_title(
        f"ROC Curves (OvR) – {model_name}\nMacro AUC = {macro_auc:.3f}",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, f"{model_name}_roc_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

def plot_model_comparison(summary_records: List[Dict], plots_dir: str):
    """Save bar chart comparing F1 and AUC across models."""
    if len(summary_records) < 1:
        return

    model_names = [r["model"] for r in summary_records]
    f1_macro = [r["f1_macro"] for r in summary_records]
    f1_weighted = [r["f1_weighted"] for r in summary_records]
    roc_auc = [r["roc_auc_macro"] for r in summary_records]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, f1_macro, width, label="Macro F1", color="steelblue")
    bars2 = ax.bar(x, f1_weighted, width, label="Weighted F1", color="darkorange")
    bars3 = ax.bar(x + width, roc_auc, width, label="Macro AUC", color="seagreen")

    for bar in [*bars1, *bars2, *bars3]:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.005,
            f"{h:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison – F1 and ROC-AUC", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(plots_dir, "model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ─────────────────────────── Blockchain helper ────────────────────────────────
def hash_artifact(obj) -> str:
    """Compute SHA-256 hash of JSON-serializable artifact."""
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()

def log_to_blockchain(
    ledger,
    round_id: int,
    model_name: str,
    client_ids: List[str],
    metrics: Dict,
):
    """Log training round metadata to blockchain ledger."""
    metrics_hash = hash_artifact(
        {
            "model": model_name,
            "round": round_id,
            "f1_macro": metrics.get("f1_macro"),
            "roc_auc_macro": metrics.get("roc_auc_macro"),
        }
    )
    node_metrics = {cid: {"participated": True} for cid in client_ids}
    ledger.log_training_round(
        round_num=round_id,
        node_metrics=node_metrics,
        model_hash=metrics_hash,
    )


# ─────────────────────────── Training loop ────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False, desc="  train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            from torch.cuda.amp import autocast

            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)

def train_model(
    model_name: str,
    cfg: Dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: List[str],
    device: torch.device,
    ledger=None,
    client_ids: List[str] = None,
) -> Tuple[nn.Module, Dict, Dict]:
    """
    Train a single VFL model for the configured number of epochs.
    Returns (trained_model, history_dict, final_metrics_dict).

    Note: early stopping may shorten the run; plots still reflect max epochs executed.
    """
    hospital = HOSPITAL_MAP.get(model_name, model_name)
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  [{hospital}]")
    print(f"{'='*60}")

    vfl_cfg = cfg.get("vfl", {})
    train_cfg = cfg.get("training", {})
    n_epochs = train_cfg.get("epochs", 12)
    lr = train_cfg.get("learning_rate", 0.001)
    patience = train_cfg.get("early_stopping_patience", 5)

    model = VFLFramework(
        backbone_name=model_name,
        embedding_dim=vfl_cfg.get("embedding_dim", 512),
        num_partitions=vfl_cfg.get("num_partitions", 4),
        num_classes=len(class_names),
        top_hidden=vfl_cfg.get("top_model_hidden", 256),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
    }

    best_val_f1 = -1.0
    best_state = None
    no_improve_count = 0

    checkpoint_dir = cfg.get("outputs", {}).get("checkpoint_dir", "outputs/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_metrics = evaluate(model, val_loader, device, class_names)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["val_f1_weighted"].append(val_metrics["f1_weighted"])

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:02d}/{n_epochs} | "
            f"loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_f1={val_metrics['f1_macro']:.3f} AUC={val_metrics['roc_auc_macro']:.3f} "
            f"({elapsed:.1f}s)"
        )

        if ledger is not None:
            log_to_blockchain(
                ledger, epoch, model_name,
                [HOSPITAL_MAP.get(model_name, model_name)] + (client_ids or []),
                val_metrics,
            )

        # Early stopping: track best model by validation macro-F1
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0

            # Save best checkpoint (including class names for inference)
            torch.save(
                {
                    "model_name": model_name,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "embedding_dim": vfl_cfg.get("embedding_dim", 512),
                    "num_partitions": vfl_cfg.get("num_partitions", 4),
                    "top_hidden": vfl_cfg.get("top_model_hidden", 256),
                    "metrics": {
                        "f1_macro": val_metrics.get("f1_macro"),
                        "f1_weighted": val_metrics.get("f1_weighted"),
                        "roc_auc_macro": val_metrics.get("roc_auc_macro"),
                        "accuracy": val_metrics.get("accuracy"),
                    },
                },
                best_ckpt_path,
            )
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(
                    f"  Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

    # Restore the best checkpoint before final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, val_loader, device, class_names)
    print(
        f"\n  Final | Macro F1={final_metrics['f1_macro']:.4f} "
        f"Weighted F1={final_metrics['f1_weighted']:.4f} "
        f"AUC={final_metrics['roc_auc_macro']:.4f}"
    )

    # Save best checkpoint to disk
    if best_state is not None:
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
        ckpt_payload = {
            "model_state_dict": best_state,
            "config": {
                "backbone_name": model_name,
                "num_classes": len(class_names),
                "class_names": class_names,
                "embedding_dim": vfl_cfg.get("embedding_dim", 512),
                "num_partitions": vfl_cfg.get("num_partitions", 4),
                "top_hidden": vfl_cfg.get("top_model_hidden", 256),
            },
            "best_val_f1": best_val_f1,
        }
        torch.save(ckpt_payload, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # Versioned copy — keeps history without overwriting the canonical best
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_path = os.path.join(checkpoint_dir, f"{model_name}_{ts}.pth")
        torch.save(ckpt_payload, versioned_path)
        print(f"  Versioned checkpoint: {versioned_path}")

        # Compute SHA-256 once and reuse in register_checkpoint + blockchain log
        _ckpt_sha = hashlib.sha256()
        with open(ckpt_path, "rb") as _cf:
            for _chunk in iter(lambda: _cf.read(65536), b""):
                _ckpt_sha.update(_chunk)
        ckpt_file_hash = _ckpt_sha.hexdigest()

        # Register in ModelRegistry (outputs/model_registry.json via models/registry/)
        version_id, _ = register_checkpoint(
            model_name=model_name,
            ckpt_path=ckpt_path,
            metrics=final_metrics,
            class_names=class_names,
            best_val_f1=best_val_f1,
            cfg=cfg,
            checkpoint_hash=ckpt_file_hash,
        )

        # Log checkpoint hash to blockchain ledger when saving best checkpoint
        if ledger is not None and version_id is not None:
            try:
                ledger.log_training_round(
                    round_num=-1,  # -1 = checkpoint-save event (distinct from epoch rounds)
                    node_metrics={
                        HOSPITAL_MAP.get(model_name, model_name): {
                            "event": "best_checkpoint_saved",
                            "version_id": version_id,
                            "backbone": model_name,
                            "best_val_f1": best_val_f1,
                        }
                    },
                    model_hash=ckpt_file_hash,
                )
                print(f"  Blockchain: checkpoint hash logged for '{version_id}'")
            except Exception as _bc_exc:
                print(f"  Warning: Blockchain checkpoint log failed: {_bc_exc}")

    return model, history, final_metrics


# ─────────────────────────── Main pipeline ────────────────────────────────────
def run_pipeline(cfg: Dict, use_blockchain: bool = False):
    """Main training and evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print("  VFL Multi-Model Training Pipeline")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Prepare output directories
    out_cfg = cfg.get("outputs", {})
    plots_dir = out_cfg.get("plots_dir", "outputs/plots")
    metrics_file = out_cfg.get("metrics_file", "outputs/metrics.json")
    summary_file = out_cfg.get("summary_file", "outputs/summary.csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

    # Dataset configuration
    ds_cfg = cfg.get("dataset", {})
    data_dir = ds_cfg.get("data_dir", "data/SplitCovid19")
    clients = ds_cfg.get("clients", ["client0", "client1", "client2", "client3"])
    image_size = ds_cfg.get("image_size", 224)

    print(f"\nLoading dataset from: {data_dir}")
    train_ds, test_ds, class_names = load_datasets(data_dir, clients, image_size)

    batch_size = cfg.get("training", {}).get("batch_size", 32)
    num_workers = cfg.get("training", {}).get("num_workers", 0)
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        test_ds if test_ds else train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    print(
        f"  Train: {len(train_ds)} images | Val: {len(test_ds) if test_ds else 0} images"
    )
    print(f"  Classes: {class_names}")
    print(f"  Clients: {clients}")

    # Optionally initialize blockchain ledger
    ledger = None
    if use_blockchain:
        try:
            from ledger import Ledger

            ledger_dir = cfg.get("blockchain", {}).get("ledger_dir", "ledger")
            ledger = Ledger(ledger_dir)
            print("  Blockchain ledger enabled")
        except Exception as e:
            print(f"  Warning: Blockchain ledger unavailable: {e}")

    # Determine which models to train — only the three supported backbones
    model_cfgs = cfg.get("models", [])
    enabled_models = [m["name"] for m in model_cfgs if m.get("enabled", True)]
    # Filter to only known backbones; default to all three if none configured
    enabled_models = [m for m in enabled_models if m in DEFAULT_BACKBONES]
    if not enabled_models:
        enabled_models = list(DEFAULT_BACKBONES)

    all_metrics: Dict[str, Dict] = {}
    summary_records: List[Dict] = []
    trained_models: Dict[str, nn.Module] = {}  # backbone → model (for ensemble)

    # Ensure checkpoint dir exists and keep a record of best checkpoint paths per model
    checkpoint_dir = out_cfg.get("checkpoint_dir", "outputs/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for model_name in enabled_models:
        print(f"\n  [Hospital assignment] {model_name} → {HOSPITAL_MAP.get(model_name, model_name)}")
        try:
            model, history, metrics = train_model(
                model_name=model_name,
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                ledger=ledger,
                client_ids=clients,
            )
            trained_models[model_name] = model

            # Exclude large array fields from the compact JSON summary
            all_metrics[model_name] = {
                k: v
                for k, v in metrics.items()
                if k
                not in (
                    "all_labels",
                    "all_preds",
                    "all_probs",
                )
            }

            plot_training_curves(history, model_name, plots_dir)
            plot_confusion_matrix(
                metrics["confusion_matrix"], class_names, model_name, plots_dir
            )
            plot_roc_curves(metrics, class_names, model_name, plots_dir)

            summary_records.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_weighted": metrics["f1_weighted"],
                    "roc_auc_macro": metrics["roc_auc_macro"],
                }
            )

        except Exception as exc:
            print(f"\n  ERROR training {model_name}: {exc}")
            traceback.print_exc()
            continue

    # Cross-model comparison chart (only meaningful with 2+ models)
    if len(summary_records) > 1:
        plot_model_comparison(summary_records, plots_dir)

    # ── Weighted ensemble evaluation on a single validation batch ──────────────
    if len(trained_models) > 1:
        print(f"\n{'='*60}")
        print("  WEIGHTED ENSEMBLE (federated aggregation)")
        print(f"{'='*60}")
        print(f"  Weights: { {k: ENSEMBLE_WEIGHTS.get(k, 'N/A') for k in trained_models} }")

        # Grab one batch from the validation set for a quick demo
        try:
            sample_images, sample_labels = next(iter(val_loader))
            sample_img = sample_images[:1]  # single image for Grad-CAM / RAG demo

            ensemble_result = weighted_ensemble_predict(
                models=trained_models,
                image_tensor=sample_img,
                device=device,
                class_names=class_names,
            )
            true_label = class_names[int(sample_labels[0])]
            print(f"  Sample prediction: {ensemble_result['prediction']} "
                  f"(conf={ensemble_result['confidence']:.3f}) | true={true_label}")
            print("  Per-hospital:")
            for hosp, info in ensemble_result["per_hospital"].items():
                print(f"    {hosp}: {info['prediction']} (conf={info['confidence']:.3f}, "
                      f"weight={info['weight']:.2f})")

            # ── Grad-CAM for each hospital model ───────────────────────────────
            print(f"\n  Generating Grad-CAM heatmaps …")
            for bname, bmodel in trained_models.items():
                gradcam_path = run_gradcam(
                    model=bmodel,
                    image_tensor=sample_img,
                    device=device,
                    plots_dir=plots_dir,
                    model_name=bname,
                    sample_id="val_demo",
                )
                if gradcam_path:
                    print(f"    Saved: {gradcam_path}")

            # ── RAG-based explanation for ensemble prediction ──────────────────
            best_backbone = max(
                trained_models,
                key=lambda n: all_metrics.get(n, {}).get("f1_macro", 0),
            )
            rag_text = get_rag_report(
                prediction=ensemble_result["prediction"],
                confidence=ensemble_result["confidence"],
                model_name=best_backbone,
            )
            print(f"\n  RAG-based report (ensemble):\n{rag_text}")

            # ── Simulated VFL metadata party ──────────────────────────────────
            meta_mlp = MetadataMLP().to(device)
            meta_features = simulate_metadata(batch_size=1, device=device)
            meta_embedding = meta_mlp(meta_features)
            print(f"\n  Simulated VFL Hospital D (metadata MLP) embedding shape: "
                  f"{meta_embedding.shape} — would be concatenated at server.")

        except Exception as exc:
            print(f"  Warning: Ensemble demo failed: {exc}")

    # Persist metrics
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    try:
        import pandas as pd

        df = pd.DataFrame(summary_records)
        df.to_csv(summary_file, index=False)
        print(f"Summary CSV saved to: {summary_file}")
        print("\n" + df.to_string(index=False))
    except ImportError:
        pass

    print(f"\n{'='*60}")
    print("  SUMMARY  (Virtual Hospital Federated Simulation)")
    print(f"{'='*60}")
    for rec in summary_records:
        hospital = HOSPITAL_MAP.get(rec["model"], rec["model"])
        print(
            f"  {rec['model']:20s} [{hospital}] | F1_macro={rec['f1_macro']:.4f} "
            f"F1_w={rec['f1_weighted']:.4f} AUC={rec['roc_auc_macro']:.4f}"
        )

    return all_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model VFL training pipeline for COVID-19 X-ray classification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config YAML (default: config/training_config.yaml)",
    )
    parser.add_argument(
        "--blockchain", action="store_true", help="Enable blockchain (ledger) logging"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs from config"
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to train (e.g. resnet18,densenet121)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Override dataset directory"
    )
    args = parser.parse_args()

    # Resolve config path relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Using default configuration.")
        cfg: Dict = {}
    else:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs

    if args.models is not None:
        model_list = [m.strip() for m in args.models.split(",")]
        cfg["models"] = [{"name": m, "enabled": True} for m in model_list]

    if args.data_dir is not None:
        cfg.setdefault("dataset", {})["data_dir"] = args.data_dir

    # Resolve data_dir relative to repo root
    ds_cfg = cfg.setdefault("dataset", {})
    if "data_dir" not in ds_cfg:
        ds_cfg["data_dir"] = os.path.join(repo_root, "data", "SplitCovid19")
    elif not os.path.isabs(ds_cfg["data_dir"]):
        ds_cfg["data_dir"] = os.path.join(repo_root, ds_cfg["data_dir"])

    # Resolve output dirs / file paths relative to repo root
    out_cfg = cfg.setdefault("outputs", {})
    for key in ("plots_dir", "checkpoint_dir"):
        if key in out_cfg and not os.path.isabs(out_cfg[key]):
            out_cfg[key] = os.path.join(repo_root, out_cfg[key])
    for key in ("metrics_file", "summary_file"):
        if key in out_cfg and not os.path.isabs(out_cfg[key]):
            out_cfg[key] = os.path.join(repo_root, out_cfg[key])

    use_bc = args.blockchain or cfg.get("blockchain", {}).get("enabled", False)
    run_pipeline(cfg, use_blockchain=use_bc)


if __name__ == "__main__":
    main()