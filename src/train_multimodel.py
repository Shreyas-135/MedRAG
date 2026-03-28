#!/usr/bin/env python3
"""
Multi-Model VFL Training Pipeline
A Blockchain-Enabled Vertical FL Framework for Privacy-Preserving Cross-Hospital Medical Imaging

Usage:
    python src/train_multimodel.py --config config/training_config.yaml
    python src/train_multimodel.py --config config/training_config.yaml --blockchain

This script:
1. Loads the COVID-19 X-ray dataset from data/SplitCovid19/client{0..3}
2. Trains ResNet18, DenseNet121, EfficientNet-B0, MobileNetV2
3. Implements VFL via feature partitioning (embedding -> 4 partitions, one per hospital)
4. Produces: confusion matrix, ROC-AUC (OvR), precision/recall/F1, training curves
5. Saves plots to outputs/plots/ and metrics to outputs/metrics.json
6. Optionally logs to blockchain (ledger.py hash-chaining)
"""

import os
import sys
import json
import hashlib
import argparse
import time
import traceback
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
    auc,
)
from sklearn.preprocessing import label_binarize
import yaml
from tqdm import tqdm

# Add src directory to path so sibling modules are importable
sys.path.insert(0, os.path.dirname(__file__))

# Pixels added on each side before random crop during training augmentation
_RESIZE_PADDING = 32

from vfl_feature_partition import VFLFramework


# ─────────────────────────── Dataset helpers ──────────────────────────────────

def get_transforms(image_size: int = 224, train: bool = True):
    """Return image transforms for training or evaluation."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + _RESIZE_PADDING, image_size + _RESIZE_PADDING)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


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


def load_per_client_datasets(data_dir: str, clients: List[str], image_size: int = 224):
    """
    Load per-client datasets separately (for FL simulation).
    Returns dict: {client_id: (train_dataset, test_dataset)}
    """
    result = {}
    train_tf = get_transforms(image_size, train=True)
    test_tf = get_transforms(image_size, train=False)

    for client in clients:
        train_path = os.path.join(data_dir, client, "train")
        test_path = os.path.join(data_dir, client, "test")
        train_ds = None
        test_ds = None
        if os.path.isdir(train_path):
            train_ds = ImageFolder(train_path, transform=train_tf)
        if os.path.isdir(test_path):
            test_ds = ImageFolder(test_path, transform=test_tf)
        if train_ds is not None:
            result[client] = (train_ds, test_ds)

    return result


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
        "fpr": {str(k): v.tolist() for k, v in fpr_dict.items()},
        "tpr": {str(k): v.tolist() for k, v in tpr_dict.items()},
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

    axes[2].plot(epochs, history["val_f1_macro"], "g-o", label="Macro F1", markersize=3)
    axes[2].plot(epochs, history["val_f1_weighted"], "m-o", label="Weighted F1", markersize=3)
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
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
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
    """Save ROC curves (OvR) with macro average."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["blue", "red", "green", "orange", "purple"]
    n_classes = len(class_names)

    for i in range(n_classes):
        fpr = metrics["fpr"][str(i)]
        tpr = metrics["tpr"][str(i)]
        auc_val = metrics["roc_auc_per_class"][i]
        ax.plot(
            fpr, tpr,
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
            bar.get_x() + bar.get_width() / 2, h + 0.005,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7,
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
    metrics_hash = hash_artifact({
        "model": model_name,
        "round": round_id,
        "f1_macro": metrics.get("f1_macro"),
        "roc_auc_macro": metrics.get("roc_auc_macro"),
    })
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
    """
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
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
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1_macro": [], "val_f1_weighted": [],
    }

    best_val_f1 = -1.0
    best_state = None
    no_improve_count = 0
    checkpoint_dir = cfg.get("outputs", {}).get("checkpoint_dir", "outputs/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

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
            log_to_blockchain(ledger, epoch, model_name, client_ids or [], val_metrics)

        # Early stopping: track best model by validation macro-F1
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_count = 0
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
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        test_ds if test_ds else train_ds,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    print(f"  Train: {len(train_ds)} images | Val: {len(test_ds) if test_ds else 0} images")
    print(f"  Classes: {class_names}")
    print(f"  Clients: {clients}")

    # Optionally initialise blockchain ledger
    ledger = None
    if use_blockchain:
        try:
            from ledger import Ledger
            ledger_dir = cfg.get("blockchain", {}).get("ledger_dir", "ledger")
            ledger = Ledger(ledger_dir)
            print("  Blockchain ledger enabled")
        except Exception as e:
            print(f"  Warning: Blockchain ledger unavailable: {e}")

    # Determine which models to train
    model_cfgs = cfg.get("models", [])
    enabled_models = [m["name"] for m in model_cfgs if m.get("enabled", True)]
    if not enabled_models:
        enabled_models = ["resnet18", "densenet121", "efficientnet_b0", "mobilenet_v2"]

    all_metrics: Dict[str, Dict] = {}
    summary_records: List[Dict] = []

    for model_name in enabled_models:
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

            # Exclude large array fields from the compact JSON summary
            all_metrics[model_name] = {
                k: v for k, v in metrics.items()
                if k not in ("fpr", "tpr", "all_labels", "all_preds", "all_probs")
            }

            plot_training_curves(history, model_name, plots_dir)
            plot_confusion_matrix(metrics["confusion_matrix"], class_names, model_name, plots_dir)
            plot_roc_curves(metrics, class_names, model_name, plots_dir)

            summary_records.append({
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "roc_auc_macro": metrics["roc_auc_macro"],
            })

        except Exception as exc:
            print(f"\n  ERROR training {model_name}: {exc}")
            traceback.print_exc()
            continue

    # Cross-model comparison chart (only meaningful with 2+ models)
    if len(summary_records) > 1:
        plot_model_comparison(summary_records, plots_dir)

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
    print("  SUMMARY")
    print(f"{'='*60}")
    for rec in summary_records:
        print(
            f"  {rec['model']:20s} | F1_macro={rec['f1_macro']:.4f} "
            f"F1_w={rec['f1_weighted']:.4f} AUC={rec['roc_auc_macro']:.4f}"
        )

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Multi-model VFL training pipeline for COVID-19 X-ray classification"
    )
    parser.add_argument(
        "--config", type=str, default="config/training_config.yaml",
        help="Path to training config YAML (default: config/training_config.yaml)",
    )
    parser.add_argument(
        "--blockchain", action="store_true",
        help="Enable blockchain (ledger) logging",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models to train (e.g. resnet18,densenet121)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override dataset directory",
    )
    args = parser.parse_args()

    # Resolve config path relative to repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_path = (
        args.config if os.path.isabs(args.config)
        else os.path.join(repo_root, args.config)
    )

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
