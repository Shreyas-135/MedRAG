"""
Prediction Calibration for Federated Medical Imaging

Provides per-hospital temperature scaling and calibration metrics
(Expected Calibration Error, Maximum Calibration Error, reliability
diagrams) to make model confidence scores clinically trustworthy.

Calibration is particularly important in this VFL setting because:
- Each hospital client trains on a different local data distribution,
  leading to systematic over- or under-confidence that varies by site.
- Clinical decision-making requires trustworthy uncertainty estimates,
  not just raw softmax scores.
- Per-site calibration temperatures integrate naturally with the existing
  weighted ensemble (ResNet-18 × 0.30, DenseNet-121 × 0.40, EfficientNet-B0
  × 0.30) in inference.py.

Key components:
- ``TemperatureScaler``: Post-hoc temperature scaling (Guo et al., 2017).
  Fits a single scalar T on a held-out calibration set without modifying
  any model weights.
- ``compute_calibration_metrics()``: ECE, MCE, accuracy, avg-confidence.
- ``plot_reliability_diagram()``: Visual calibration curve + confidence
  histogram.
- ``HospitalCalibrationRegistry``: Tracks per-hospital temperatures and
  metrics for ensemble weight adjustment and the Model Registry.

Usage::

    # After training, calibrate on a held-out validation loader:
    scaler = TemperatureScaler(model)
    optimal_T = scaler.fit(val_loader, device='cpu')
    print(f"Hospital_A optimal T = {optimal_T:.3f}")

    # Evaluate calibration on a test set:
    metrics = compute_calibration_metrics(logits_array, labels_array)
    print(f"ECE = {metrics['ece']:.4f},  MCE = {metrics['mce']:.4f}")

    # Plot and save reliability diagram:
    plot_reliability_diagram(
        logits_array, labels_array,
        title="Hospital_A Calibration",
        save_path="outputs/hospital_a_reliability.png",
    )

    # Register in the per-hospital registry:
    registry = HospitalCalibrationRegistry()
    registry.register("Hospital_A", optimal_T, metrics)
    print(registry.summary())
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for server environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration (Guo et al., 2017).

    A single scalar parameter *T* is optimised to minimise the negative
    log-likelihood on a held-out calibration set.  The underlying model
    weights are **not** modified — only the output logits are rescaled.

    Calibrated probability = softmax(logits / T)

    A value T > 1 softens over-confident predictions; T < 1 sharpens
    under-confident ones.

    Args:
        model: Any ``nn.Module`` that returns raw logits (pre-softmax).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        # Initialise slightly above 1 — typical networks are over-confident
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits for input *x*."""
        return self.model(x) / self.temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale pre-computed logits — faster when logits are already available."""
        return logits / self.temperature

    def fit(
        self,
        val_loader: DataLoader,
        device: str = "cpu",
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """
        Fit temperature on a validation ``DataLoader``.

        Args:
            val_loader: DataLoader yielding ``(inputs, labels)`` batches.
            device: ``'cpu'`` or ``'cuda'``.
            max_iter: Maximum L-BFGS iterations.
            lr: Learning rate for L-BFGS.

        Returns:
            Optimal temperature value (scalar float).
        """
        self.model.eval()
        self.to(device)

        # Collect all logits and labels in one pass (no gradients needed here)
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                logits = self.model(inputs)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)

        # Optimise temperature with L-BFGS (fast for a single scalar)
        nll = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = nll(logits_cat / self.temperature, labels_cat)
            loss.backward()
            return loss

        optimizer.step(_closure)

        # Keep temperature in a sensible range to prevent numerical issues
        self.temperature.data.clamp_(min=0.1, max=10.0)
        return float(self.temperature.item())


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error
    (MCE) from raw logits or probability arrays.

    Binning strategy: equal-width bins over [0, 1] confidence.

    Args:
        logits: (N, C) array of raw logits **or** class probabilities.
                If values do not sum to 1 they are treated as logits and
                converted via softmax.
        labels: (N,) integer ground-truth class indices.
        n_bins: Number of equal-width confidence bins (default 10).

    Returns:
        Dict with keys: ``ece``, ``mce``, ``accuracy``, ``avg_confidence``,
        ``n_bins``, ``n_samples``.
    """
    # Convert to probabilities if the input looks like raw logits
    if logits.ndim == 2:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)
    else:
        probs = logits

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(confidences)

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_err = abs(bin_acc - bin_conf)
        ece += (mask.sum() / n) * bin_err
        mce = max(mce, bin_err)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "accuracy": float(correct.mean()),
        "avg_confidence": float(confidences.mean()),
        "n_bins": n_bins,
        "n_samples": int(n),
    }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    logits: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a reliability diagram comparing mean predicted confidence against
    actual accuracy per confidence bin, plus a confidence histogram.

    Args:
        logits: (N, C) raw logits or probability array.
        labels: (N,) ground-truth integer labels.
        n_bins: Number of confidence bins.
        title: Figure suptitle.
        save_path: If provided, save the figure to this path.
    """
    if logits.ndim == 2:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum(axis=1, keepdims=True)
    else:
        probs = logits

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs: List[float] = []
    bin_confs: List[float] = []
    bin_sizes: List[int] = []
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        bin_accs.append(float(correct[mask].mean()))
        bin_confs.append(float(confidences[mask].mean()))
        bin_sizes.append(int(mask.sum()))

    metrics = compute_calibration_metrics(logits, labels, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Reliability curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    bar_width = 0.9 / n_bins
    ax.bar(
        bin_confs, bin_accs,
        width=bar_width, alpha=0.7, color="steelblue",
        label="Model",
    )
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Predictions")
    ax.set_title(
        f"ECE = {metrics['ece']:.4f}  |  MCE = {metrics['mce']:.4f}"
    )
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Confidence histogram
    ax2 = axes[1]
    ax2.bar(bin_confs, bin_sizes, width=bar_width, alpha=0.7, color="coral")
    ax2.set_xlabel("Confidence Bin")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Confidence Distribution")

    plt.tight_layout()

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Reliability diagram saved to {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-hospital calibration registry
# ---------------------------------------------------------------------------

class HospitalCalibrationRegistry:
    """
    Tracks per-hospital temperature scalers and calibration metrics.

    In the VFL setting each hospital operates on a potentially different
    chest-X-ray distribution (scanner vendor, patient demographics, disease
    prevalence).  This registry stores the per-site calibration temperature
    and ECE/MCE metrics so that:

    - The ensemble weighting in ``inference.py`` can be adjusted to
      down-weight poorly-calibrated hospitals.
    - Calibration quality can be surfaced in the Streamlit Model Registry
      page alongside ROC curves and confusion matrices.
    - Per-site calibration hashes can be anchored on-chain as additional
      model governance metadata.
    """

    def __init__(self) -> None:
        self._temperatures: Dict[str, float] = {}
        self._metrics: Dict[str, Dict[str, float]] = {}

    def register(
        self,
        hospital_id: str,
        temperature: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Register calibration results for a hospital.

        Args:
            hospital_id: Unique hospital / client identifier.
            temperature: Optimal temperature scalar from ``TemperatureScaler.fit()``.
            metrics: Optional dict from ``compute_calibration_metrics()``.
        """
        self._temperatures[hospital_id] = temperature
        if metrics is not None:
            self._metrics[hospital_id] = metrics

    def get_temperature(self, hospital_id: str, default: float = 1.0) -> float:
        """Return the calibration temperature for *hospital_id*, or *default*."""
        return self._temperatures.get(hospital_id, default)

    def get_metrics(self, hospital_id: str) -> Optional[Dict[str, float]]:
        """Return calibration metrics for *hospital_id*, or ``None``."""
        return self._metrics.get(hospital_id)

    def all_hospitals(self) -> List[str]:
        """Return list of calibrated hospital IDs."""
        return list(self._temperatures.keys())

    def summary(self) -> Dict[str, Any]:
        """Return an aggregate summary across all registered hospitals."""
        if not self._temperatures:
            return {"calibrated_hospitals": 0}
        temps = list(self._temperatures.values())
        eces = [m["ece"] for m in self._metrics.values() if "ece" in m]
        summary: Dict[str, Any] = {
            "calibrated_hospitals": len(self._temperatures),
            "mean_temperature": float(np.mean(temps)),
            "std_temperature": float(np.std(temps)),
            "per_hospital": {
                h: {"temperature": self._temperatures[h]}
                for h in self._temperatures
            },
        }
        if eces:
            summary["mean_ece"] = float(np.mean(eces))
            summary["std_ece"] = float(np.std(eces))
            for h in self._metrics:
                summary["per_hospital"][h].update(self._metrics[h])
        return summary
