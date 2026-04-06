"""
Federated Knowledge Base Manager (FedKBManager)

Enables privacy-preserving, cross-hospital contributions to the shared
ChromaDB knowledge base without any raw patient data leaving hospital
boundaries, directly supporting the project's core privacy-preserving
cross-hospital medical imaging goal.

Protocol (per federated round):
  1. Each hospital client computes per-class centroid embeddings from its
     correctly-classified samples (local to that client).
  2. Calibrated Gaussian noise is applied to each centroid via
     ``apply_local_dp_noise()`` — this is the local DP boundary.
  3. The client calls ``make_contribution()`` which hashes the finding text
     (so no raw text leaves the hospital) and wraps everything in a
     ``KBContribution``.
  4. The server calls ``FedKBManager.submit()`` for each received
     contribution, then ``FedKBManager.aggregate_round()`` after all clients
     have submitted.
  5. The manager performs FedAvg on the noised embeddings and admits the
     averaged entry to ChromaDB when at least ``min_hospitals`` hospitals
     contributed to the same (condition, severity) pair.
  6. A SHA-256 commitment of each round's contributions is appended to an
     append-only JSON-lines audit log.  ``get_round_commitment()`` returns
     this hash for optional on-chain anchoring via FederatedKBRegistry.sol.

Privacy guarantee:
    Gaussian mechanism with σ = dp_sigma applied to each centroid before it
    leaves the hospital.  Only the averaged entry (after FedAvg across ≥
    min_hospitals contributions) enters the shared KB.

Usage::

    # Hospital-side (inside FlowerMedicalClient.fit or post-fit hook):
    centroid = compute_class_centroids(embeddings, labels, class_names)
    for condition, c in centroid.items():
        contrib = make_contribution(
            hospital_id="Hospital_A",
            round_id=server_round,
            condition=condition,
            severity="moderate",
            centroid=c,
            finding_text="Bilateral ground-glass opacities consistent with viral pneumonia.",
        )
        # send contrib to server

    # Server-side (inside FlowerVFLStrategy.aggregate_fit):
    for contrib in received_contributions:
        fed_kb_manager.submit(contrib)
    new_entries = fed_kb_manager.aggregate_round(round_id=server_round)
    commitment_hash = fed_kb_manager.get_round_commitment(server_round)
    # anchor commitment_hash on FederatedKBRegistry.sol
"""

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KBContribution:
    """A single hospital's DP-noised contribution for one FL round."""

    hospital_id: str
    round_id: int
    condition: str            # e.g. 'covid', 'normal'
    severity: str             # e.g. 'moderate', 'none'
    noised_embedding: np.ndarray  # DP-noised centroid — never raw patient data
    text_hash: str            # SHA-256 of the original finding text
    finding_snippet: str      # Short non-identifying summary (≤ 120 chars)
    timestamp: float = field(default_factory=time.time)
    contribution_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Hospital-side helpers
# ---------------------------------------------------------------------------

def compute_class_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Compute per-class mean embeddings from a batch of correctly-classified
    samples.

    Args:
        embeddings: (N, D) array of embedding vectors.
        labels: (N,) integer class indices matching *class_names*.
        class_names: Ordered list mapping integer index → class name.

    Returns:
        Dict mapping class_name → mean centroid vector of shape (D,).
        Classes with no samples in the batch are omitted.
    """
    centroids: Dict[str, np.ndarray] = {}
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if mask.sum() > 0:
            centroids[class_name] = embeddings[mask].mean(axis=0)
    return centroids


def apply_local_dp_noise(
    centroid: np.ndarray,
    dp_sigma: float = 0.1,
    sensitivity: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Add calibrated Gaussian noise for local differential privacy.

    Noise scale = dp_sigma * sensitivity  (Gaussian mechanism).
    The noised centroid is L2-normalised before returning so it stays on
    the unit hypersphere expected by cosine-similarity retrieval.

    Args:
        centroid: Original (non-noised) centroid embedding.
        dp_sigma: Noise multiplier (larger → more private, less accurate).
        sensitivity: L2 sensitivity of the centroid (typically 1/n_samples).
        seed: Optional RNG seed for reproducible unit tests.

    Returns:
        Noised, L2-normalised embedding of the same shape as *centroid*.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=dp_sigma * sensitivity, size=centroid.shape)
    noised = centroid + noise
    norm = np.linalg.norm(noised)
    if norm > 0:
        noised = noised / norm
    return noised


def make_contribution(
    hospital_id: str,
    round_id: int,
    condition: str,
    severity: str,
    centroid: np.ndarray,
    finding_text: str,
    dp_sigma: float = 0.1,
    sensitivity: float = 1.0,
) -> KBContribution:
    """
    Build a privacy-preserving ``KBContribution`` from a per-class centroid.

    The raw *finding_text* is hashed (SHA-256) before being packaged; only
    the first 120 characters are retained as a non-identifying snippet.  The
    centroid is DP-noised before it leaves the hospital.

    Args:
        hospital_id: Identifier for the contributing hospital.
        round_id: Current Flower federated learning round number.
        condition: Medical condition label (e.g. 'covid').
        severity: Severity label (e.g. 'moderate').
        centroid: Raw (non-noised) per-class centroid embedding.
        finding_text: Short textual description of the clinical finding.
        dp_sigma: Gaussian noise multiplier.
        sensitivity: L2 sensitivity of the centroid.

    Returns:
        ``KBContribution`` ready to submit to ``FedKBManager``.
    """
    noised = apply_local_dp_noise(centroid, dp_sigma=dp_sigma, sensitivity=sensitivity)
    text_hash = hashlib.sha256(finding_text.encode()).hexdigest()
    snippet = finding_text[:120]
    return KBContribution(
        hospital_id=hospital_id,
        round_id=round_id,
        condition=condition,
        severity=severity,
        noised_embedding=noised,
        text_hash=text_hash,
        finding_snippet=snippet,
    )


# ---------------------------------------------------------------------------
# Server-side aggregation manager
# ---------------------------------------------------------------------------

class FedKBManager:
    """
    Server-side manager for federated knowledge base construction.

    Collects DP-noised embeddings from participating hospitals, performs
    FedAvg aggregation on the embeddings, and inserts the averaged entry
    into the ChromaDB knowledge base.

    An append-only JSON-lines audit log records every round's contribution
    hashes.  ``get_round_commitment()`` returns a single SHA-256 hash over
    all entries added in a round, suitable for on-chain anchoring via
    ``FederatedKBRegistry.sol``.

    Args:
        knowledge_base: A ``ChromaDBMedicalKnowledgeBase`` instance.
        min_hospitals: Minimum hospitals that must contribute to a given
                       (condition, severity) pair before it is admitted to
                       the KB.  Guards against single-hospital memorisation.
        audit_log_path: Path for the append-only JSONL audit log file.
    """

    def __init__(
        self,
        knowledge_base,
        min_hospitals: int = 2,
        audit_log_path: str = "./fed_kb_audit.jsonl",
    ) -> None:
        self.knowledge_base = knowledge_base
        self.min_hospitals = min_hospitals
        self.audit_log_path = audit_log_path

        # Pending contributions: round_id → condition_key → [KBContribution]
        self._pending: Dict[int, Dict[str, List[KBContribution]]] = {}

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(self, contribution: KBContribution) -> None:
        """
        Accept a contribution from one hospital for a given round.

        Args:
            contribution: A ``KBContribution`` produced by ``make_contribution()``
                          on the hospital side.
        """
        key = contribution.condition
        round_bucket = self._pending.setdefault(contribution.round_id, {})
        round_bucket.setdefault(key, []).append(contribution)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_round(self, round_id: int) -> List[Dict[str, Any]]:
        """
        Aggregate all contributions for *round_id* and add entries to the KB.

        Entries are only admitted when at least ``min_hospitals`` hospitals
        contributed the same (condition, severity) pair.  FedAvg is applied
        over the DP-noised embeddings.

        Args:
            round_id: The federated learning round to finalise.

        Returns:
            List of entry dicts that were actually added to the KB, each
            containing ``{entry_id, condition, severity, num_hospitals,
            round_id}``.
        """
        round_bucket = self._pending.pop(round_id, {})
        added_entries: List[Dict[str, Any]] = []
        audit_records: List[Dict[str, Any]] = []

        for condition, contributions in round_bucket.items():
            # Group by severity
            by_severity: Dict[str, List[KBContribution]] = {}
            for c in contributions:
                by_severity.setdefault(c.severity, []).append(c)

            for severity, group in by_severity.items():
                if len(group) < self.min_hospitals:
                    # Not enough contributors — skip to prevent memorisation
                    continue

                # FedAvg: mean of DP-noised embeddings, then re-normalise
                stacked = np.stack([g.noised_embedding for g in group], axis=0)
                avg_embedding = stacked.mean(axis=0)
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm

                # Use the most common snippet as the canonical text label
                snippets = [g.finding_snippet for g in group]
                text = max(set(snippets), key=snippets.count)

                metadata = {
                    "condition": condition,
                    "severity": severity,
                    "federated": True,
                    "round_id": round_id,
                    "num_hospitals": len(group),
                    "text_hashes": [g.text_hash for g in group],
                }
                entry_id = (
                    f"fed_r{round_id}_{condition}_{severity}_{uuid.uuid4().hex[:8]}"
                )

                self.knowledge_base.add_entry(
                    text=text,
                    embedding=avg_embedding,
                    metadata=metadata,
                    entry_id=entry_id,
                )

                entry = {
                    "entry_id": entry_id,
                    "condition": condition,
                    "severity": severity,
                    "num_hospitals": len(group),
                    "round_id": round_id,
                }
                added_entries.append(entry)

                # Build audit record (hashes only — no raw data)
                contribution_hashes = [
                    hashlib.sha256(
                        json.dumps(
                            {
                                "id": g.contribution_id,
                                "text_hash": g.text_hash,
                                "hospital": g.hospital_id,
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest()
                    for g in group
                ]
                audit_records.append(
                    {
                        "round_id": round_id,
                        "entry_id": entry_id,
                        "condition": condition,
                        "severity": severity,
                        "num_hospitals": len(group),
                        "contribution_hashes": contribution_hashes,
                        "avg_embedding_hash": hashlib.sha256(
                            avg_embedding.tobytes()
                        ).hexdigest(),
                        "timestamp": time.time(),
                    }
                )

        self._write_audit_log(audit_records)
        return added_entries

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def _write_audit_log(self, records: List[Dict[str, Any]]) -> None:
        """Append audit records to the JSONL log file."""
        if not records:
            return
        log_dir = os.path.dirname(self.audit_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.audit_log_path, "a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, default=str) + "\n")

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Read and return all audit log entries."""
        if not os.path.exists(self.audit_log_path):
            return []
        records = []
        with open(self.audit_log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def get_round_commitment(self, round_id: int) -> str:
        """
        Compute a SHA-256 commitment hash over all entries added in *round_id*.

        This hash can be submitted to ``FederatedKBRegistry.sol`` as on-chain
        evidence of the federated KB update for that round.

        Args:
            round_id: Round to compute the commitment for.

        Returns:
            Hex SHA-256 string.
        """
        log = self.get_audit_log()
        round_records = [r for r in log if r.get("round_id") == round_id]
        if not round_records:
            return hashlib.sha256(b"empty").hexdigest()
        canonical = json.dumps(
            sorted(round_records, key=lambda r: r.get("entry_id", "")),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Return a statistics summary of federated KB contributions."""
        log = self.get_audit_log()
        total_contributions = sum(
            len(r.get("contribution_hashes", [])) for r in log
        )
        rounds = sorted({r.get("round_id") for r in log})
        return {
            "total_contributions_logged": total_contributions,
            "rounds_completed": len(rounds),
            "round_ids": rounds,
            "kb_stats": self.knowledge_base.get_stats(),
        }
