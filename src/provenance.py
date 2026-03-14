"""
Cryptographic Provenance Anchoring for RAG Citations

Provides utilities to:
- Build a canonical provenance bundle (hashes only, no raw text)
- Compute a deterministic bundle_hash (SHA-256)
- Verify EIP-191 signatures produced by MetaMask
- Verify bundle integrity by recomputing the hash

Bundle format (canonical JSON, sorted keys):
{
    "bundle_version": "1.0",
    "timestamp": <float epoch seconds>,
    "hospital_id": <str>,
    "site_id": <str>,
    "device_id": <str>,
    "model_version_hash": <sha256 hex>,
    "knowledge_base_hash": <sha256 hex>,
    "explanation_hash": <sha256 hex>,
    "retrieval_hash": <sha256 hex>,
    "prompt_hash": <sha256 hex>,
    "generation_params_hash": <sha256 hex>
}
bundle_hash = sha256(canonical_json)
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _sha256_json(obj: Any) -> str:
    """Return hex SHA-256 of the canonical JSON representation of *obj*."""
    content = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


def hash_prompt(prompt: str) -> str:
    """Return SHA-256 of a prompt string."""
    return hashlib.sha256(prompt.encode()).hexdigest()


def hash_generation_params(
    temperature: float,
    max_tokens: int,
    model_id: str,
    **kwargs: Any,
) -> str:
    """
    Return SHA-256 of generation parameters.

    Args:
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        model_id: Model identifier string.
        **kwargs: Any additional generation parameters.

    Returns:
        Hex SHA-256 string.
    """
    params: Dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "model_id": model_id,
    }
    params.update(kwargs)
    return _sha256_json(params)


def hash_retrieval_params(
    item_ids: List[str],
    similarity_scores: List[float],
    top_k: int,
    reranker_params: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return SHA-256 over retrieved item IDs, similarity scores, top_k, and
    optional reranker parameters.

    Scores are rounded to 8 decimal places to ensure cross-platform
    determinism for floating-point values.

    Args:
        item_ids: List of retrieved document/case IDs.
        similarity_scores: Corresponding similarity scores.
        top_k: Number of top results requested.
        reranker_params: Optional reranker configuration dict.

    Returns:
        Hex SHA-256 string.
    """
    payload = {
        "item_ids": item_ids,
        "similarity_scores": [round(float(s), 8) for s in similarity_scores],
        "top_k": top_k,
        "reranker_params": reranker_params or {},
    }
    return _sha256_json(payload)


def hash_model_version(version_id: str, model_hash: str = "") -> str:
    """
    Return SHA-256 of the model version information.

    Args:
        version_id: Version identifier from ModelRegistry (or 'unknown').
        model_hash: Pre-computed model weight hash (empty string if unavailable).

    Returns:
        Hex SHA-256 string.
    """
    return _sha256_json({"version_id": version_id, "model_hash": model_hash})


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------

def build_provenance_bundle(
    knowledge_base_hash: str,
    explanation_hash: str,
    retrieval_hash: str,
    model_version_hash: str,
    prompt_hash: str,
    generation_params_hash: str,
    hospital_id: str = "unknown",
    site_id: str = "unknown",
    device_id: str = "unknown",
    bundle_version: str = "1.0",
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a canonical provenance bundle and compute its bundle_hash.

    All sensitive content is represented as hashes only; no raw text is
    stored in the bundle.

    Args:
        knowledge_base_hash: SHA-256 of the knowledge base state.
        explanation_hash: SHA-256 of the generated explanation.
        retrieval_hash: SHA-256 of retrieved IDs + scores + top_k.
        model_version_hash: SHA-256 of model version metadata.
        prompt_hash: SHA-256 of the inference prompt.
        generation_params_hash: SHA-256 of generation parameters.
        hospital_id: Configurable hospital/institution identifier.
        site_id: Configurable site identifier.
        device_id: Configurable device identifier.
        bundle_version: Schema version string.
        timestamp: Unix epoch seconds (uses current time if None).

    Returns:
        Dict containing all fields plus ``bundle_hash``.
    """
    bundle: Dict[str, Any] = {
        "bundle_version": bundle_version,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "hospital_id": hospital_id,
        "site_id": site_id,
        "device_id": device_id,
        "model_version_hash": model_version_hash,
        "knowledge_base_hash": knowledge_base_hash,
        "explanation_hash": explanation_hash,
        "retrieval_hash": retrieval_hash,
        "prompt_hash": prompt_hash,
        "generation_params_hash": generation_params_hash,
    }
    bundle["bundle_hash"] = compute_bundle_hash(bundle)
    return bundle


def compute_bundle_hash(bundle: Dict[str, Any]) -> str:
    """
    Compute the SHA-256 hash of the canonical provenance bundle.

    The ``bundle_hash`` field itself is excluded before hashing so the
    operation is idempotent when called on a complete bundle.

    Args:
        bundle: Provenance bundle dict (with or without bundle_hash).

    Returns:
        Hex SHA-256 string.
    """
    canonical = {k: v for k, v in bundle.items() if k != "bundle_hash"}
    return _sha256_json(canonical)


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

def verify_signature(
    bundle_hash: str,
    signature: str,
    signer_address: str,
) -> bool:
    """
    Verify an EIP-191 ``personal_sign`` signature over the bundle_hash.

    MetaMask's ``personal_sign`` signs the UTF-8 string of the hex bundle
    hash prefixed with the Ethereum personal message header.

    Args:
        bundle_hash: Hex string (64 chars, without ``0x``) of the bundle hash.
        signature: Hex signature returned by MetaMask (with or without ``0x``).
        signer_address: Expected Ethereum address of the signer.

    Returns:
        ``True`` if the recovered address matches *signer_address*.
    """
    from eth_account import Account
    from eth_account.messages import encode_defunct

    message = encode_defunct(text=bundle_hash)
    try:
        recovered = Account.recover_message(message, signature=signature)
        return recovered.lower() == signer_address.lower()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Bundle integrity verification
# ---------------------------------------------------------------------------

def verify_bundle(
    bundle: Dict[str, Any],
    signature: Optional[str] = None,
    signer_address: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Verify the integrity of a provenance bundle.

    Recomputes ``bundle_hash`` from the bundle fields and optionally checks
    the EIP-191 signature.

    Args:
        bundle: Provenance bundle dict (must contain ``bundle_hash``).
        signature: Optional MetaMask signature hex string.
        signer_address: Optional expected signer Ethereum address.

    Returns:
        Dict with keys:
          - ``hash_valid`` (bool): True if recomputed hash matches stored hash.
          - ``signature_valid`` (bool): Present only when signature/address given.
    """
    result: Dict[str, bool] = {}

    recomputed = compute_bundle_hash(bundle)
    result["hash_valid"] = recomputed == bundle.get("bundle_hash", "")

    if signature is not None and signer_address is not None:
        stored_hash = bundle.get("bundle_hash", "")
        result["signature_valid"] = verify_signature(
            stored_hash, signature, signer_address
        )

    return result
