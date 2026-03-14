# Cryptographic Provenance Anchoring for RAG Citations

This document explains the end-to-end cryptographic provenance anchoring
system added to MedRAG. It covers the bundle format, how to run Ganache,
how to use MetaMask to sign bundles, how to anchor and verify on-chain,
and security considerations.

---

## Overview

Every inference run in MedRAG produces a **provenance bundle** — a canonical
JSON document containing *only hashes* (no raw text). The bundle is signed by
the operator via MetaMask (EIP-191 `personal_sign`) and anchored on a local
Ethereum chain (Ganache) via the `ProvenanceRegistry` smart contract.

```
Inference → Provenance Bundle → MetaMask Sign → Ganache Anchor
                    │                    │               │
              bundle_hash           signature        tx_hash
```

---

## Files Added / Modified

| Path | Purpose |
|------|---------|
| `src/ProvenanceRegistry.sol` | Minimal Solidity contract; stores anchors and emits `ProvenanceAnchored` events |
| `src/provenance.py` | Bundle construction, hash helpers, signature verification |
| `src/provenance_integrator.py` | Web3/Ganache integration; deploys or connects to `ProvenanceRegistry` |
| `src/langchain_rag.py` | Extended `query()` to return `retrieval_hash`, `prompt_hash`, `generation_params_hash`, and `provenance_bundle` |
| `webapp/pages/1_🔬_Inference.py` | Added 3-step provenance UI (bundle display → MetaMask sign → anchor on-chain) |
| `tests/test_provenance.py` | Unit + integration tests (no Ganache required) |

---

## Provenance Bundle Format

```json
{
  "bundle_version": "1.0",
  "timestamp": 1700000000.0,
  "hospital_id": "hospital-1",
  "site_id": "site-A",
  "device_id": "device-X",
  "model_version_hash": "<sha256-hex>",
  "knowledge_base_hash": "<sha256-hex>",
  "explanation_hash": "<sha256-hex>",
  "retrieval_hash": "<sha256-hex>",
  "prompt_hash": "<sha256-hex>",
  "generation_params_hash": "<sha256-hex>",
  "bundle_hash": "<sha256-hex>"
}
```

### Field descriptions

| Field | Source |
|-------|--------|
| `bundle_version` | Schema version (currently `"1.0"`) |
| `timestamp` | Unix epoch seconds at inference time |
| `hospital_id` / `site_id` / `device_id` | Configurable via `provenance_config` kwarg |
| `model_version_hash` | `sha256({"version_id": ..., "model_hash": ...})` |
| `knowledge_base_hash` | Existing `ChromaDBMedicalKnowledgeBase.get_hash()` |
| `explanation_hash` | Existing `_compute_explanation_hash()` in the pipeline |
| `retrieval_hash` | `sha256({"item_ids": [...], "similarity_scores": [...], "top_k": N, "reranker_params": {}})` |
| `prompt_hash` | `sha256(explanation_text)` |
| `generation_params_hash` | `sha256({"temperature": T, "max_tokens": N, "model_id": ...})` |
| `bundle_hash` | `sha256(canonical_json_of_all_above_fields)` |

`bundle_hash` is computed as:

```python
import hashlib, json

canonical = {k: v for k, v in bundle.items() if k != "bundle_hash"}
content = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
bundle_hash = hashlib.sha256(content.encode()).hexdigest()
```

---

## How to Run Ganache

1. Install Ganache CLI:
   ```bash
   npm install -g ganache
   ```

2. Start a local chain (network ID 1337, chainId 1337, port 7545):
   ```bash
   ganache --port 7545 --networkId 1337 --chainId 1337 --deterministic
   ```
   The `--deterministic` flag gives you reproducible accounts and private keys.

3. Note the first account address and private key from the Ganache output, e.g.:
   ```
   Available Accounts
   ==================
   (0) 0x90F8bf6A479f320ead074411a4B0e7944Ea8c9C1 (100 ETH)

   Private Keys
   ==================
   (0) 0x4f3edf983ac636a65a842ce7c78d9aa706d3b113b3a462f9d8edeb5a5f8ef57e
   ```

---

## How to Use MetaMask

1. Install the [MetaMask browser extension](https://metamask.io/).

2. Add the local Ganache network:
   - **Network Name**: Ganache Local
   - **RPC URL**: `http://127.0.0.1:7545`
   - **Chain ID**: `1337`
   - **Currency Symbol**: ETH

3. Import the Ganache account using its private key (Account → Import Account → Paste key).

4. When the Streamlit inference page shows the **Sign Bundle Hash** button, MetaMask
   will pop up asking you to sign the bundle hash string. Approve the signature.

5. Copy the **Address** and **Signature** from the result box and paste them into
   the Streamlit form fields.

---

## How to Anchor and Verify

### Via the Streamlit webapp

1. Upload an X-ray and click **Analyze Image**.
2. Scroll to **🔏 Cryptographic Provenance Anchoring**.
3. Review the provenance bundle and copy the bundle hash.
4. Click **🦊 Sign Bundle Hash** — MetaMask popup appears; approve.
5. Paste the signer address and signature into the form.
6. Enter the Ganache RPC URL and the Ganache account private key.
7. Click **🚀 Anchor Provenance On-Chain**.
8. The page will display the transaction hash and confirm on-chain verification.

For a quick demo without Ganache, click **🧪 Demo: Mock Anchor (no Ganache)** instead.

### Via Python

```python
from src.provenance import build_provenance_bundle, verify_bundle, verify_signature
from src.provenance_integrator import ProvenanceIntegrator

# 1. Build bundle (typically returned by LangChainRAGPipeline.query())
bundle = build_provenance_bundle(
    knowledge_base_hash="<64-hex>",
    explanation_hash="<64-hex>",
    retrieval_hash="<64-hex>",
    model_version_hash="<64-hex>",
    prompt_hash="<64-hex>",
    generation_params_hash="<64-hex>",
    hospital_id="hospital-1",
)

# 2. Anchor on Ganache
integrator = ProvenanceIntegrator(
    rpc_url="http://127.0.0.1:7545",
    private_key="0x<ganache-private-key>",
)
tx_hash = integrator.anchor_provenance(
    bundle_hash=bundle["bundle_hash"],
    model_hash=bundle["model_version_hash"],
    kb_hash=bundle["knowledge_base_hash"],
    explanation_hash=bundle["explanation_hash"],
    signer_address="0x<metamask-address>",
)
print(f"Anchored: {tx_hash}")

# 3. Verify on-chain
print(integrator.is_anchored(bundle["bundle_hash"]))  # True

# 4. Verify signature (optional, requires MetaMask signature)
valid = verify_signature(bundle["bundle_hash"], "<0x-signature>", "<0x-address>")

# 5. Verify bundle integrity
result = verify_bundle(bundle, signature="<0x-sig>", signer_address="<0x-addr>")
print(result)  # {"hash_valid": True, "signature_valid": True}
```

---

## Running the Tests

Tests run entirely without Ganache (mock mode):

```bash
# From repository root
python -m pytest tests/test_provenance.py -v
```

Expected output: all tests pass including hash helpers, bundle construction,
signature verification (with deterministic test key), bundle integrity checks,
and mock integrator tests.

---

## Security Notes

- **Hash-only on-chain**: No raw text (prompts, explanations, knowledge base content)
  is ever stored in the provenance bundle or on-chain. Only SHA-256 hashes are anchored,
  preserving patient privacy and minimising data exposure.

- **Signature proves origin**: The MetaMask EIP-191 `personal_sign` signature binds the
  `bundle_hash` to a specific Ethereum address. This proves that the key-holder endorsed
  the inference at the given point in time, without disclosing the underlying data.

- **Tamper evidence**: Any modification to the bundle fields (IDs, hashes, timestamps)
  will produce a different `bundle_hash`, causing `verify_bundle()` to return
  `hash_valid: False`. The on-chain record is immutable once anchored.

- **Private key handling**: Never commit Ganache or MetaMask private keys to source
  control. Use environment variables or a secrets manager for production deployments.

- **Ganache is for development**: The local Ganache chain (chainId 1337) is suitable
  for demos and integration tests. For production, replace with a permissioned or
  public chain and audit the `ProvenanceRegistry` contract before deployment.
