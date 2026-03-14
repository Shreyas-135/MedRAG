# Model Governance & Provenance Verification

This document explains the two blockchain capstone features added to MedRAG,
how to run them with a local Ganache node, and how to exercise each workflow
end-to-end.

---

## Overview

| Feature | Purpose |
|---------|---------|
| **Provenance Verification Gate** | Cryptographically anchors every inference on-chain. Export of results is **blocked** until the bundle hash is anchored and verified. |
| **Model Governance Multi-Sig** | Trained model versions must receive ≥ 3-of-4 hospital approvals before inference is allowed. |

Both features are designed to run on **local Ganache** (HTTP RPC).  
No raw medical data is put on-chain – only SHA-256 hashes and Ethereum addresses.

---

## Files Added / Modified

| Path | Purpose |
|------|---------|
| `src/ModelGovernanceRegistry.sol` | Solidity contract; 3-of-4 multi-sig approval, `ModelRegistered / ModelApprovalRecorded / ModelApproved` events |
| `src/model_governance_integrator.py` | Python integration; deploys/connects to the contract, registers models, submits approvals, queries status. Supports mock mode. |
| `webapp/pages/1_🔬_Inference.py` | Governance gate before inference + provenance verification gate blocking export |
| `webapp/utils.py` | `check_model_governance_approval()` helper |
| `scripts/run_pipeline.py` | `--with-governance` CLI flag |
| `tests/test_model_governance.py` | Unit tests (mock mode, no Ganache required) |
| `.env.example` | Documents new `GANACHE_URL`, `GOVERNANCE_*` env vars |

---

## Prerequisites

```bash
# Install Ganache (desktop app or CLI)
npm install -g ganache

# Python dependencies (already in requirements.txt)
pip install web3 py-solc-x eth-tester py-evm
```

---

## Feature 1: Provenance Verification Gate

### What Changed

After every inference run the UI now:

1. Builds a canonical provenance bundle (hashes only).
2. Displays an **⚠️ UNVERIFIED** banner.
3. Walks the user through MetaMask signing → on-chain anchoring → verification.
4. Sets the result to **✅ VERIFIED** and unlocks the **Download Result & Provenance (JSON)** button only after the bundle hash is confirmed on-chain.
5. Provides a **mock anchor** button (no Ganache) for demo/test environments.
6. Provides an **Admin Override** expander that bypasses the gate with a warning banner.

### Running with Ganache

```bash
# 1. Start Ganache (default port 7545)
ganache --port 7545

# 2. Copy an account private key from Ganache output
# 3. Open the webapp
streamlit run webapp/app.py

# 4. Upload an X-ray, click Analyze Image.
# 5. In the Provenance section, enter your Ganache private key and click
#    "Anchor Provenance On-Chain".  The VERIFIED banner appears and the
#    download button unlocks.
```

### Mock mode (no Ganache)

Click **🧪 Demo: Mock Anchor (no Ganache)** – the bundle is stored in an
in-process dict, verification succeeds, and the download button unlocks.
This mode is suitable for development and CI.

---

## Feature 2: Model Governance Multi-Sig Approval

### Contract: `ModelGovernanceRegistry.sol`

```solidity
constructor(uint256 _requiredApprovals)  // default 3

registerModel(bytes32 modelHash, bytes32 metadataHash)   // admin only, sets PENDING
approveModel(bytes32 modelHash)                          // any hospital address, max once
isApproved(bytes32 modelHash) → bool
getStatus(bytes32 modelHash)  → uint8  // 0=UNKNOWN 1=PENDING 2=APPROVED
getApprovalCount(bytes32 modelHash) → uint256
```

Events: `ModelRegistered`, `ModelApprovalRecorded`, `ModelApproved`.

### Python Integration: `model_governance_integrator.py`

```python
from src.model_governance_integrator import ModelGovernanceIntegrator

# --- Real Ganache ---
integrator = ModelGovernanceIntegrator(
    rpc_url="http://127.0.0.1:7545",
    admin_key="0x<admin-private-key>",          # deploys contract automatically
    # contract_address="0x…"                    # optional: use pre-deployed
    required_approvals=3,
)

# Compute stable hash for a model version
model_hash = ModelGovernanceIntegrator.compute_model_version_hash(
    version_id="v1.0_round5_20231223_143022",
    model_hash="<sha256-from-model-registry>",
)

integrator.register_model(model_hash)                              # PENDING
integrator.approve_model(model_hash, hospital_key="0x<hosp1-key>")
integrator.approve_model(model_hash, hospital_key="0x<hosp2-key>")
integrator.approve_model(model_hash, hospital_key="0x<hosp3-key>")  # → APPROVED

print(integrator.get_approval_status(model_hash))
# {'status': 'APPROVED', 'approval_count': 3, 'required_approvals': 3, 'is_approved': True}

# --- Mock mode (no Ganache) ---
integrator = ModelGovernanceIntegrator(use_mock=True)
```

### Training Pipeline Integration

```bash
# Set environment variables first
export GANACHE_URL=http://127.0.0.1:7545
export GOVERNANCE_ADMIN_KEY=0x<admin-private-key>
export GOVERNANCE_HOSPITAL_KEYS=0x<hosp1>,0x<hosp2>,0x<hosp3>

# Run pipeline with governance enabled
python scripts/run_pipeline.py \
    --datapath ./data \
    --use-rag \
    --with-governance

# The pipeline will:
#   1. Train the VFL model
#   2. Update the model registry
#   3. Register the new version as PENDING on-chain
#   4. Submit approvals from each GOVERNANCE_HOSPITAL_KEYS address
#   5. Report APPROVED when ≥3 approvals are received
```

If `GOVERNANCE_ADMIN_KEY` is not set, the governance step runs in **mock mode**
(in-process dict, no Ganache required) and prints `ℹ️  running in mock mode`.

### Inference Governance Gate

The inference page (`webapp/pages/1_🔬_Inference.py`) checks governance status
before allowing an image to be analyzed:

| Condition | Behaviour |
|-----------|-----------|
| Mock mode (no env vars) | Green info banner; inference allowed |
| Ganache connected, model APPROVED | Green success banner; inference allowed |
| Ganache connected, model NOT APPROVED | Red error banner; **Analyze Image button blocked** |

Set these env vars to activate the live gate:
```bash
export GANACHE_URL=http://127.0.0.1:7545
export GOVERNANCE_ADMIN_KEY=0x<admin-key>
export GOVERNANCE_CONTRACT_ADDRESS=0x<deployed-contract>
```

---

## Running Tests

```bash
# Governance tests (no Ganache required)
python tests/test_model_governance.py

# Provenance tests (no Ganache required)
python tests/test_provenance.py
```

---

## Gas Usage Notes

- `registerModel`: ~60 000 gas (one storage write + event)
- `approveModel`:  ~45 000 gas (one storage read/write + event; +5 000 on final approval)
- `isApproved` / `getStatus` / `getApprovalCount`: view calls (0 gas)
