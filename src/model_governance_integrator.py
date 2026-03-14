"""
Model Governance Integrator - On-Chain Multi-Sig Approval via Web3 / Ganache

Deploys or connects to ModelGovernanceRegistry.sol and manages model
registration and hospital approval workflows.

Usage (real Ganache):
    integrator = ModelGovernanceIntegrator(
        rpc_url="http://127.0.0.1:7545",
        contract_path="/path/to/ModelGovernanceRegistry.sol",
        admin_key="0x<ganache-admin-private-key>",
    )
    integrator.register_model(model_hash="<64-hex>")
    integrator.approve_model(model_hash="<64-hex>",
                             hospital_key="0x<hospital-private-key>")
    status = integrator.get_approval_status("<64-hex>")

Usage (mock / testing):
    integrator = ModelGovernanceIntegrator(use_mock=True)
    integrator.register_model(model_hash="<64-hex>")
    integrator.approve_model(model_hash="<64-hex>",
                             hospital_key="0x<any-key>")
    assert integrator.is_approved("<64-hex>") is True
"""

import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from web3 import Web3
from web3.types import TxReceipt


# ---------------------------------------------------------------------------
# ABI for ModelGovernanceRegistry.sol  (pre-compiled for convenience)
# ---------------------------------------------------------------------------

MODEL_GOVERNANCE_REGISTRY_ABI = [
    # Events
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "bytes32", "name": "modelHash",     "type": "bytes32"},
            {"indexed": False, "internalType": "bytes32", "name": "metadataHash",  "type": "bytes32"},
            {"indexed": True,  "internalType": "address", "name": "registeredBy",  "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",     "type": "uint256"},
        ],
        "name": "ModelRegistered",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "bytes32", "name": "modelHash",     "type": "bytes32"},
            {"indexed": True,  "internalType": "address", "name": "hospital",      "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "approvalCount", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",     "type": "uint256"},
        ],
        "name": "ModelApprovalRecorded",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "bytes32", "name": "modelHash",  "type": "bytes32"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",  "type": "uint256"},
        ],
        "name": "ModelApproved",
        "type": "event",
    },
    # Constructor
    {
        "inputs": [{"internalType": "uint256", "name": "_requiredApprovals", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "constructor",
    },
    # Functions
    {
        "inputs": [
            {"internalType": "bytes32", "name": "modelHash",    "type": "bytes32"},
            {"internalType": "bytes32", "name": "metadataHash", "type": "bytes32"},
        ],
        "name": "registerModel",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}],
        "name": "approveModel",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}],
        "name": "getStatus",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}],
        "name": "isApproved",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}],
        "name": "getApprovalCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "requiredApprovals",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "admin",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Status codes returned by getStatus()
STATUS_UNKNOWN = 0
STATUS_PENDING = 1
STATUS_APPROVED = 2


# ---------------------------------------------------------------------------
# Helper: compile and deploy from source (requires py-solc-x + solc binary)
# ---------------------------------------------------------------------------

def _compile_and_deploy_governance(
    w3: Web3,
    contract_path: str,
    deployer_address: str,
    deployer_key: str,
    required_approvals: int = 3,
) -> str:
    """
    Compile ModelGovernanceRegistry.sol and deploy to the connected chain.

    Returns the deployed contract address.
    """
    try:
        from solcx import compile_source
    except ImportError:
        raise ImportError(
            "py-solc-x is required to compile contracts. "
            "Install with: pip install py-solc-x"
        )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if sys.platform == "win32":
        solc_binary = os.path.join(repo_root, "tests", "solc-0.8.23-win32", "solc.exe")
    elif sys.platform == "darwin":
        solc_binary = os.path.join(repo_root, "tests", "solc-0.8.23-macos", "solc-macos")
    else:
        solc_binary = os.path.join(repo_root, "tests", "solc-0.8.23-linux", "solc-static-linux")

    with open(contract_path) as fh:
        source = fh.read()

    compiled = compile_source(
        source,
        output_values=["abi", "bin"],
        solc_binary=solc_binary,
    )
    _, interface = compiled.popitem()
    bytecode = interface["bin"]
    abi = interface["abi"]

    contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(deployer_address)
    tx = contract.constructor(required_approvals).build_transaction(
        {"from": deployer_address, "nonce": nonce}
    )
    signed = w3.eth.account.sign_transaction(tx, private_key=deployer_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt: TxReceipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# ModelGovernanceIntegrator
# ---------------------------------------------------------------------------

class ModelGovernanceIntegrator:
    """
    Connects to a Web3 provider (Ganache or mock EVM) and manages model
    governance via ModelGovernanceRegistry.

    Args:
        rpc_url: HTTP RPC URL of Ganache (default ``http://127.0.0.1:7545``).
        contract_address: Pre-deployed contract address. If ``None``, the
            contract will be compiled and deployed from *contract_path*.
        contract_path: Path to ``ModelGovernanceRegistry.sol``. Required when
            *contract_address* is ``None`` and *use_mock* is ``False``.
        admin_key: Ethereum private key (hex with ``0x``) of the admin account
            used to deploy and register models.
        required_approvals: Minimum approvals needed (default 3).
        use_mock: When ``True``, an in-process dict is used instead of a real
            Ganache node. Useful for testing.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:7545",
        contract_address: Optional[str] = None,
        contract_path: Optional[str] = None,
        admin_key: Optional[str] = None,
        required_approvals: int = 3,
        use_mock: bool = False,
    ) -> None:
        self.use_mock = use_mock
        self.admin_key = admin_key
        self._required_approvals = required_approvals

        if use_mock:
            self._mock_models: Dict[str, Any] = {}
            self._mock_approvals: Dict[str, List[str]] = {}
            self._required_approvals_mock = required_approvals
            self.contract_address = "0x" + "0" * 40
            self.registry = None
            self.w3 = None
            self.admin_address = "0x" + "0" * 40
        else:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.w3.is_connected():
                raise ConnectionError(
                    f"Cannot connect to Web3 provider at {rpc_url}. "
                    "Ensure Ganache is running."
                )
            if not admin_key:
                raise ValueError("admin_key is required when use_mock=False.")

            account = self.w3.eth.account.from_key(admin_key)
            self.admin_address = account.address

            if contract_address:
                self.contract_address = Web3.to_checksum_address(contract_address)
            else:
                if not contract_path:
                    contract_path = os.path.join(
                        os.path.dirname(__file__), "ModelGovernanceRegistry.sol"
                    )
                print("Deploying ModelGovernanceRegistry.sol to Ganache…")
                self.contract_address = _compile_and_deploy_governance(
                    self.w3,
                    contract_path,
                    self.admin_address,
                    admin_key,
                    required_approvals,
                )
                print(f"  Contract deployed at: {self.contract_address}")

            self.registry = self.w3.eth.contract(
                address=self.contract_address,
                abi=MODEL_GOVERNANCE_REGISTRY_ABI,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hex_to_bytes32(self, hex_str: str) -> bytes:
        """Convert a 64-char hex string to a 32-byte value for Solidity bytes32."""
        hex_str = hex_str.removeprefix("0x")
        padded = hex_str.zfill(64)[:64]
        return bytes.fromhex(padded)

    def _send_tx(self, fn, sender_address: str, sender_key: str) -> str:
        """Build, sign, and send a transaction. Returns tx hash hex string."""
        nonce = self.w3.eth.get_transaction_count(sender_address)
        tx = fn.build_transaction({"from": sender_address, "nonce": nonce, "gas": 200_000})
        signed = self.w3.eth.account.sign_transaction(tx, private_key=sender_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash.hex() if isinstance(tx_hash, (bytes, bytearray)) else str(tx_hash)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def compute_model_version_hash(version_id: str, model_hash: str = "") -> str:
        """
        Compute a stable SHA-256 hash for a model version suitable for on-chain use.

        Args:
            version_id: Version identifier (e.g. ``v1.0_round5_20231223_143022``).
            model_hash: Optional pre-computed weight hash from ModelRegistry.

        Returns:
            64-char hex SHA-256 string.
        """
        payload = {"version_id": version_id, "model_hash": model_hash}
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    def register_model(
        self,
        model_hash: str,
        metadata_hash: str = "0" * 64,
    ) -> str:
        """
        Register a model version as PENDING on-chain.

        Args:
            model_hash: 64-char hex SHA-256 of the model version.
            metadata_hash: 64-char hex SHA-256 of model metadata (optional).

        Returns:
            Transaction hash hex string.
        """
        if self.use_mock:
            if model_hash in self._mock_models:
                raise ValueError(f"Model {model_hash[:16]}… already registered")
            self._mock_models[model_hash] = {
                "metadata_hash": metadata_hash,
                "status": STATUS_PENDING,
                "registered_at": time.time(),
            }
            self._mock_approvals[model_hash] = []
            fake_tx = "0x" + hashlib.sha256((model_hash + "register").encode()).hexdigest()
            return fake_tx

        mh = self._hex_to_bytes32(model_hash)
        meta = self._hex_to_bytes32(metadata_hash)
        fn = self.registry.functions.registerModel(mh, meta)
        return self._send_tx(fn, self.admin_address, self.admin_key)

    def approve_model(self, model_hash: str, hospital_key: str) -> str:
        """
        Submit a hospital approval for a PENDING model.

        Args:
            model_hash: 64-char hex SHA-256 of the model version.
            hospital_key: Ethereum private key (0x…) of the approving hospital.

        Returns:
            Transaction hash hex string.
        """
        if self.use_mock:
            rec = self._mock_models.get(model_hash)
            if rec is None:
                raise ValueError(f"Model {model_hash[:16]}… not registered")
            if rec["status"] == STATUS_APPROVED:
                raise ValueError(f"Model {model_hash[:16]}… already APPROVED")
            # Derive a deterministic mock "Ethereum address" from the private key
            # by truncating the SHA-256 digest to 20 bytes (40 hex chars).  Each
            # distinct hospital key maps to a distinct mock address, providing
            # sufficient uniqueness for test scenarios.
            mock_addr = "0x" + hashlib.sha256(hospital_key.encode()).hexdigest()[:40]
            if mock_addr in self._mock_approvals[model_hash]:
                raise ValueError("Hospital has already approved this model")
            self._mock_approvals[model_hash].append(mock_addr)
            count = len(self._mock_approvals[model_hash])
            if count >= self._required_approvals_mock:
                rec["status"] = STATUS_APPROVED
            fake_tx = "0x" + hashlib.sha256((model_hash + hospital_key + "approve").encode()).hexdigest()
            return fake_tx

        hospital_account = self.w3.eth.account.from_key(hospital_key)
        hospital_address = hospital_account.address
        mh = self._hex_to_bytes32(model_hash)
        fn = self.registry.functions.approveModel(mh)
        return self._send_tx(fn, hospital_address, hospital_key)

    def is_approved(self, model_hash: str) -> bool:
        """
        Return ``True`` if the model has reached APPROVED status.

        Args:
            model_hash: 64-char hex SHA-256 of the model version.

        Returns:
            Boolean.
        """
        if self.use_mock:
            rec = self._mock_models.get(model_hash)
            return rec is not None and rec["status"] == STATUS_APPROVED

        mh = self._hex_to_bytes32(model_hash)
        return self.registry.functions.isApproved(mh).call()

    def get_approval_status(self, model_hash: str) -> Dict[str, Any]:
        """
        Return a dict with status label, approval count, and required count.

        Args:
            model_hash: 64-char hex SHA-256 of the model version.

        Returns:
            Dict with keys: ``status`` (str), ``approval_count`` (int),
            ``required_approvals`` (int), ``is_approved`` (bool).
        """
        if self.use_mock:
            rec = self._mock_models.get(model_hash)
            if rec is None:
                return {
                    "status": "UNKNOWN",
                    "approval_count": 0,
                    "required_approvals": self._required_approvals_mock,
                    "is_approved": False,
                }
            status_map = {STATUS_PENDING: "PENDING", STATUS_APPROVED: "APPROVED"}
            count = len(self._mock_approvals.get(model_hash, []))
            return {
                "status": status_map.get(rec["status"], "UNKNOWN"),
                "approval_count": count,
                "required_approvals": self._required_approvals_mock,
                "is_approved": rec["status"] == STATUS_APPROVED,
            }

        mh = self._hex_to_bytes32(model_hash)
        raw_status = self.registry.functions.getStatus(mh).call()
        count = self.registry.functions.getApprovalCount(mh).call()
        req = self.registry.functions.requiredApprovals().call()
        status_map = {STATUS_UNKNOWN: "UNKNOWN", STATUS_PENDING: "PENDING", STATUS_APPROVED: "APPROVED"}
        return {
            "status": status_map.get(raw_status, "UNKNOWN"),
            "approval_count": count,
            "required_approvals": req,
            "is_approved": raw_status == STATUS_APPROVED,
        }
