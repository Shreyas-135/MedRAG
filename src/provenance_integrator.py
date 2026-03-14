"""
Provenance Integrator - On-Chain Anchoring via Web3 / Ganache

Deploys or connects to ProvenanceRegistry.sol and submits
``anchorProvenance`` transactions.

Usage (real Ganache):
    integrator = ProvenanceIntegrator(
        rpc_url="http://127.0.0.1:7545",
        contract_path="/path/to/ProvenanceRegistry.sol",
        private_key="0x<your-ganache-private-key>",
    )
    tx_hash = integrator.anchor_provenance(
        bundle_hash="<64-hex>",
        model_hash="<64-hex>",
        kb_hash="<64-hex>",
        explanation_hash="<64-hex>",
        signer_address="0x<address>",
    )

Usage (mock / testing):
    integrator = ProvenanceIntegrator(use_mock=True)
    tx_hash = integrator.anchor_provenance(...)
"""

import os
import sys
from typing import Any, Dict, Optional

from web3 import Web3
from web3.types import TxReceipt

# ---------------------------------------------------------------------------
# ABI for ProvenanceRegistry.sol  (pre-compiled for convenience)
# ---------------------------------------------------------------------------

PROVENANCE_REGISTRY_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "bytes32", "name": "bundleHash",      "type": "bytes32"},
            {"indexed": False, "internalType": "bytes32", "name": "modelHash",       "type": "bytes32"},
            {"indexed": False, "internalType": "bytes32", "name": "kbHash",          "type": "bytes32"},
            {"indexed": False, "internalType": "bytes32", "name": "explanationHash", "type": "bytes32"},
            {"indexed": True,  "internalType": "address", "name": "signer",          "type": "address"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",       "type": "uint256"},
        ],
        "name": "ProvenanceAnchored",
        "type": "event",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "bundleHash",      "type": "bytes32"},
            {"internalType": "bytes32", "name": "modelHash",       "type": "bytes32"},
            {"internalType": "bytes32", "name": "kbHash",          "type": "bytes32"},
            {"internalType": "bytes32", "name": "explanationHash", "type": "bytes32"},
            {"internalType": "address", "name": "signer",          "type": "address"},
        ],
        "name": "anchorProvenance",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "bundleHash", "type": "bytes32"}],
        "name": "isAnchored",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "bytes32", "name": "bundleHash", "type": "bytes32"}],
        "name": "getAnchor",
        "outputs": [
            {"internalType": "bytes32", "name": "modelHash",       "type": "bytes32"},
            {"internalType": "bytes32", "name": "kbHash",          "type": "bytes32"},
            {"internalType": "bytes32", "name": "explanationHash", "type": "bytes32"},
            {"internalType": "address", "name": "signer",          "type": "address"},
            {"internalType": "uint256", "name": "timestamp",       "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]


# ---------------------------------------------------------------------------
# Helper: compile and deploy from source (requires py-solc-x + solc binary)
# ---------------------------------------------------------------------------

def _compile_and_deploy(
    w3: Web3,
    contract_path: str,
    deployer_address: str,
    deployer_key: str,
) -> str:
    """
    Compile ProvenanceRegistry.sol and deploy to the connected chain.

    Returns the deployed contract address.
    """
    try:
        from solcx import compile_source, install_solc
    except ImportError:
        raise ImportError(
            "py-solc-x is required to compile contracts. "
            "Install with: pip install py-solc-x"
        )

    # Determine solc binary path (mirrors existing BlockchainVFLIntegrator logic)
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
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
    tx = contract.constructor().build_transaction(
        {"from": deployer_address, "nonce": nonce}
    )
    signed = w3.eth.account.sign_transaction(tx, private_key=deployer_key)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt: TxReceipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt["contractAddress"]


# ---------------------------------------------------------------------------
# ProvenanceIntegrator
# ---------------------------------------------------------------------------

class ProvenanceIntegrator:
    """
    Connects to a Web3 provider (Ganache or mock EVM) and anchors provenance
    bundles on-chain via ProvenanceRegistry.

    Args:
        rpc_url: HTTP RPC URL of Ganache (default ``http://127.0.0.1:7545``).
        contract_address: Pre-deployed contract address. If ``None``, the
            contract will be compiled and deployed from *contract_path*.
        contract_path: Path to ``ProvenanceRegistry.sol``.  Required when
            *contract_address* is ``None`` and *use_mock* is ``False``.
        private_key: Ethereum private key (hex with ``0x``) used to sign
            deployment and anchoring transactions.
        use_mock: When ``True``, an in-process EthereumTesterProvider is used
            instead of a real Ganache node. Useful for testing.
    """

    def __init__(
        self,
        rpc_url: str = "http://127.0.0.1:7545",
        contract_address: Optional[str] = None,
        contract_path: Optional[str] = None,
        private_key: Optional[str] = None,
        use_mock: bool = False,
    ) -> None:
        self.use_mock = use_mock
        self.private_key = private_key

        if use_mock:
            from eth_tester import PyEVMBackend
            from web3.providers.eth_tester import EthereumTesterProvider

            self.w3 = Web3(EthereumTesterProvider(PyEVMBackend()))
            # Use the first pre-funded account when no private key is given
            self.deployer_address = self.w3.eth.accounts[0]
            self._deploy_mock_contract()
        else:
            self.w3 = Web3(Web3.HTTPProvider(rpc_url))
            if not self.w3.is_connected():
                raise ConnectionError(
                    f"Cannot connect to Web3 provider at {rpc_url}. "
                    "Ensure Ganache is running."
                )

            if not private_key:
                raise ValueError(
                    "private_key is required when use_mock=False."
                )
            account = self.w3.eth.account.from_key(private_key)
            self.deployer_address = account.address

            if contract_address:
                self.contract_address = Web3.to_checksum_address(contract_address)
            else:
                if not contract_path:
                    # Default to sibling ProvenanceRegistry.sol
                    contract_path = os.path.join(
                        os.path.dirname(__file__), "ProvenanceRegistry.sol"
                    )
                print("Deploying ProvenanceRegistry.sol to Ganache…")
                self.contract_address = _compile_and_deploy(
                    self.w3, contract_path, self.deployer_address, private_key
                )
                print(f"  Contract deployed at: {self.contract_address}")

            self.registry = self.w3.eth.contract(
                address=self.contract_address,
                abi=PROVENANCE_REGISTRY_ABI,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deploy_mock_contract(self) -> None:
        """Set up in-process mock store (no contract compilation required)."""
        # The mock implementation stores anchors in a Python dict, bypassing
        # the need to compile and deploy the real Solidity contract. This keeps
        # tests fast and removes the solc binary dependency for mock mode.
        self._mock_store: Dict[str, Any] = {}
        self.contract_address = "0x" + "0" * 40  # placeholder
        self.registry = None  # not used in mock path

    def _hex_to_bytes32(self, hex_str: str) -> bytes:
        """Convert a 64-char hex string to a 32-byte value for Solidity bytes32."""
        hex_str = hex_str.removeprefix("0x")
        # Pad or truncate to 32 bytes
        padded = hex_str.zfill(64)[:64]
        return bytes.fromhex(padded)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anchor_provenance(
        self,
        bundle_hash: str,
        model_hash: str,
        kb_hash: str,
        explanation_hash: str,
        signer_address: str,
    ) -> str:
        """
        Submit an ``anchorProvenance`` transaction to the registry contract.

        Args:
            bundle_hash: 64-char hex SHA-256 of the provenance bundle.
            model_hash: 64-char hex of model version hash.
            kb_hash: 64-char hex of knowledge base hash.
            explanation_hash: 64-char hex of explanation hash.
            signer_address: Ethereum address of the MetaMask signer.

        Returns:
            Transaction hash hex string (``0x…``).
        """
        if self.use_mock:
            return self._anchor_mock(
                bundle_hash, model_hash, kb_hash, explanation_hash, signer_address
            )

        bh = self._hex_to_bytes32(bundle_hash)
        mh = self._hex_to_bytes32(model_hash)
        kh = self._hex_to_bytes32(kb_hash)
        eh = self._hex_to_bytes32(explanation_hash)
        signer = Web3.to_checksum_address(signer_address)

        nonce = self.w3.eth.get_transaction_count(self.deployer_address)
        tx = self.registry.functions.anchorProvenance(
            bh, mh, kh, eh, signer
        ).build_transaction({
            "from": self.deployer_address,
            "nonce": nonce,
            "gas": 200_000,
        })
        signed = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash.hex() if isinstance(tx_hash, (bytes, bytearray)) else str(tx_hash)

    def _anchor_mock(
        self,
        bundle_hash: str,
        model_hash: str,
        kb_hash: str,
        explanation_hash: str,
        signer_address: str,
    ) -> str:
        """Store anchor in the in-memory mock store and return a fake tx hash."""
        # Deterministic fake tx hash derived from bundle_hash
        import hashlib
        import time as _time

        self._mock_store[bundle_hash] = {
            "model_hash": model_hash,
            "kb_hash": kb_hash,
            "explanation_hash": explanation_hash,
            "signer": signer_address,
            "timestamp": _time.time(),
        }
        # Deterministic fake tx hash derived from bundle_hash
        fake_tx = "0x" + hashlib.sha256(
            (bundle_hash + "mock_tx").encode()
        ).hexdigest()
        return fake_tx

    def is_anchored(self, bundle_hash: str) -> bool:
        """
        Check whether a bundle hash has been anchored on-chain.

        Args:
            bundle_hash: 64-char hex SHA-256 of the bundle.

        Returns:
            ``True`` if the bundle is anchored.
        """
        if self.use_mock:
            return bundle_hash in self._mock_store

        bh = self._hex_to_bytes32(bundle_hash)
        return self.registry.functions.isAnchored(bh).call()

    def get_anchor(self, bundle_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve anchor details from the registry.

        Args:
            bundle_hash: 64-char hex SHA-256 of the bundle.

        Returns:
            Dict with model_hash, kb_hash, explanation_hash, signer, timestamp;
            or ``None`` if not anchored.
        """
        if self.use_mock:
            return self._mock_store.get(bundle_hash)

        bh = self._hex_to_bytes32(bundle_hash)
        try:
            model_hash, kb_hash, explanation_hash, signer, timestamp = (
                self.registry.functions.getAnchor(bh).call()
            )
            return {
                "model_hash": model_hash.hex(),
                "kb_hash": kb_hash.hex(),
                "explanation_hash": explanation_hash.hex(),
                "signer": signer,
                "timestamp": timestamp,
            }
        except Exception:
            return None
