"""
Blockchain + VFL Integration
=============================
Connects the VFL training loop to a real Ethereum-compatible node
(default: local Hardhat node at http://127.0.0.1:8545, chainId 31337).

Configuration via environment variables
----------------------------------------
WEB3_PROVIDER_URI          RPC endpoint (default: http://127.0.0.1:8545)
CHAIN_ID                   EVM chain ID  (default: 31337)
BLOCKCHAIN_PRIVATE_KEY     Private key for client #0  (overrides Hardhat default)
BLOCKCHAIN_PRIVATE_KEY_<N> Private key for client N   (0-indexed, highest priority)
AGGREGATOR_CONTRACT_ADDRESS
    If set, attach to this already-deployed contract instead of deploying a new one.

Quick-start (Windows PowerShell)
---------------------------------
  # Terminal 1 – start local Hardhat node
  cd blockchain
  npm install
  npx hardhat node

  # Terminal 2 – run training with blockchain
  cd src
  python demo_rag_vfl_with_zip.py --datapath .\\data --withblockchain --use-rag
"""

from eth_account import Account
from web3 import Web3
import sys
import os
import json

# ---------------------------------------------------------------------------
# Hardhat default private keys
# (from mnemonic: test test test test test test test test test test test junk)
# These are the well-known deterministic keys produced by `npx hardhat node`.
# ---------------------------------------------------------------------------
_HARDHAT_DEFAULT_KEYS = [
    "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",  # account #0
    "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",  # account #1
    "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a",  # account #2
    "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6",  # account #3
]


class BlockchainVFLIntegrator:
    """
    Integrates the FederatedAggregator smart contract with the VFL training loop.

    Each instance:
      1. Connects to a local Ethereum node (Hardhat by default).
      2. Deploys (or attaches to) the FederatedAggregator contract.
      3. Registers ``num_clients`` participant accounts.
      4. Provides ``update_client_weights`` / ``aggregate_weights`` that send
         *real* on-chain transactions and return ``(tx_hash, block_number)``.
    """

    def __init__(self, num_clients, contract_path, erc20_path=None):
        # ------------------------------------------------------------------ #
        # 1. Read configuration                                                #
        # ------------------------------------------------------------------ #
        self.provider_uri = os.getenv("WEB3_PROVIDER_URI", "http://127.0.0.1:8545")
        self.chain_id = int(os.getenv("CHAIN_ID", "31337"))

        # Build client accounts: per-index env var > shared env var > Hardhat default
        self.client_accounts = []
        for i in range(num_clients):
            key = (
                os.getenv(f"BLOCKCHAIN_PRIVATE_KEY_{i}")
                or (os.getenv("BLOCKCHAIN_PRIVATE_KEY") if i == 0 else None)
                or _HARDHAT_DEFAULT_KEYS[i % len(_HARDHAT_DEFAULT_KEYS)]
            )
            self.client_accounts.append(Account.from_key(key))

        # ------------------------------------------------------------------ #
        # 2. Connect to the Ethereum node                                      #
        # ------------------------------------------------------------------ #
        self.w3 = Web3(Web3.HTTPProvider(self.provider_uri))
        if not self.w3.is_connected():
            raise ConnectionError(
                f"\n[Blockchain] Cannot connect to Ethereum node at {self.provider_uri}\n"
                "  To start a local Hardhat node:\n"
                "    cd blockchain && npm install && npx hardhat node\n"
                "  Or set WEB3_PROVIDER_URI to point to your running node.\n"
                "  For Windows PowerShell:\n"
                "    $env:WEB3_PROVIDER_URI=\"http://127.0.0.1:8545\""
            )

        # Owner is always the first client account (deploys + calls aggregate)
        self.owner_account = self.client_accounts[0]

        # Storage for the most recent on-chain tx (read by demo script per epoch)
        self.last_tx_hash = None
        self.last_block_number = None

        # ------------------------------------------------------------------ #
        # 3. Load ABI + bytecode                                               #
        # ------------------------------------------------------------------ #
        abi, bytecode = self._load_contract(contract_path)

        # ------------------------------------------------------------------ #
        # 4. Deploy or attach                                                  #
        # ------------------------------------------------------------------ #
        contract_address_env = os.getenv("AGGREGATOR_CONTRACT_ADDRESS", "").strip()
        if contract_address_env:
            self.contract_address = Web3.to_checksum_address(contract_address_env)
            print(f"  Attaching to existing contract at {self.contract_address}")
            should_register = False
        else:
            self.contract_address = self._deploy_contract(abi, bytecode)
            print(
                f"  Contract deployed at {self.contract_address}\n"
                f"  (block {self.w3.eth.block_number})"
            )
            should_register = True

        self.aggregator = self.w3.eth.contract(
            address=self.contract_address, abi=abi
        )

        if should_register:
            self.add_clients_to_contract()

    # ---------------------------------------------------------------------- #
    # Contract loading helpers                                                 #
    # ---------------------------------------------------------------------- #

    def _load_contract(self, contract_path):
        """
        Return ``(abi, bytecode)``.

        Looks for a pre-compiled Hardhat artifact first
        (``blockchain/artifacts/contracts/Aggregator.sol/FederatedAggregator.json``).
        Falls back to compiling ``src/Aggregator.sol`` with the bundled solc binary.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        artifact_path = os.path.join(
            repo_root,
            "blockchain", "artifacts", "contracts",
            "Aggregator.sol", "FederatedAggregator.json",
        )
        if os.path.exists(artifact_path):
            with open(artifact_path) as f:
                artifact = json.load(f)
            print(f"  Using Hardhat artifact: {os.path.relpath(artifact_path, repo_root)}")
            return artifact["abi"], artifact["bytecode"]

        # Fall back: compile with platform-bundled solc binary
        return self._compile_with_solcx(contract_path)

    def _compile_with_solcx(self, contract_path):
        """Compile *contract_path* with the bundled solc binary; return ``(abi, bytecode)``."""
        from solcx import compile_source

        with open(contract_path) as f:
            source = f.read()

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if sys.platform == "win32":
            solc_binary = os.path.join(repo_root, "tests", "solc-0.8.23-win32", "solc.exe")
        elif sys.platform == "darwin":
            solc_binary = os.path.join(repo_root, "tests", "solc-0.8.23-macos", "solc-macos")
        else:
            solc_binary = os.path.join(repo_root, "tests", "solc-0.8.23-linux", "solc-static-linux")

        compiled = compile_source(
            source,
            output_values=["abi", "bin"],
            solc_binary=solc_binary,
        )
        _, interface = compiled.popitem()
        print("  Compiled contract with py-solc-x")
        return interface["abi"], interface["bin"]

    # ---------------------------------------------------------------------- #
    # Deployment                                                               #
    # ---------------------------------------------------------------------- #

    def _deploy_contract(self, abi, bytecode):
        """Deploy FederatedAggregator; return the checksummed contract address."""
        nonce = self.w3.eth.get_transaction_count(self.owner_account.address)
        deploy_tx = (
            self.w3.eth.contract(abi=abi, bytecode=bytecode)
            .constructor()
            .build_transaction({
                "from": self.owner_account.address,
                "nonce": nonce,
                "chainId": self.chain_id,
            })
        )
        signed = self.w3.eth.account.sign_transaction(deploy_tx, self.owner_account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.contractAddress

    # ---------------------------------------------------------------------- #
    # Participant management                                                   #
    # ---------------------------------------------------------------------- #

    def add_clients_to_contract(self):
        """Register each client as a participant in the on-chain contract."""
        for client_account in self.client_accounts:
            try:
                nonce = self.w3.eth.get_transaction_count(client_account.address)
                tx = self.aggregator.functions.addParticipant().build_transaction({
                    "from": client_account.address,
                    "nonce": nonce,
                    "chainId": self.chain_id,
                })
                signed = self.w3.eth.account.sign_transaction(tx, client_account.key)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                self.w3.eth.wait_for_transaction_receipt(tx_hash)
            except Exception:
                # Client already registered (e.g. attaching to existing contract) – skip
                pass

    # ---------------------------------------------------------------------- #
    # Weight update / aggregation                                              #
    # ---------------------------------------------------------------------- #

    def update_client_weights(self, client_account, weights):
        """
        Submit quantised embeddings from *client_account* to the smart contract.

        Parameters
        ----------
        client_account : eth_account.Account
            One of ``self.client_accounts``.
        weights : list[list[int]]
            2-D list of integer-quantised embedding values.

        Returns
        -------
        tuple[str, int]
            ``(tx_hash_hex, block_number)`` of the confirmed transaction.
        """
        nonce = self.w3.eth.get_transaction_count(client_account.address)
        tx = self.aggregator.functions.updateParticipantParameters(weights).build_transaction({
            "from": client_account.address,
            "nonce": nonce,
            "chainId": self.chain_id,
        })
        signed = self.w3.eth.account.sign_transaction(tx, client_account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.transactionHash.hex(), receipt.blockNumber

    def aggregate_weights(self):
        """
        Trigger on-chain aggregation of all client parameters (owner-only call).

        Also stores the result in ``self.last_tx_hash`` / ``self.last_block_number``
        so the demo script can read them after each training epoch.

        Returns
        -------
        tuple[str, int]
            ``(tx_hash_hex, block_number)`` of the confirmed transaction.
        """
        nonce = self.w3.eth.get_transaction_count(self.owner_account.address)
        tx = self.aggregator.functions.aggregate().build_transaction({
            "from": self.owner_account.address,
            "nonce": nonce,
            "chainId": self.chain_id,
        })
        signed = self.w3.eth.account.sign_transaction(tx, self.owner_account.key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        self.last_tx_hash = receipt.transactionHash.hex()
        self.last_block_number = receipt.blockNumber
        return self.last_tx_hash, self.last_block_number

    def get_aggregated_weights(self):
        """Return the aggregated parameters currently stored in the contract."""
        return self.aggregator.functions.getAggregatedWeights().call()


# ---------------------------------------------------------------------------
# Quick smoke-test (requires a running Hardhat node)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CONTRACT_SOURCE = os.path.join(repo_root, "src", "Aggregator.sol")

    print("Connecting to Hardhat node…")
    integrator = BlockchainVFLIntegrator(4, CONTRACT_SOURCE)
    print(f"Provider URI : {integrator.provider_uri}")
    print(f"Chain ID     : {integrator.chain_id}")
    print(f"Contract     : {integrator.contract_address}")

    # Build dummy 10 x 64 parameter matrices for 4 clients
    client_parameters = [
        [[i + j + k for k in range(64)] for j in range(10)]
        for i in range(4)
    ]

    for i, account in enumerate(integrator.client_accounts):
        tx_hash, block_num = integrator.update_client_weights(account, client_parameters[i])
        print(f"Client {i} update  tx: {tx_hash}  block: {block_num}")

    tx_hash, block_num = integrator.aggregate_weights()
    print(f"Aggregation       tx: {tx_hash}  block: {block_num}")
    print("Aggregated weights (first row):", integrator.get_aggregated_weights()[0])
