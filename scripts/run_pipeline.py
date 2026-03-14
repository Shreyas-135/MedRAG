#!/usr/bin/env python3
"""
End-to-End MedRAG Pipeline

Runs the full federated learning pipeline starting from a ZIP file:
  1. Extract ZIP dataset and distribute across 4 hospital clients
  2. Train VFL model with RAG enhancement
  3. Update model registry with versioned checkpoints
  4. Verify training rounds on blockchain (optional)

Usage:
    # Full pipeline with blockchain verification
    python scripts/run_pipeline.py --zip-file /path/to/xray_dataset.zip --use-rag --withblockchain

    # Training only (no blockchain)
    python scripts/run_pipeline.py --zip-file /path/to/xray_dataset.zip --use-rag

    # Use already-extracted dataset (skip ZIP step)
    python scripts/run_pipeline.py --datapath ./data --use-rag --withblockchain

    # Quick smoke-test with 1 epoch
    python scripts/run_pipeline.py --zip-file /path/to/xray_dataset.zip --num-epochs 1
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Resolve project root so imports work regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def step_extract_zip(zip_file: str, output_dir: str, num_hospitals: int,
                     train_split: float, binary: bool, seed: int) -> bool:
    """
    Step 1: Extract ZIP dataset and distribute across hospital clients.

    Returns True on success.
    """
    print("\n" + "=" * 80)
    print("STEP 1: Extracting ZIP dataset and distributing across hospitals")
    print("=" * 80)

    loader_script = SRC_DIR / "load_zip_dataset.py"
    cmd = [
        sys.executable, str(loader_script),
        "--zip-file", zip_file,
        "--output-dir", output_dir,
        "--num-hospitals", str(num_hospitals),
        "--train-split", str(train_split),
        "--seed", str(seed),
    ]
    if binary:
        cmd.append("--binary")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ ZIP extraction failed (exit code {result.returncode})")
        return False

    print("\n✅ ZIP extraction complete")
    return True


def step_train(datapath: str, num_epochs: int, theta: float, datasize: float,
               use_rag: bool, withblockchain: bool, model_type: str) -> bool:
    """
    Step 2: Run VFL training, update model registry, and optionally verify on
    blockchain.

    Returns True on success.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Training VFL model + updating model registry")
    if withblockchain:
        print("        + verifying aggregation on blockchain")
    print("=" * 80)

    train_script = SRC_DIR / "demo_rag_vfl_with_zip.py"
    cmd = [
        sys.executable, str(train_script),
        "--datapath", datapath,
        "--num-epochs", str(num_epochs),
        "--theta", str(theta),
        "--datasize", str(datasize),
        "--model-type", model_type,
    ]
    if use_rag:
        cmd.append("--use-rag")
    if withblockchain:
        cmd.append("--withblockchain")

    result = subprocess.run(cmd, cwd=str(SRC_DIR), check=False)
    if result.returncode != 0:
        print(f"\n❌ Training failed (exit code {result.returncode})")
        return False

    print("\n✅ Training complete – model registry updated")
    return True


def step_verify_registry(datapath: str) -> bool:
    """
    Step 3: Print a summary of the model registry to confirm updates were saved.

    Returns True if the registry contains at least one version.
    """
    print("\n" + "=" * 80)
    print("STEP 3: Verifying model registry")
    print("=" * 80)

    try:
        from model_registry import ModelRegistry

        registry = ModelRegistry()
        summary = registry.get_summary()

        print(f"  Registry location : {registry.registry_dir}")
        print(f"  Total versions    : {summary['total_versions']}")
        print(f"  Latest version    : {summary.get('latest_version', 'N/A')}")
        if summary.get("best_accuracy") is not None:
            print(f"  Best accuracy     : {summary['best_accuracy']:.4f}")
        print(f"  Storage           : {summary['storage_size_mb']:.2f} MB")

        if summary["total_versions"] == 0:
            print("\n⚠️  Registry is empty – training may not have completed")
            return False

        print("\n✅ Model registry verified")
        return True
    except Exception as exc:
        print(f"\n⚠️  Could not read model registry: {exc}")
        return False


def step_verify_ledger() -> bool:
    """
    Step 4: Verify the training ledger integrity (blockchain-style hash chain).

    Returns True if the ledger is intact.
    """
    print("\n" + "=" * 80)
    print("STEP 4: Verifying training ledger integrity")
    print("=" * 80)

    try:
        from ledger import Ledger

        ledger = Ledger()
        summary = ledger.get_summary()

        print(f"  Ledger location       : {ledger.ledger_dir}")
        print(f"  Training entries      : {summary['training_entries']}")
        print(f"  Access entries        : {summary['access_entries']}")
        training_ok = summary["training_integrity"]
        access_ok = summary["access_integrity"]
        print(f"  Training integrity    : {'✅ Valid' if training_ok else '❌ Invalid'}")
        print(f"  Access integrity      : {'✅ Valid' if access_ok else '❌ Invalid'}")

        if not (training_ok and access_ok):
            print("\n❌ Ledger integrity check failed")
            return False

        print("\n✅ Ledger integrity verified")
        return True
    except Exception as exc:
        print(f"\n⚠️  Could not read ledger: {exc}")
        return False


def step_register_governance() -> bool:
    """
    Step 5 (optional): Register the latest model version with the on-chain
    ModelGovernanceRegistry and submit approvals from configured hospital keys.

    Environment variables (all optional – skips if missing):
        GANACHE_URL                   Ganache HTTP RPC URL (default: http://127.0.0.1:7545)
        GOVERNANCE_ADMIN_KEY          Admin private key (0x…) for deploying / registering
        GOVERNANCE_CONTRACT_ADDRESS   Pre-deployed contract address (skip deploy if set)
        GOVERNANCE_HOSPITAL_KEYS      Comma-separated hospital private keys (0x…,0x…,…)

    When GOVERNANCE_ADMIN_KEY is absent the step runs in mock mode.

    Returns True on success.
    """
    print("\n" + "=" * 80)
    print("STEP 5: Model Governance – register + submit approvals")
    print("=" * 80)

    import os
    from model_governance_integrator import ModelGovernanceIntegrator

    ganache_url = os.getenv("GANACHE_URL", "http://127.0.0.1:7545")
    admin_key = os.getenv("GOVERNANCE_ADMIN_KEY", "")
    contract_address = os.getenv("GOVERNANCE_CONTRACT_ADDRESS", "")
    hospital_keys_env = os.getenv("GOVERNANCE_HOSPITAL_KEYS", "")
    hospital_keys = [k.strip() for k in hospital_keys_env.split(",") if k.strip()]

    use_mock = not bool(admin_key)
    if use_mock:
        print("  ℹ️  GOVERNANCE_ADMIN_KEY not set – running in mock mode")

    try:
        from model_registry import ModelRegistry
        registry = ModelRegistry()
        summary = registry.get_summary()
        latest_vid = summary.get("latest_version")
        if not latest_vid:
            print("  ⚠️  No model version found in registry – skipping governance step")
            return True

        version = registry.get_version(latest_vid)
        model_hash = ModelGovernanceIntegrator.compute_model_version_hash(
            version_id=latest_vid,
            model_hash=version.model_hash if version else "",
        )
        print(f"  Model version : {latest_vid}")
        print(f"  Model hash    : {model_hash[:16]}…")

        kwargs: dict = {"use_mock": use_mock}
        if not use_mock:
            kwargs["rpc_url"] = ganache_url
            kwargs["admin_key"] = admin_key
            if contract_address:
                kwargs["contract_address"] = contract_address

        integrator = ModelGovernanceIntegrator(**kwargs)
        tx = integrator.register_model(model_hash)
        print(f"  ✅ Model registered (PENDING) – TX: {tx}")

        for i, hkey in enumerate(hospital_keys):
            try:
                tx_a = integrator.approve_model(model_hash, hospital_key=hkey)
                print(f"  ✅ Hospital {i+1} approval submitted – TX: {tx_a}")
            except Exception as ae:
                print(f"  ⚠️  Hospital {i+1} approval failed: {ae}")

        status = integrator.get_approval_status(model_hash)
        print(
            f"  Status: {status['status']}  "
            f"({status['approval_count']}/{status['required_approvals']} approvals)"
        )
        print("\n✅ Governance step complete")
        return True
    except Exception as exc:
        print(f"\n⚠️  Governance step failed: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end MedRAG pipeline: ZIP → train → registry → blockchain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input: either a ZIP file or an already-extracted dataset directory
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--zip-file",
        type=str,
        help="Path to ZIP file containing X-ray images (runs extraction first)",
    )
    input_group.add_argument(
        "--datapath",
        type=str,
        help="Path to already-extracted dataset directory (skips ZIP step)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Output directory for extracted dataset (default: <project>/data)",
    )
    parser.add_argument(
        "--num-hospitals",
        type=int,
        default=4,
        help="Number of hospital clients (default: 4)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Binary classification: covid vs normal only",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.1,
        help="Differential privacy noise parameter 0-0.25 (default: 0.1)",
    )
    parser.add_argument(
        "--datasize",
        type=float,
        default=1.0,
        choices=[0.0125, 0.25, 0.5, 1.0],
        # 0.0125 ≈ 1/80 for quick smoke-tests; 0.25/0.5/1.0 for full runs
        help="Fraction of dataset to use (default: 1.0)",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable RAG knowledge enhancement (recommended)",
    )
    parser.add_argument(
        "--withblockchain",
        action="store_true",
        help="Enable blockchain-verified weight aggregation",
    )
    parser.add_argument(
        "--with-governance",
        action="store_true",
        help=(
            "Register the trained model version on-chain with ModelGovernanceRegistry "
            "and submit approvals from GOVERNANCE_HOSPITAL_KEYS env var. "
            "Requires GOVERNANCE_ADMIN_KEY env var (falls back to mock mode)."
        ),
    )
    parser.add_argument(
        type=str,
        default="resnet_vgg",
        choices=["resnet_vgg", "vit", "vit_small", "hybrid", "yolo5", "yolo8", "resnet_yolo"],
        help="Client model architecture (default: resnet_vgg)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    if args.zip_file is None and args.datapath is None:
        parser.error("Provide either --zip-file (to extract a new dataset) "
                 "or --datapath (to use an already-extracted dataset directory)")

    print("=" * 80)
    print("MedRAG End-to-End Pipeline")
    print("ZIP → Dataset Distribution → VFL Training → Model Registry → Blockchain")
    print("=" * 80)
    print(f"  Hospitals      : {args.num_hospitals}")
    print(f"  Epochs         : {args.num_epochs}")
    print(f"  DP noise θ     : {args.theta}")
    print(f"  RAG            : {'enabled' if args.use_rag else 'disabled'}")
    print(f"  Blockchain     : {'enabled' if args.withblockchain else 'disabled'}")
    print(f"  Governance     : {'enabled' if args.with_governance else 'disabled'}")
    print(f"  Model type     : {args.model_type}")
    print("=" * 80)

    results = {}

    # ------------------------------------------------------------------
    # Step 1: Extract ZIP (if a zip file was provided)
    # ------------------------------------------------------------------
    if args.zip_file:
        ok = step_extract_zip(
            zip_file=args.zip_file,
            output_dir=args.output_dir,
            num_hospitals=args.num_hospitals,
            train_split=args.train_split,
            binary=args.binary,
            seed=args.seed,
        )
        results["extract_zip"] = ok
        if not ok:
            _print_summary(results)
            return 1
        datapath = args.output_dir
    else:
        datapath = args.datapath
        results["extract_zip"] = "skipped"

    # ------------------------------------------------------------------
    # Step 2: Train VFL model (updates registry + optional blockchain)
    # ------------------------------------------------------------------
    ok = step_train(
        datapath=datapath,
        num_epochs=args.num_epochs,
        theta=args.theta,
        datasize=args.datasize,
        use_rag=args.use_rag,
        withblockchain=args.withblockchain,
        model_type=args.model_type,
    )
    results["train"] = ok
    if not ok:
        _print_summary(results)
        return 1

    # ------------------------------------------------------------------
    # Step 3: Verify model registry
    # ------------------------------------------------------------------
    results["registry"] = step_verify_registry(datapath)

    # ------------------------------------------------------------------
    # Step 4: Verify ledger integrity
    # ------------------------------------------------------------------
    results["ledger"] = step_verify_ledger()

    # ------------------------------------------------------------------
    # Step 5 (optional): Register latest model version with on-chain
    # governance contract and submit all configured hospital approvals.
    # ------------------------------------------------------------------
    if args.with_governance:
        results["governance"] = step_register_governance()

    _print_summary(results)
    overall_ok = all(v is True or v == "skipped" for v in results.values())
    return 0 if overall_ok else 1


def _print_summary(results: dict):
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    labels = {
        "extract_zip": "ZIP Extraction",
        "train":       "VFL Training + Registry Update",
        "registry":    "Model Registry Verification",
        "ledger":      "Ledger Integrity Verification",
        "governance":  "On-Chain Governance Registration",
    }
    for key, label in labels.items():
        if key not in results:
            continue
        val = results[key]
        if val == "skipped":
            icon = "⏭️ "
            status = "skipped"
        elif val:
            icon = "✅"
            status = "passed"
        else:
            icon = "❌"
            status = "FAILED"
        print(f"  {icon}  {label:<40} {status}")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())
