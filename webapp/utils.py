"""
Utility functions for MedRAG web application.
"""

import sys
import os
import hashlib
import json
import streamlit as st
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_registry import ModelRegistry
from ledger import Ledger
from inference import load_inference_model


@st.cache_resource
def get_model_registry():
    """Get or create model registry instance."""
    repo_root = Path(__file__).parent.parent
    registry_dir = repo_root / 'models' / 'registry'
    return ModelRegistry(str(registry_dir))


@st.cache_resource
def get_ledger():
    """Get or create ledger instance."""
    repo_root = Path(__file__).parent.parent
    ledger_dir = repo_root / 'ledger'
    return Ledger(str(ledger_dir))


@st.cache_resource
def get_inference_engine(_version_id=None):
    """
    Get inference engine, optionally loading a specific model version.
    
    Args:
        _version_id: Optional version ID to load specific model
    
    Returns:
        MedRAGInference instance
    """
    try:
        checkpoint_path = None
        if _version_id:
            registry = get_model_registry()
            version = registry.get_version(_version_id)
            if version:
                checkpoint_path = version.checkpoint_path

        # Allow the dataset directory to be supplied via env var so that
        # load_inference_model can auto-detect class names when the
        # checkpoint does not contain class metadata.
        repo_root = Path(__file__).parent.parent
        dataset_dir = os.environ.get(
            'MEDRAG_DATASET_DIR',
            str(repo_root / 'data')
        )
        # Only pass dataset_dir if the directory actually exists
        if not os.path.isdir(dataset_dir):
            dataset_dir = None

        inference = load_inference_model(
            checkpoint_path=checkpoint_path,
            use_rag=True,
            num_clients=4,
            dataset_dir=dataset_dir,
        )
        return inference
    except Exception as e:
        st.error(f"Error loading inference engine: {e}")
        return None


def format_timestamp(timestamp_str):
    """Format ISO timestamp for display."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return timestamp_str


def format_hash(hash_str, length=16):
    """Format hash for display (truncated with ellipsis)."""
    if len(hash_str) > length:
        return hash_str[:length] + "..."
    return hash_str


def get_metric_color(metric_name, value):
    """Get color for metric based on value."""
    if 'accuracy' in metric_name.lower():
        if value >= 0.9:
            return 'green'
        elif value >= 0.7:
            return 'orange'
        else:
            return 'red'
    elif 'loss' in metric_name.lower():
        if value <= 0.3:
            return 'green'
        elif value <= 0.6:
            return 'orange'
        else:
            return 'red'
    return 'blue'


def create_metric_card(title, value, delta=None):
    """Create a metric display card."""
    st.metric(label=title, value=value, delta=delta)


def check_system_status():
    """Check if system components are initialized."""
    registry = get_model_registry()
    ledger = get_ledger()
    
    registry_summary = registry.get_summary()
    ledger_summary = ledger.get_summary()
    
    status = {
        'registry_ok': registry_summary['total_versions'] >= 0,
        'ledger_ok': ledger_summary['training_integrity'] and ledger_summary['access_integrity'],
        'models_available': registry_summary['total_versions'] > 0,
        'training_data_available': ledger_summary['training_entries'] > 0
    }
    
    return status, registry_summary, ledger_summary


def display_system_status():
    """Display system status in sidebar."""
    status, registry_summary, ledger_summary = check_system_status()
    
    st.sidebar.markdown("### 🔧 System Status")
    
    status_icon = "✅" if all(status.values()) else "⚠️"
    st.sidebar.markdown(f"{status_icon} **System Status**")
    
    st.sidebar.markdown(f"""
    - Registry: {'✅' if status['registry_ok'] else '❌'}
    - Ledger: {'✅' if status['ledger_ok'] else '❌'}
    - Models: {registry_summary['total_versions']} versions
    - Training Logs: {ledger_summary['training_entries']} entries
    """)


def create_download_button(data, filename, button_text="Download"):
    """Create a download button for data."""
    st.download_button(
        label=button_text,
        data=data,
        file_name=filename,
        mime='application/octet-stream'
    )


def display_error_message(message):
    """Display a formatted error message."""
    st.error(f"❌ {message}")


def display_success_message(message):
    """Display a formatted success message."""
    st.success(f"✅ {message}")


def display_info_message(message):
    """Display a formatted info message."""
    st.info(f"ℹ️ {message}")


def display_warning_message(message):
    """Display a formatted warning message."""
    st.warning(f"⚠️ {message}")


# ============================================================================
# Ganache Integration Utilities
# ============================================================================

@st.cache_data(ttl=30)  # Cache for 30 seconds
def check_ganache_connection():
    """Check if Ganache is running."""
    try:
        from web3 import Web3
        ganache_url = os.getenv('GANACHE_URL', 'http://127.0.0.1:7545')
        w3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 2}))
        return w3.is_connected()
    except:
        return False


@st.cache_data(ttl=30)
def get_ganache_blocks(num_blocks=5):
    """Fetch recent blocks from Ganache."""
    try:
        from web3 import Web3
        ganache_url = os.getenv('GANACHE_URL', 'http://127.0.0.1:7545')
        w3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 2}))
        
        if not w3.is_connected():
            return None
        
        latest_block = w3.eth.block_number
        blocks = []
        for i in range(num_blocks):
            block_num = latest_block - i
            if block_num >= 0:
                block = w3.eth.get_block(block_num)
                blocks.append({
                    'number': block_num,
                    'hash': block['hash'].hex(),
                    'timestamp': block['timestamp'],
                    'transactions': len(block['transactions'])
                })
        return blocks
    except:
        return None


def check_model_governance_approval(version_id: str) -> dict:
    """
    Check whether a model version is APPROVED via the on-chain governance
    contract (ModelGovernanceRegistry).

    Falls back to mock mode if Ganache is not reachable or env vars are not
    configured, so the UI always gets a response.

    Returns:
        Dict with keys: ``is_approved`` (bool), ``status`` (str),
        ``approval_count`` (int), ``required_approvals`` (int),
        ``mock_mode`` (bool).
    """
    ganache_url = os.getenv("GANACHE_URL", "http://127.0.0.1:7545")
    admin_key = os.getenv("GOVERNANCE_ADMIN_KEY", "")
    contract_address = os.getenv("GOVERNANCE_CONTRACT_ADDRESS", "")

    # Compute deterministic model version hash (same formula as integrator)
    def _version_hash(vid: str) -> str:
        payload = {"version_id": vid, "model_hash": ""}
        content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()

    model_hash = _version_hash(version_id)

    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
        from model_governance_integrator import ModelGovernanceIntegrator

        if admin_key and contract_address:
            integrator = ModelGovernanceIntegrator(
                rpc_url=ganache_url,
                contract_address=contract_address,
                admin_key=admin_key,
            )
            info = integrator.get_approval_status(model_hash)
            info["mock_mode"] = False
            return info
        # No Ganache config available → mock mode
        integrator = ModelGovernanceIntegrator(use_mock=True)
        info = integrator.get_approval_status(model_hash)
        info["mock_mode"] = True
        return info
    except Exception:
        # If the integrator is unavailable, fail open in mock mode
        return {
            "is_approved": False,
            "status": "UNKNOWN",
            "approval_count": 0,
            "required_approvals": 3,
            "mock_mode": True,
        }


def get_transaction_details(tx_hash):
    """Get transaction details from Ganache."""
    try:
        from web3 import Web3
        ganache_url = os.getenv('GANACHE_URL', 'http://127.0.0.1:7545')
        w3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 2}))
        
        if not w3.is_connected():
            return None
        
        tx = w3.eth.get_transaction(tx_hash)
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        
        return {
            'hash': tx['hash'].hex(),
            'from': tx['from'],
            'to': tx['to'],
            'block_number': tx['blockNumber'],
            'gas_used': receipt['gasUsed'],
            'status': 'Success' if receipt['status'] == 1 else 'Failed'
        }
    except:
        return None
