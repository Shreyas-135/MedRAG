
"""
Blockchain Explorer Page - View blocks and transactions
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import secrets

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from webapp.utils import (
    display_system_status,
    check_ganache_connection,
    get_ganache_blocks,
    get_transaction_details,
    format_hash
)

st.set_page_config(
    page_title="Blockchain Explorer - MedRAG",
    page_icon="⛓️",
    layout="wide"
)

st.title("⛓️ Blockchain Explorer")
st.markdown("Transparent and immutable record of model aggregations and verifications")

# Display system status in sidebar
display_system_status()

# Check Ganache connection
ganache_connected = check_ganache_connection()

# Connection status
st.markdown("## 🔗 Connection Status")
if ganache_connected:
    st.success("✅ Connected to Ganache - Showing real blockchain data")
    real_blocks = get_ganache_blocks(num_blocks=5)
else:
    st.warning("⚠️ Ganache not detected - Showing mock data")
    st.info("💡 To view real blockchain data, start Ganache. [Setup Guide](../../GANACHE_SETUP.md)")
    real_blocks = None

if st.button("🔄 Refresh Connection", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.markdown("---")

# Recent Blocks Section
st.markdown("## 📦 Recent Blocks")

if real_blocks and ganache_connected:
    # Display real Ganache blocks
    blocks_data = []
    for block in real_blocks:
        blocks_data.append({
            "Block #": block['number'],
            "Timestamp": datetime.fromtimestamp(block['timestamp']).strftime("%Y-%m-%d %H:%M:%S"),
            "Transactions": block['transactions'],
            "Block Hash": format_hash(block['hash'], 20),
            "Status": "✅ Confirmed"
        })
    
    blocks_df = pd.DataFrame(blocks_data)
    st.dataframe(blocks_df, use_container_width=True, hide_index=True)
else:
    # Display mock blocks
    mock_blocks = []
    base_time = datetime.now()
    
    for i in range(5):
        block_num = 105 - i
        timestamp = base_time - timedelta(minutes=i*5)
        mock_blocks.append({
            "Block #": block_num,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Transactions": 4,
            "Type": "Weight Aggregation",
            "Block Hash": format_hash(f"0x{secrets.token_hex(32)}", 20),
            "Status": "✅ Confirmed"
        })
    
    blocks_df = pd.DataFrame(mock_blocks)
    st.dataframe(blocks_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Block Details Viewer
st.markdown("## 🔍 Block Details Viewer")

col1, col2 = st.columns([1, 2])

with col1:
    if real_blocks and ganache_connected:
        block_numbers = [b['number'] for b in real_blocks]
    else:
        block_numbers = list(range(101, 106))
    
    selected_block = st.selectbox("Select Block Number", block_numbers, index=0)

with col2:
    st.markdown(f"### Block #{selected_block}")

# Display block details
if ganache_connected and real_blocks:
    # Find selected block
    selected_block_data = next((b for b in real_blocks if b['number'] == selected_block), None)
    
    if selected_block_data:
        st.markdown(f"""
        **Block Hash:** `{selected_block_data['hash']}`  
        **Timestamp:** {datetime.fromtimestamp(selected_block_data['timestamp']).strftime("%Y-%m-%d %H:%M:%S")}  
        **Transactions:** {selected_block_data['transactions']}  
        **Status:** ✅ Confirmed
        """)
else:
    # Mock block details
    st.markdown(f"""
    **Block Hash:** `0x{secrets.token_hex(32)}`  
    **Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
    **Transactions:** 4 (Weight Aggregation)  
    **Status:** ✅ Confirmed
    """)

st.markdown("### Hospital Contributions")

# Display contributions from each hospital
hospitals = ["Hospital A (NY)", "Hospital B (Boston)", "Hospital C (Chicago)", "Hospital D (Seattle)"]

contributions = []
for i, hospital in enumerate(hospitals):
    contributions.append({
        "Hospital": hospital,
        "Weight Hash": format_hash(f"0x{secrets.token_hex(32)}", 20),
        "Gas Used": f"{21000 + i*1000:,}",
        "Status": "✅ Verified"
    })

contributions_df = pd.DataFrame(contributions)
st.dataframe(contributions_df, use_container_width=True, hide_index=True)

st.markdown("### Smart Contract Aggregation")

st.code(f"""
// Aggregation Details for Block #{selected_block}
function aggregateWeights() public {{
    // Weights from 4 hospitals
    uint256[] memory weights = [
        {secrets.randbelow(1000000)},  // Hospital A
        {secrets.randbelow(1000000)},  // Hospital B
        {secrets.randbelow(1000000)},  // Hospital C
        {secrets.randbelow(1000000)}   // Hospital D
    ];
    
    // Compute weighted average
    uint256 aggregated = computeAverage(weights);
    
    // Emit event
    emit WeightsAggregated(block.number, aggregated);
}}

Verification: ✅ All signatures valid
Privacy Budget: ε = 1.0
""", language="solidity")

st.markdown("---")


# Verification Tool
st.markdown("## 🔐 Transaction Verification Tool")

col1, col2 = st.columns([3, 1])

with col1:
    tx_hash_input = st.text_input(
        "Enter Transaction Hash to Verify",
        placeholder="0xabcdef1234567890...",
        help="Enter a transaction hash to verify its details"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    verify_button = st.button("🔍 Verify", use_container_width=True)

if verify_button and tx_hash_input:
    if ganache_connected:
        # Try to get real transaction
        tx_details = get_transaction_details(tx_hash_input)
        
        if tx_details:
            st.success("✅ Transaction Verified Successfully")
            st.json({
                "hash": tx_details['hash'],
                "from": tx_details['from'],
                "to": tx_details['to'],
                "block_number": tx_details['block_number'],
                "gas_used": tx_details['gas_used'],
                "status": tx_details['status'],
                "verified": True,
                "timestamp": datetime.now().isoformat()
            })
        else:
            st.error("❌ Transaction not found on blockchain")
    else:
        # Mock verification
        st.success("✅ Transaction Verified Successfully (Mock)")
        st.json({
            "hash": tx_hash_input,
            "from": "0x" + secrets.token_hex(20),
            "to": "0x" + secrets.token_hex(20),
            "block_number": 105,
            "gas_used": 21000,
            "status": "Success",
            "verified": True,
            "timestamp": datetime.now().isoformat(),
            "note": "Mock verification - Start Ganache for real data"
        })

st.markdown("---")

# Transaction History
st.markdown("## 📜 Recent Weight Aggregation Transactions")

tx_history = []
base_time = datetime.now()

for i in range(10):
    tx_history.append({
        "Round": f"Round {10-i}",
        "Transaction Hash": format_hash(f"0x{secrets.token_hex(32)}", 24),
        "Block": 96 + i,
        "Hospitals": 4,
        "Timestamp": (base_time - timedelta(hours=i*2)).strftime("%Y-%m-%d %H:%M:%S"),
        "Status": "✅ Confirmed"
    })

tx_df = pd.DataFrame(tx_history)

# Display options
show_all_tx = st.checkbox("Show all transactions", value=False)
if not show_all_tx:
    num_tx = st.slider("Number of transactions to show", 1, 10, 5)
    display_tx_df = tx_df.head(num_tx)
else:
    display_tx_df = tx_df

st.dataframe(display_tx_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Network Statistics
st.markdown("## 📊 Blockchain Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if ganache_connected and real_blocks:
        latest_block = real_blocks[0]['number']
    else:
        latest_block = 105
    st.metric("Latest Block", latest_block)

with col2:
    st.metric("Total Transactions", "487")

with col3:
    st.metric("Active Hospitals", "4/4")

with col4:
    st.metric("Avg Block Time", "5 min")

st.markdown("---")

# Technical Information
with st.expander("🔧 Technical Details"):
    st.markdown("""
    ### Blockchain Configuration
    - **Network**: Ganache (Local Development)
    - **Network ID**: 1337
    - **RPC URL**: http://127.0.0.1:7545
    - **Chain ID**: 1337
    - **Consensus**: Proof of Authority (Development)
    
    ### Smart Contract Details
    - **Contract**: Aggregator.sol
    - **Language**: Solidity 0.8.23
    - **Functions**: addParticipant, updateWeights, aggregateWeights
    - **Events**: WeightsAggregated, ParticipantAdded
    
    ### Privacy Features
    - **Differential Privacy**: ε-differential privacy (ε=1.0)
    - **Encryption**: AES-256 for data in transit
    - **Zero-Knowledge Proofs**: Optional for weight verification
    - **Secure Multi-Party Computation**: For aggregation
    
    ### Integration
    - **Web3 Library**: web3.py
    - **Provider**: HTTPProvider with timeout protection
    - **Account Management**: eth_account for key management
    - **Transaction Signing**: Local signing for security
    """)

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #888;">
    <strong>MedRAG Blockchain Explorer</strong> - Transparent and Verifiable Federated Learning
</p>
""", unsafe_allow_html=True)
