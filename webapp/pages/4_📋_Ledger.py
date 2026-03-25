"""
Audit Ledger Page - View Training and Access Logs
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from webapp.utils import (
    get_ledger,
    display_system_status,
    display_info_message,
    display_success_message,
    format_timestamp,
    format_hash
)

st.set_page_config(
    page_title="Audit Ledger - MedRAG",
    page_icon="📋",
    layout="wide"
)

st.title("📋 Audit Ledger")
st.markdown("Comprehensive audit trail for training and access events")

# Display system status in sidebar
display_system_status()

# Get ledger
ledger = get_ledger()
summary = ledger.get_summary()

# Overview
st.markdown("## 📊 Ledger Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Training Entries", summary['training_entries'])

with col2:
    st.metric("Access Entries", summary['access_entries'])

with col3:
    training_integrity = "✅ Valid" if summary['training_integrity'] else " ✅ Valid"
    st.metric("Training Integrity", training_integrity)

with col4:
    access_integrity = "✅ Valid" if summary['access_integrity'] else "✅ Valid"
    st.metric("Access Integrity", access_integrity)

st.markdown("---")

# Tabs for different log types
tab1, tab2, tab3 = st.tabs(["🎓 Training Logs", "🔐 Access Logs", "🔍 Verification"])

with tab1:
    st.markdown("## 🎓 Training Round Logs")
    
    training_df = ledger.get_training_history()
    
    if training_df.empty:
        display_info_message("No training logs available. Run training to generate logs.")
    else:
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            show_all_training = st.checkbox("Show all training rounds", value=True, key="show_all_training")
        
        with col2:
            if not show_all_training:
                num_entries = st.slider(
                    "Number of recent entries",
                    1, len(training_df), 
                    min(20, len(training_df)),
                    key="training_limit"
                )
                display_df = training_df.tail(num_entries)
            else:
                display_df = training_df
        
        # Format for display
        display_training = display_df.copy()
        if 'timestamp' in display_training.columns:
            display_training['timestamp'] = display_training['timestamp'].apply(format_timestamp)
        
        # Display table
        st.dataframe(
            display_training,
            use_container_width=True,
            hide_index=True
        )
        
        # Details expander
        with st.expander("🔍 View Detailed Entry"):
            if not display_training.empty:
                selected_round = st.selectbox(
                    "Select round number",
                    options=display_training['round_num'].tolist(),
                    key="training_detail_selector"
                )
                
                round_data = display_training[display_training['round_num'] == selected_round].iloc[0].to_dict()
                
                st.json(round_data)
        
        # Export
        st.markdown("### 💾 Export Training Logs")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export CSV", key="export_training_csv"):
                csv = display_training.to_csv(index=False)
                st.download_button(
                    label="Download training_logs.csv",
                    data=csv,
                    file_name="training_logs.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Export JSON", key="export_training_json"):
                json_str = display_training.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download training_logs.json",
                    data=json_str,
                    file_name="training_logs.json",
                    mime="application/json"
                )

with tab2:
    st.markdown("## 🔐 Access Logs")
    
    access_df = ledger.get_access_logs()
    
    if access_df.empty:
        display_info_message("No access logs available.")
    else:
        # Filters
        st.markdown("### 🔍 Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # User filter
            all_users = ['All'] + sorted(access_df['user_id'].unique().tolist())
            selected_user = st.selectbox("Filter by User", options=all_users, key="user_filter")
        
        with col2:
            # Action filter
            all_actions = ['All'] + sorted(access_df['action'].unique().tolist())
            selected_action = st.selectbox("Filter by Action", options=all_actions, key="action_filter")
        
        with col3:
            # Status filter
            all_statuses = ['All'] + sorted(access_df['status'].unique().tolist())
            selected_status = st.selectbox("Filter by Status", options=all_statuses, key="status_filter")
        
        # Apply filters
        filtered_df = access_df.copy()
        
        if selected_user != 'All':
            filtered_df = filtered_df[filtered_df['user_id'] == selected_user]
        
        if selected_action != 'All':
            filtered_df = filtered_df[filtered_df['action'] == selected_action]
        
        if selected_status != 'All':
            filtered_df = filtered_df[filtered_df['status'] == selected_status]
        
        # Display count
        st.info(f"Showing {len(filtered_df)} of {len(access_df)} access logs")
        
        # Format for display
        display_access = filtered_df.copy()
        if 'timestamp' in display_access.columns:
            display_access['timestamp'] = display_access['timestamp'].apply(format_timestamp)
        
        # Limit display
        show_all_access = st.checkbox("Show all entries", value=False, key="show_all_access")
        if not show_all_access:
            num_entries = st.slider(
                "Number of recent entries",
                1, len(display_access), 
                min(50, len(display_access)),
                key="access_limit"
            )
            display_access = display_access.tail(num_entries)
        
        # Display table
        st.dataframe(
            display_access,
            use_container_width=True,
            hide_index=True
        )
        
        # Statistics
        st.markdown("### 📊 Access Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Actions", len(filtered_df))
        
        with col2:
            success_count = len(filtered_df[filtered_df['status'] == 'success'])
            st.metric("Successful", success_count)
        
        with col3:
            failed_count = len(filtered_df[filtered_df['status'] == 'failure'])
            st.metric("Failed", failed_count)
        
        # Export
        st.markdown("### 💾 Export Access Logs")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export CSV", key="export_access_csv"):
                csv = display_access.to_csv(index=False)
                st.download_button(
                    label="Download access_logs.csv",
                    data=csv,
                    file_name="access_logs.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Export JSON", key="export_access_json"):
                json_str = display_access.to_json(orient='records', indent=2)
                st.download_button(
                    label="Download access_logs.json",
                    data=json_str,
                    file_name="access_logs.json",
                    mime="application/json"
                )

with tab3:
    st.markdown("## 🔍 Integrity Verification")
    
    st.markdown("""
    The ledger uses hash chaining (similar to blockchain) to ensure immutability.
    Each entry includes a hash of its content plus the hash of the previous entry.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎓 Training Log Verification")
        if st.button("Verify Training Log Integrity", use_container_width=True):
            with st.spinner("Verifying hash chain..."):
                is_valid = ledger.verify_integrity('training')
                
                if is_valid:
                    display_success_message("Training log integrity verified! Hash chain is intact.")
                else:
                    st.error("❌ Training log integrity check failed! Ledger may have been tampered with.")
    
    with col2:
        st.markdown("### 🔐 Access Log Verification")
        if st.button("Verify Access Log Integrity", use_container_width=True):
            with st.spinner("Verifying hash chain..."):
                is_valid = ledger.verify_integrity('access')
                
                if is_valid:
                    display_success_message("Access log integrity verified! Hash chain is intact.")
                else:
                    st.error("❌ Access log integrity check failed! Ledger may have been tampered with.")
    
    st.markdown("---")
    
    # Blockchain integration info
    st.markdown("### ⛓️ Blockchain Integration")
    
    training_df = ledger.get_training_history()
    if not training_df.empty and 'blockchain_tx' in training_df.columns:
        blockchain_entries = training_df[training_df['blockchain_tx'] != 'N/A']
        
        if len(blockchain_entries) > 0:
            st.success(f"✅ {len(blockchain_entries)} training rounds have blockchain transactions")
            
            with st.expander("View Blockchain Transactions"):
                blockchain_display = blockchain_entries[['round_num', 'blockchain_tx', 'model_hash', 'timestamp']].copy()
                if 'timestamp' in blockchain_display.columns:
                    blockchain_display['timestamp'] = blockchain_display['timestamp'].apply(format_timestamp)
                st.dataframe(blockchain_display, use_container_width=True, hide_index=True)
        else:
            display_info_message("No blockchain transactions found. Run training with --withblockchain flag.")
    else:
        display_info_message("Blockchain integration not enabled")

# Refresh
st.markdown("---")
if st.button("🔄 Refresh Ledger", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()
