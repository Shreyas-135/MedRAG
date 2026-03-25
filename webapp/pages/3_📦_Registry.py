"""
Model Registry Page - Browse and Compare Models
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from webapp.utils import (
    get_model_registry,
    display_system_status,
    format_timestamp,
    format_hash
)

st.set_page_config(
    page_title="Model Registry - MedRAG",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Model Registry")
st.markdown("Browse, compare, and manage model versions across training rounds")

# Display system status in sidebar
display_system_status()

# Get registry
registry = get_model_registry()
summary = registry.get_summary()

# ============================================================================
# SECTION 1: Enhanced Overview with Visual Indicators
# ============================================================================
st.markdown("## 📊 Registry Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Versions", summary['total_versions'], delta="+1 recent" if summary['total_versions'] > 0 else None)

with col2:
    if summary.get('latest_version'):
        latest_num = summary['latest_version'].split('_')[0]
        st.metric("Latest Version", latest_num, delta="Active")
    else:
        st.metric("Latest Version", "None")

with col3:
    if summary.get('best_accuracy'):
        st.metric("Best Accuracy", f"{summary['best_accuracy']:.2f}%", delta="🏆 Top")
    else:
        st.metric("Best Accuracy", "N/A")

with col4:
    st.metric("Storage", f"{summary['storage_size_mb']:.1f} MB", delta=f"{summary['total_versions']} models")

st.markdown("---")

# ============================================================================
# SECTION 2: Model Versions Table (Enhanced)
# ============================================================================
versions = registry.get_model_history()

if not versions:
    st.info("""
    ### 🎯 No Models Yet - Start Training!
    
    Train your first model:
    ```bash
    cd src
    python demo_rag_vfl.py --datapath ../demo_data --use-rag --num-epochs 3
    ```
    """)
    st.stop()

st.markdown("## 🗂️ Model Versions")

# Convert to enhanced DataFrame
version_data = []
for i, v in enumerate(versions):
    # Calculate improvement from previous version
    improvement = ""
    if i < len(versions) - 1:
        prev_acc = versions[i+1].metrics.get('test_accuracy', 0)
        curr_acc = v.metrics.get('test_accuracy', 0)
        if isinstance(prev_acc, (int, float)) and isinstance(curr_acc, (int, float)):
            diff = curr_acc - prev_acc
            improvement = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"
    
    version_data.append({
        'Version': v.version_id[:16],
        'Round': v.round_num,
        'Accuracy': f"{v.metrics.get('test_accuracy', 0):.2f}%" if isinstance(v.metrics.get('test_accuracy'), (int, float)) else 'N/A',
        'Improvement': improvement,
        'Loss': f"{v.metrics.get('test_loss', 0):.4f}" if isinstance(v.metrics.get('test_loss'), (int, float)) else 'N/A',
        'RAG': '✅' if v.config.get('use_rag', False) else '❌',
        'Blockchain': '✅' if v.config.get('use_blockchain', False) else '❌',
        'Date': format_timestamp(v.timestamp),
        'Status': '🟢 Active' if i == 0 else '⚪ Archived'
    })

df = pd.DataFrame(version_data)
st.dataframe(df, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================================
# NEW SECTION 3: Performance Trend Chart
# ============================================================================
st.markdown("## 📈 Performance Trends Over Training Rounds")

if len(versions) > 1:
    # Prepare data for chart
    rounds = []
    accuracies = []
    losses = []
    
    for v in reversed(versions):  # Oldest to newest
        if isinstance(v.metrics.get('test_accuracy'), (int, float)):
            rounds.append(v.round_num)
            accuracies.append(v.metrics.get('test_accuracy', 0))
            losses.append(v.metrics.get('test_loss', 0))
    
    # Create dual-axis chart
    fig = go.Figure()
    
    # Accuracy line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy (%)',
        line=dict(color='#4CAF50', width=3),
        marker=dict(size=10),
        yaxis='y'
    ))
    
    # Loss line
    fig.add_trace(go.Scatter(
        x=rounds,
        y=losses,
        mode='lines+markers',
        name='Loss',
        line=dict(color='#FF5722', width=3),
        marker=dict(size=10),
        yaxis='y2'
    ))
    
    fig.update_layout(
        xaxis=dict(title='Training Round'),
        yaxis=dict(title=dict(text='Accuracy (%)', font=dict(color='#4CAF50')),tickfont=dict(color='#4CAF50')
        ),
        yaxis2=dict(title=dict(text='Loss', font=dict(color='#FF5722')),tickfont=dict(color='#FF5722'),overlaying='y',
        side='right'
        ),
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("📊 Train more models to see performance trends")

st.markdown("---")

# ============================================================================
# NEW SECTION 4: Model Feature Comparison
# ============================================================================
st.markdown("## 🔬 Model Feature Analysis")

if len(versions) >= 2:
    col1, col2 = st.columns(2)
    
    with col1:
        # RAG vs Non-RAG comparison
        rag_models = [v for v in versions if v.config.get('use_rag', False)]
        non_rag_models = [v for v in versions if not v.config.get('use_rag', False)]
        
        if rag_models and non_rag_models:
            avg_rag_acc = sum(v.metrics.get('test_accuracy', 0) for v in rag_models) / len(rag_models)
            avg_non_rag_acc = sum(v.metrics.get('test_accuracy', 0) for v in non_rag_models) / len(non_rag_models)
            
            st.markdown("### 🧠 RAG Impact")
            st.metric(
                "RAG Models Avg Accuracy",
                f"{avg_rag_acc:.2f}%",
                delta=f"+{avg_rag_acc - avg_non_rag_acc:.2f}%" if avg_rag_acc > avg_non_rag_acc else None
            )
            st.metric("Non-RAG Models Avg", f"{avg_non_rag_acc:.2f}%")
    
    with col2:
        # Blockchain vs Non-Blockchain
        bc_models = [v for v in versions if v.config.get('use_blockchain', False)]
        non_bc_models = [v for v in versions if not v.config.get('use_blockchain', False)]
        
        if bc_models and non_bc_models:
            avg_bc_acc = sum(v.metrics.get('test_accuracy', 0) for v in bc_models) / len(bc_models)
            avg_non_bc_acc = sum(v.metrics.get('test_accuracy', 0) for v in non_bc_models) / len(non_bc_models)
            
            st.markdown("### ⛓️ Blockchain Impact")
            st.metric(
                "Blockchain Models Avg",
                f"{avg_bc_acc:.2f}%",
                delta=f"+{avg_bc_acc - avg_non_bc_acc:.2f}%" if avg_bc_acc > avg_non_bc_acc else None
            )
            st.metric("Standard Models Avg", f"{avg_non_bc_acc:.2f}%")

st.markdown("---")

# ============================================================================
# SECTION 5: Model Details (Enhanced)
# ============================================================================
st.markdown("## 🔍 Detailed Model Information")

version_ids = [v.version_id for v in versions]
selected_version_id = st.selectbox(
    "Select a model to view full details",
    options=version_ids,
    index=0
)

if selected_version_id:
    selected_version = registry.get_version(selected_version_id)
    
    if selected_version:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Basic Information")
            st.markdown(f"""
            - **Version ID**: `{selected_version.version_id[:32]}...`
            - **Training Round**: {selected_version.round_num}
            - **Created**: {format_timestamp(selected_version.timestamp)}
            - **Model Hash**: `{selected_version.model_hash[:32]}...`
            - **Checkpoint**: `{Path(selected_version.checkpoint_path).name}`
            - **File Size**: {Path(selected_version.checkpoint_path).stat().st_size / (1024*1024):.2f} MB
            """)
            
            st.markdown("### ⚙️ Training Configuration")
            config_items = []
            for k, v in selected_version.config.items():
                if isinstance(v, bool):
                    config_items.append(f"- **{k}**: {'✅ Yes' if v else '❌ No'}")
                else:
                    config_items.append(f"- **{k}**: {v}")
            st.markdown("\n".join(config_items))
        
        with col2:
            st.markdown("### 📊 Performance Metrics")
            
            # Display metrics with visual indicators
            for metric_name, metric_value in selected_version.metrics.items():
                if isinstance(metric_value, (int, float)):
                    if 'accuracy' in metric_name.lower():
                        # Color code accuracy
                        if metric_value >= 90:
                            st.success(f"🟢 **{metric_name.replace('_', ' ').title()}**: {metric_value:.2f}%")
                        elif metric_value >= 80:
                            st.info(f"🔵 **{metric_name.replace('_', ' ').title()}**: {metric_value:.2f}%")
                        else:
                            st.warning(f"🟡 **{metric_name.replace('_', ' ').title()}**: {metric_value:.2f}%")
                    elif 'loss' in metric_name.lower():
                        st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.4f}")
                    else:
                        st.metric(metric_name.replace('_', ' ').title(), f"{metric_value}")
            
            # NEW: Hospital-specific metrics (if available)
            st.markdown("### 🏥 Per-Hospital Performance")
            st.info("📝 Note: Hospital-specific metrics available after training")
        
        st.markdown("---")
        
        # Download section
        st.markdown("### 💾 Export Model")
        checkpoint_path = Path(selected_version.checkpoint_path)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if checkpoint_path.exists():
                file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
                st.info(f"📦 Size: {file_size_mb:.2f} MB")
                
                with open(checkpoint_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Checkpoint",
                        data=f,
                        file_name=checkpoint_path.name,
                        mime='application/octet-stream',
                        use_container_width=True
                    )
        
        with col2:
            # Export model metadata
            import json
            metadata = {
                'version_id': selected_version.version_id,
                'round': selected_version.round_num,
                'metrics': selected_version.metrics,
                'config': selected_version.config,
                'timestamp': selected_version.timestamp,
                'hash': selected_version.model_hash
            }
            
            st.download_button(
                label="📄 Export Metadata",
                data=json.dumps(metadata, indent=2),
                file_name=f"{selected_version.version_id}_metadata.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            if st.button("🚀 Deploy This Model", use_container_width=True):
                st.success("✅ Model marked for deployment!")
                st.info("In production: This would deploy to inference servers")

st.markdown("---")

# ============================================================================
# SECTION 6: Model Comparison (Enhanced)
# ============================================================================
st.markdown("## ⚖️ Side-by-Side Model Comparison")

if len(versions) >= 2:
    compare_versions = st.multiselect(
        "Select models to compare (2-4 recommended)",
        options=version_ids,
        default=version_ids[:min(3, len(version_ids))]
    )
    
    if len(compare_versions) >= 2:
        comparison_data = []
        for vid in compare_versions:
            v = registry.get_version(vid)
            if v:
                row = {
                    'Version': v.version_id[:16],
                    'Round': v.round_num,
                    'RAG': '✅' if v.config.get('use_rag') else '❌',
                    'Blockchain': '✅' if v.config.get('use_blockchain') else '❌',
                }
                # Add metrics
                for metric_name, metric_value in v.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if 'accuracy' in metric_name.lower():
                            row[metric_name] = f"{metric_value:.2f}%"
                        elif 'loss' in metric_name.lower():
                            row[metric_name] = f"{metric_value:.4f}"
                        else:
                            row[metric_name] = metric_value
                
                comparison_data.append(row)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            # Highlight best
            best_version = registry.get_best_model('accuracy')
            if best_version and best_version.version_id in compare_versions:
                st.success(f"🏆 **Recommended Model**: {best_version.version_id} (Highest Accuracy)")
else:
    st.info("📊 Train at least 2 models to enable comparison")

st.markdown("---")

# ============================================================================
# NEW SECTION 7: Deployment Recommendations
# ============================================================================
st.markdown("## 💡 Deployment Recommendations")

if versions:
    best_model = registry.get_best_model('accuracy')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏆 Best Overall")
        if best_model:
            st.success(f"""
            **Version**: {best_model.version_id[:16]}...  
            **Accuracy**: {best_model.metrics.get('test_accuracy', 0):.2f}%  
            **Round**: {best_model.round_num}
            """)
    
    with col2:
        st.markdown("### 🧠 Best RAG Model")
        rag_models = [v for v in versions if v.config.get('use_rag', False)]
        if rag_models:
            best_rag = max(rag_models, key=lambda v: v.metrics.get('test_accuracy', 0))
            st.info(f"""
            **Version**: {best_rag.version_id[:16]}...  
            **Accuracy**: {best_rag.metrics.get('test_accuracy', 0):.2f}%  
            **Round**: {best_rag.round_num}
            """)
    
    with col3:
        st.markdown("### ⛓️ Best Blockchain Model")
        bc_models = [v for v in versions if v.config.get('use_blockchain', False)]
        if bc_models:
            best_bc = max(bc_models, key=lambda v: v.metrics.get('test_accuracy', 0))
            st.info(f"""
            **Version**: {best_bc.version_id[:16]}...  
            **Accuracy**: {best_bc.metrics.get('test_accuracy', 0):.2f}%  
            **Round**: {best_bc.round_num}
            """)

st.markdown("---")

# ============================================================================
# SECTION 8: Export Options
# ============================================================================
st.markdown("## 💾 Bulk Export Options")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export All as JSON", use_container_width=True):
        import json
        registry_data = {vid: v.to_dict() for vid, v in registry.versions.items()}
        json_str = json.dumps(registry_data, indent=2)
        
        st.download_button(
            label="Download registry.json",
            data=json_str,
            file_name="model_registry_full.json",
            mime="application/json"
        )

with col2:
    if st.button("📊 Export as CSV", use_container_width=True):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download models.csv",
            data=csv,
            file_name="model_registry.csv",
            mime="text/csv"
        )

with col3:
    if st.button("📋 Generate Report", use_container_width=True):
        report = f"""
# MedRAG Model Registry Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary
- Total Models: {summary['total_versions']}
- Best Accuracy: {summary.get('best_accuracy', 'N/A')}%
- Storage Used: {summary['storage_size_mb']:.2f} MB

## Top 5 Models
{chr(10).join([f"- {v.version_id}: {v.metrics.get('test_accuracy', 0):.2f}%" for v in versions[:5]])}
        """
        
        st.download_button(
            label="Download report.md",
            data=report,
            file_name="model_registry_report.md",
            mime="text/markdown"
        )

# Refresh button
st.markdown("---")
if st.button("🔄 Refresh Registry", use_container_width=True, type="primary"):
    st.cache_resource.clear()
    st.rerun()

# Footer
st.markdown("---")
st.caption("💡 Tip: Train models with different configurations (RAG, Blockchain) to compare performance")