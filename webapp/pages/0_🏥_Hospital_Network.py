"""
Hospital Network Overview Page - Visualize Federated Learning Network
"""

import streamlit as st
import sys
import os
from pathlib import Path
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from webapp.utils import (
    display_system_status,
    display_info_message,
)

st.set_page_config(
    page_title="Hospital Network - MedRAG",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Network Overview")
st.markdown("Visualize the 4-hospital federated learning network topology and data distribution")

# Display system status in sidebar
display_system_status()

# ============================================================================
# Data Loading from SplitCovid19
# ============================================================================

# Try multiple possible paths to locate the dataset
_base_search_dirs = [
    Path("src/data"),           # If running from project root
    Path("../src/data"),        # If running from webapp directory
    Path("../../src/data"),     # If running from webapp/pages
    Path(__file__).parent.parent.parent / "src" / "data",  # Absolute path relative to this file
    Path("data"),               # If running from project root with data at top level
    Path("../data"),
    Path(__file__).parent.parent.parent / "data",
]

possible_paths = [d / "SplitCovid19" for d in _base_search_dirs]

dataset_path = None
for path in possible_paths:
    if path.exists():
        dataset_path = path
        break

# ============================================================================
# Hospital Statistics
# ============================================================================

def count_images_in_directory(directory):
    """Count image files in a directory."""
    if not directory.exists():
        return 0
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    count = 0
    for file in directory.rglob('*'):
        if file.suffix in image_extensions:
            count += 1
    return count

# Load real data if dataset is found
if dataset_path:
    display_info_message(f"✅ Dataset found at: {dataset_path.absolute()}")
    
    hospitals = []
    # Support both zip-loader format (hospitalA/B/C/D) and legacy format (client0/1/2/3)
    hospital_mapping = [
        ('hospitalA', 'client0', 'Hospital A - Chennai', '#667eea'),
        ('hospitalB', 'client1', 'Hospital B - Bangalore', '#4caf50'),
        ('hospitalC', 'client2', 'Hospital C - Hyderabad', '#ff9800'),
        ('hospitalD', 'client3', 'Hospital D - Mumbai', '#f44336'),
    ]
    
    for new_dir, legacy_dir, hospital_name, color in hospital_mapping:
        # Prefer new hospitalA/B/C/D naming (from zip loader), fall back to client0/1/2/3
        if (dataset_path / new_dir).exists():
            client_path = dataset_path / new_dir
        elif (dataset_path / legacy_dir).exists():
            client_path = dataset_path / legacy_dir
        else:
            continue

        # Count images for all known classes; sum non-covid classes as 'normal'
        train_covid = count_images_in_directory(client_path / 'train' / 'covid')
        train_normal = count_images_in_directory(client_path / 'train' / 'normal')
        test_covid = count_images_in_directory(client_path / 'test' / 'covid')
        test_normal = count_images_in_directory(client_path / 'test' / 'normal')

        # Also count other classes if present (pneumonia, etc.)
        for extra_class in ['pneumonia', 'tuberculosis', 'lung_opacity', 'unknown']:
            train_normal += count_images_in_directory(client_path / 'train' / extra_class)
            test_normal += count_images_in_directory(client_path / 'test' / extra_class)

        hospitals.append({
            'name': hospital_name,
            'color': color,
            'train_covid': train_covid,
            'train_normal': train_normal,
            'test_covid': test_covid,
            'test_normal': test_normal,
            'total_images': train_covid + train_normal + test_covid + test_normal,
            'status': '🟢 Online'
        })
else:
    st.warning("⚠️ Dataset not found in expected locations. Using demo values.")
    st.info("Searched paths:\n" + "\n".join([f"- {p}" for p in possible_paths]))
    
    # Demo values
    hospitals = [
        {'name': 'Hospital A - Chennai', 'color': '#667eea', 'train_covid': 45, 'train_normal': 52, 'test_covid': 12, 'test_normal': 13, 'total_images': 122, 'status': '🟢 Online'},
        {'name': 'Hospital B - Bangalore', 'color': '#4caf50', 'train_covid': 48, 'train_normal': 50, 'test_covid': 11, 'test_normal': 14, 'total_images': 123, 'status': '🟢 Online'},
        {'name': 'Hospital C - Hyderabad', 'color': '#ff9800', 'train_covid': 47, 'train_normal': 51, 'test_covid': 13, 'test_normal': 12, 'total_images': 123, 'status': '🟢 Online'},
        {'name': 'Hospital D - Mumbai', 'color': '#f44336', 'train_covid': 46, 'train_normal': 53, 'test_covid': 11, 'test_normal': 13, 'total_images': 123, 'status': '🟢 Online'},
    ]

# ============================================================================
# Network Visualization
# ============================================================================

st.markdown("## 🌐 Network Topology")

col1, col2 = st.columns([2, 1])

with col1:
    # Create network diagram
    fig = go.Figure()
    
    # Central aggregator
    fig.add_trace(go.Scatter(
        x=[0.5], y=[0.5],
        mode='markers+text',
        marker=dict(size=60, color='#667eea', line=dict(width=2, color='white')),
        text=['Central<br>Aggregator'],
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial Black'),
        hoverinfo='text',
        hovertext='Blockchain-enabled Federated Aggregator',
        name='Aggregator'
    ))
    
    # Hospital nodes in a circle
    import math
    num_hospitals = len(hospitals)
    for i, hospital in enumerate(hospitals):
        angle = 2 * math.pi * i / num_hospitals - math.pi / 2
        x = 0.5 + 0.35 * math.cos(angle)
        y = 0.5 + 0.35 * math.sin(angle)
        
        # Hospital node
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=50, color=hospital['color'], line=dict(width=2, color='white')),
            text=[hospital['name'].split('-')[0].strip()],
            textposition='middle center',
            textfont=dict(size=9, color='white', family='Arial Black'),
            hoverinfo='text',
            hovertext=f"{hospital['name']}<br>Total Images: {hospital['total_images']}<br>Status: {hospital['status']}",
            name=hospital['name']
        ))
        
        # Connection line
        fig.add_trace(go.Scatter(
            x=[x, 0.5], y=[y, 0.5],
            mode='lines',
            line=dict(color=hospital['color'], width=2, dash='dot'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 1]),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 📊 Network Summary")
    st.metric("Total Hospitals", len(hospitals))
    st.metric("Total Training Images", sum(h['train_covid'] + h['train_normal'] for h in hospitals))
    st.metric("Total Test Images", sum(h['test_covid'] + h['test_normal'] for h in hospitals))
    st.metric("Network Status", "🟢 Operational")

# ============================================================================
# Hospital Details
# ============================================================================

st.markdown("## 🏥 Hospital Details")

cols = st.columns(len(hospitals))
for idx, hospital in enumerate(hospitals):
    with cols[idx]:
        st.markdown(f"### {hospital['name'].split('-')[0].strip()}")
        st.markdown(f"**Location:** {hospital['name'].split('-')[1].strip()}")
        st.markdown(f"**Status:** {hospital['status']}")
        
        st.markdown("**Training Data:**")
        st.markdown(f"- COVID: {hospital['train_covid']}")
        st.markdown(f"- Normal: {hospital['train_normal']}")
        
        st.markdown("**Test Data:**")
        st.markdown(f"- COVID: {hospital['test_covid']}")
        st.markdown(f"- Normal: {hospital['test_normal']}")
        
        st.markdown(f"**Total:** {hospital['total_images']} images")

# ============================================================================
# Data Distribution Chart
# ============================================================================

st.markdown("## 📈 Data Distribution Across Hospitals")

fig = go.Figure()

hospital_names = [h['name'].split('-')[0].strip() for h in hospitals]

fig.add_trace(go.Bar(
    name='Training COVID',
    x=hospital_names,
    y=[h['train_covid'] for h in hospitals],
    marker_color='#f44336'
))

fig.add_trace(go.Bar(
    name='Training Normal',
    x=hospital_names,
    y=[h['train_normal'] for h in hospitals],
    marker_color='#4caf50'
))

fig.add_trace(go.Bar(
    name='Test COVID',
    x=hospital_names,
    y=[h['test_covid'] for h in hospitals],
    marker_color='#ff9800'
))

fig.add_trace(go.Bar(
    name='Test Normal',
    x=hospital_names,
    y=[h['test_normal'] for h in hospitals],
    marker_color='#2196f3'
))

fig.update_layout(
    barmode='group',
    xaxis_title="Hospital",
    yaxis_title="Number of Images",
    height=400,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)
