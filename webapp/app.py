"""
MedRAG Demo Web Application
Main Streamlit application for demonstrating the MedRAG system.
"""

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

st.set_page_config(
    page_title="MedRAG Demo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4788;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main page
st.markdown('<h1 class="main-header">🏥 MedRAG System</h1>', unsafe_allow_html=True)

# Introduction
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔬 Inference")
    st.markdown("Upload X-rays for COVID-19 detection with RAG-enhanced predictions")
    if st.button("🔬 Go to Inference", key="btn_inference"):
        st.switch_page("pages/1_🔬_Inference.py")

with col2:
    st.markdown("### 📊 Training Dashboard")
    st.markdown("Monitor federated learning training with real-time metrics")
    if st.button("📊 View Dashboard", key="btn_dashboard"):
        st.switch_page("pages/2_📊_Training.py")

with col3:
    st.markdown("### 📦 Model Registry")
    st.markdown("Browse and compare model versions")
    if st.button("📦 View Registry", key="btn_registry"):
        st.switch_page("pages/3_📦_Registry.py")

st.markdown("---")

# System Architecture
st.markdown("## 🏗️ System Architecture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Core Components")
    st.markdown("""
    - **VFL**: Privacy-preserving collaborative learning
    - **Blockchain**: Transparent aggregation
    - **RAG**: Enhanced predictions with medical knowledge
    - **Differential Privacy**: Model update protection
    """)

with col2:
    st.markdown("### Key Features")
    st.markdown("""
    - Privacy-preserving cross-hospital collaboration
    - Blockchain-verified model updates
    - RAG-enhanced diagnostic accuracy
    - Comprehensive audit trail
    """)

st.markdown("---")

