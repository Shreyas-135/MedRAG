"""
Inference Page - X-ray Upload and Prediction

Blockchain capstone features:
  - Model Governance Gate: verifies the selected model version is APPROVED
    on-chain (3-of-4 hospital multi-sig) before allowing inference.
  - Provenance Verification Gate: after inference, the user must anchor and
    verify the provenance bundle on-chain (or in mock mode) before the
    explanation/result can be exported.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from PIL import Image
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from webapp.utils import (
    get_inference_engine,
    get_model_registry,
    get_ledger,
    display_system_status,
    display_error_message,
    display_success_message,
    display_info_message,
    check_model_governance_approval,
)

st.set_page_config(
    page_title="Inference - MedRAG",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 X-Ray Analysis & Prediction")
st.markdown("Upload a chest X-ray image for AI-powered COVID-19 detection with RAG-enhanced explanations.")

# Display system status in sidebar
display_system_status()

# ============================================================================
# Session-state initialisation for verification gate
# ============================================================================
if 'prov_verified' not in st.session_state:
    st.session_state['prov_verified'] = False
if 'admin_override' not in st.session_state:
    st.session_state['admin_override'] = False

# ============================================================================
# Main content
# ============================================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, PNG)",
        type=['jpg', 'jpeg', 'png']
    )

    # Model selection
    registry = get_model_registry()
    versions = registry.get_model_history()

    if versions:
        version_options = ["Latest Model"] + [v.version_id for v in versions]
        selected_version = st.selectbox(
            "Select Model Version",
            options=version_options
        )

        if selected_version != "Latest Model":
            version_info = registry.get_version(selected_version)
            if version_info:
                st.info(f"Accuracy: {version_info.metrics.get('test_accuracy', 'N/A'):.2f}%")
    else:
        display_info_message("No trained models available. Train a model first using demo_rag_vfl.py")
        selected_version = None

    # -----------------------------------------------------------------------
    # Model Governance Gate
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### ⛓️ Model Governance Status")
    _gov_result = check_model_governance_approval(selected_version or "unknown")
    _gov_approved = _gov_result.get("is_approved", False)
    _gov_status   = _gov_result.get("status", "UNKNOWN")
    _gov_count    = _gov_result.get("approval_count", 0)
    _gov_required = _gov_result.get("required_approvals", 3)
    _gov_mock     = _gov_result.get("mock_mode", True)

    if _gov_mock:
        st.info(
            f"ℹ️ Governance running in **mock mode** – status: **{_gov_status}** "
            f"({_gov_count}/{_gov_required} approvals). "
            "Connect Ganache for live governance."
        )
    elif _gov_approved:
        st.success(f"✅ Model APPROVED on-chain ({_gov_count}/{_gov_required} approvals)")
    else:
        st.error(
            f"🚫 Model NOT APPROVED on-chain – status: **{_gov_status}** "
            f"({_gov_count}/{_gov_required} approvals). "
            "Inference is blocked. Have hospital administrators approve this model version."
        )

    # Preview uploaded image
    if uploaded_file is not None:
        st.markdown("### 🖼️ Image Preview")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

with col2:
    st.markdown("### 🔍 Analysis Results")

    # Block inference if governance check hard-fails (non-mock, not approved)
    _governance_blocks = (
        not _gov_mock
        and not _gov_approved
    )

    if _governance_blocks:
        st.error(
            "🚫 **Inference blocked** – the selected model version has not received "
            "the required on-chain governance approvals. "
            "Please select an APPROVED model version or contact an administrator."
        )
    elif uploaded_file is not None and st.button("🚀 Analyze Image", type="primary", use_container_width=True):
        # Save uploaded file temporarily
        temp_dir = Path("/tmp/medrag_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Get inference engine
            with st.spinner("Loading inference model..."):
                version_id = selected_version if selected_version != "Latest Model" else None
                inference = get_inference_engine(_version_id=version_id)
            
            if inference is None:
                display_error_message("Failed to load inference engine")
            else:
                # Run inference
                with st.spinner("Analyzing X-ray image..."):
                    result = inference.predict(str(temp_path), return_explanations=True)
                
                # Log access to ledger
                ledger = get_ledger()
                ledger.log_access(
                    user_id="webapp_user",
                    action="predict",
                    resource=uploaded_file.name,
                    status="success",
                    details=result
                )
                
                # Display results
                st.markdown("---")

                prediction = result['prediction']
                confidence = float(result.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))

                # Prediction header
                st.markdown(f"### 🩺 Prediction: **{prediction.upper()}**")
                st.markdown(f"**Confidence:** {confidence:.1%}")
                st.progress(confidence)

                st.markdown("---")

                # Probabilities (render dynamically; don't hardcode keys)
                st.markdown("### 📊 Class Probabilities")
                probabilities = result.get("probabilities") or {}
                items = list(probabilities.items())
                if not items:
                    st.info("No class probabilities returned.")
                else:
                    cols = st.columns(min(len(items), 4))
                    for i, (cls, prob) in enumerate(items):
                        with cols[i % min(len(items), 4)]:
                            st.metric(str(cls).replace("_", " ").title(), f"{float(prob):.1%}")

                st.markdown("---")

                # RAG Explanation
                st.markdown("### 🧠 RAG Explanation")
                explanation_text = result.get("explanation_text") or result.get("rag_explanation", "")
                if explanation_text:
                    st.info(explanation_text)
                else:
                    guidelines = result.get("guidelines", [])
                    if guidelines:
                        for g in guidelines:
                            st.markdown(f"- {g}")
                    else:
                        st.info("No RAG explanation available for this prediction.")

                st.markdown("---")

                # Citations
                st.markdown("### 📚 Citations")
                citations = result.get("citations", [])
                if citations:
                    for idx, cit in enumerate(citations, 1):
                        source = cit.get("source", "Unknown source")
                        url = cit.get("url", "")
                        snippet = cit.get("snippet", "")
                        title = cit.get("title", "")
                        with st.expander(f"[{idx}] {source}"):
                            if title:
                                st.markdown(f"**{title}**")
                            if snippet:
                                st.markdown(f"*{snippet}*")
                            if url:
                                st.markdown(f"🔗 [{url}]({url})")
                else:
                    st.info("No citations available.")

                st.markdown("---")

                # Technical Details
                with st.expander("🔧 Technical Details"):
                    st.markdown(f"""
                    - **Model Type**: {result['model_type']}
                    - **Inference Time**: {result['inference_time']:.3f} seconds
                    - **Model Version**: {selected_version}
                    - **Image**: {uploaded_file.name}
                    """)

                # ---- Export gate ----
                _can_export = (
                    st.session_state.get('prov_verified', False)
                    or st.session_state.get('admin_override', False)
                )
                if _can_export:
                    import json as _json
                    _export_data = _json.dumps({
                        "prediction": result.get('prediction'),
                        "confidence": result.get('confidence'),
                        "probabilities": result.get('probabilities'),
                        "provenance": st.session_state.get('prov_bundle'),
                        "tx_hash": st.session_state.get('prov_tx_hash'),
                        "anchor_info": st.session_state.get('prov_anchor_info'),
                        "admin_override": st.session_state.get('admin_override', False),
                    }, indent=2, default=str)
                    if st.session_state.get('admin_override') and not st.session_state.get('prov_verified'):
                        st.warning(
                            "⚠️ **Admin Override Active** – this export was NOT verified on-chain."
                        )
                    st.download_button(
                        label="⬇️ Download Result & Provenance (JSON)",
                        data=_export_data,
                        file_name="medrag_result_provenance.json",
                        mime="application/json",
                    )
                else:
                    st.info(
                        "🔒 Export locked – anchor provenance on-chain (Step 3) to enable download."
                    )

                display_success_message("Analysis complete!")
        
        except Exception as e:
            display_error_message(f"Error during inference: {str(e)}")
            
            # Log failed access
            ledger = get_ledger()
            ledger.log_access(
                user_id="webapp_user",
                action="predict",
                resource=uploaded_file.name,
                status="failure",
                details={"error": str(e)}
            )
    
    elif uploaded_file is None:
        display_info_message("👆 Please upload an X-ray image to begin analysis")
