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
    get_ensemble_engine,
    get_model_registry,
    get_ledger,
    display_system_status,
    display_error_message,
    display_success_message,
    display_info_message,
    check_model_governance_approval,
    generate_clinician_pdf,
    compute_inference_provenance_hashes,
)

st.set_page_config(
    page_title="Inference - MedRAG",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 X-Ray Analysis & Prediction")
st.markdown(
    "Upload a chest X-ray image for AI-powered multi-class detection using "
    "a weighted-average ensemble of three virtual hospital backbones "
    "(ResNet-18 · DenseNet-121 · EfficientNet-B0) with clinician-grade reporting."
)

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
# Optional Patient Information (sidebar / expander)
# ============================================================================
with st.expander("👤 Patient Information (optional)", expanded=False):
    pi_col1, pi_col2 = st.columns(2)
    with pi_col1:
        pt_name   = st.text_input("Patient Name",          placeholder="e.g. Jane Doe")
        pt_id     = st.text_input("Patient ID",            placeholder="e.g. PT-00123")
        pt_dob    = st.text_input("Date of Birth",         placeholder="YYYY-MM-DD")
    with pi_col2:
        pt_date   = st.text_input("Study Date",            placeholder="YYYY-MM-DD")
        pt_ref    = st.text_input("Referring Physician",   placeholder="e.g. Dr. Smith")
        pt_notes  = st.text_area("Clinical Notes",         placeholder="Symptoms, history …", height=68)

patient_info = {
    "name":                 pt_name,
    "patient_id":           pt_id,
    "dob":                  pt_dob,
    "study_date":           pt_date,
    "referring_physician":  pt_ref,
    "notes":                pt_notes,
}

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
            # ----------------------------------------------------------------
            # Try weighted-ensemble engine first; fall back to single-model
            # ----------------------------------------------------------------
            result = None
            inference_mode = "ensemble"

            with st.spinner("Loading inference models (weighted ensemble)…"):
                ensemble_engine = get_ensemble_engine()

            if ensemble_engine is not None:
                with st.spinner("Analyzing X-ray with weighted ensemble…"):
                    result = ensemble_engine.predict(str(temp_path), return_gradcam=True)
            else:
                inference_mode = "single"
                with st.spinner("Loading single-model inference engine…"):
                    version_id = selected_version if selected_version != "Latest Model" else None
                    inference = get_inference_engine(_version_id=version_id)
                if inference is None:
                    display_error_message("Failed to load inference engine")
                else:
                    with st.spinner("Analyzing X-ray image…"):
                        result = inference.predict(str(temp_path), return_explanations=True)

            if result is not None:
                # Log access to ledger
                ledger = get_ledger()
                ledger.log_access(
                    user_id="webapp_user",
                    action="predict",
                    resource=uploaded_file.name,
                    status="success",
                    details=result
                )

                # Store result in session state for PDF generation
                st.session_state['last_result']    = result
                st.session_state['last_temp_path'] = str(temp_path)
                st.session_state['last_patient_info'] = patient_info
                st.session_state['last_selected_version'] = selected_version

                # ── Diagnostic banners ──────────────────────────────────────
                if result.get("using_random_weights"):
                    st.warning(
                        "⚠️ **No trained checkpoints found** — the model is "
                        "running with **randomly initialised weights**. "
                        "The confidence score and predictions are **not meaningful**. "
                        "Train the model first (e.g. `python src/train_multimodel.py`) "
                        "and place the checkpoint files under "
                        "`outputs/checkpoints/<backbone>_best.pth`."
                    )

                rag_src = result.get("rag_source", "template")
                if rag_src == "template":
                    st.info(
                        "ℹ️ **RAG explanation source:** static radiology template. "
                        "Set the `GEMINI_API_KEY` environment variable to enable "
                        "Gemini-generated explanations."
                    )
                elif rag_src == "fallback":
                    st.info(
                        "ℹ️ **RAG explanation source:** Gemini fallback (API call failed). "
                        "Check that `GEMINI_API_KEY` is valid and try again."
                    )

                # ────────────────────────────────────────────────────────────
                # Display results
                # ────────────────────────────────────────────────────────────
                st.markdown("---")

                prediction = result['prediction']
                confidence = float(result.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))

                # Prediction header
                st.markdown(f"### 🩺 Prediction: **{prediction.upper()}**")
                st.markdown(f"**Class Probability:** {confidence:.1%}")
                st.progress(confidence)

                # Uncertainty / Review flag
                if result.get("needs_review", False):
                    st.warning(
                        f"⚠️ **NEEDS RADIOLOGIST REVIEW** — {result.get('review_reason', '')}"
                    )

                st.markdown("---")

                # Probabilities
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

                # Per-Hospital Predictions (ensemble mode only)
                per_hospital = result.get("per_hospital")
                if per_hospital:
                    st.markdown("### 🏥 Per-Hospital Predictions")
                    ph_cols = st.columns(len(per_hospital))
                    for idx, (hosp, info) in enumerate(per_hospital.items()):
                        with ph_cols[idx]:
                            st.markdown(f"**{hosp}**")
                            st.markdown(
                                f"*{info.get('backbone','?')}* · weight {info.get('weight',0):.0%}"
                            )
                            st.metric(
                                info.get("prediction", "?").replace("_", " ").title(),
                                f"{info.get('confidence', 0):.1%}",
                            )

                    st.markdown("---")

                # RAG Explanation
                rag_src_display = {
                    "gemini": "🤖 Gemini AI",
                    "gemini-1.5-flash": "🤖 Gemini AI",
                    "fallback": "📋 Fallback (Gemini unavailable)",
                    "template": "📄 Static Template",
                }.get(rag_src, f"📄 {rag_src}")
                st.markdown(f"### 🧠 RAG Explanation  <small style='font-weight:normal;color:grey'>— source: {rag_src_display}</small>", unsafe_allow_html=True)
                explanation_text = result.get("rag_explanation") or result.get("explanation_text") or result.get("guidelines", None)
                if isinstance(explanation_text, str) and explanation_text:
                    st.info(explanation_text)
                elif isinstance(explanation_text, list) and explanation_text:
                    for g in explanation_text:
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

                # Grad-CAM Visualisations
                gradcam_images = result.get("gradcam_images", {})
                valid_gcams = {k: v for k, v in gradcam_images.items() if v is not None}
                if valid_gcams:
                    st.markdown("### 🗺️ Grad-CAM Activation Maps")
                    gcam_cols = st.columns(len(valid_gcams))
                    hospital_labels = {
                        "resnet18": "Hospital_A",
                        "densenet121": "Hospital_B",
                        "efficientnet_b0": "Hospital_C",
                    }
                    for idx, (backbone, pil_img) in enumerate(valid_gcams.items()):
                        with gcam_cols[idx]:
                            st.image(
                                pil_img,
                                caption=f"{hospital_labels.get(backbone, backbone)} ({backbone})",
                                use_column_width=True,
                            )
                    st.markdown("---")

                # Technical Details
                with st.expander("🔧 Technical Details"):
                    st.markdown(f"""
                    - **Model Type**: {result['model_type']}
                    - **Inference Time**: {result['inference_time']:.3f} seconds
                    - **Backbones**: {', '.join(result.get('ensemble_weights', {}).keys()) or 'N/A'}
                    - **Model Version**: {selected_version}
                    - **Image**: {uploaded_file.name}
                    """)

                # ---- Clinician PDF Report download ----
                st.markdown("---")
                st.markdown("### 📄 Clinician Report")

                # Gather version info for audit section
                _version_info_dict = {}
                if selected_version and selected_version != "Latest Model":
                    _vi = registry.get_version(selected_version)
                    if _vi:
                        _version_info_dict = {
                            "version_id": _vi.version_id,
                            "model_hash": _vi.model_hash,
                            "timestamp":  _vi.timestamp,
                        }

                try:
                    pdf_bytes = generate_clinician_pdf(
                        result=result,
                        patient_info=patient_info,
                        image_path=str(temp_path),
                        model_version_info=_version_info_dict,
                    )
                    _base_name = (
                        uploaded_file.name.rsplit(".", 1)[0]
                        if "." in uploaded_file.name
                        else uploaded_file.name
                    )
                    st.download_button(
                        label="📥 Download Clinician Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"medrag_report_{_base_name}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )
                except ImportError:
                    st.warning(
                        "⚠️ PDF generation requires **reportlab**. "
                        "Install it with: `pip install reportlab`"
                    )
                except Exception as _pdf_err:
                    st.warning(f"⚠️ Could not generate PDF: {_pdf_err}")

                # ---- Provenance / JSON export gate ----
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
                        "per_hospital": result.get('per_hospital'),
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
                        "🔒 JSON provenance export locked – anchor provenance on-chain (Step 3) to enable."
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

# ============================================================================
# Step 3: Cryptographic Provenance Anchoring
# ============================================================================
# This section appears whenever inference has been run (result stored in
# session state) and lets the user anchor the provenance bundle on-chain,
# setting prov_verified=True which unlocks the JSON export.

_stored_result = st.session_state.get('last_result')

if _stored_result is not None:
    st.markdown("---")
    st.markdown("## 🔐 Step 3: Cryptographic Provenance Anchoring")
    st.markdown(
        "Anchor a tamper-proof SHA-256 hash of the inference provenance bundle "
        "on-chain (Ganache). Only hashes are stored – no patient data or raw text "
        "is sent to the blockchain. Completing this step unlocks the JSON export."
    )

    if st.session_state.get('prov_verified', False):
        _prov_tx = st.session_state.get('prov_tx_hash', 'N/A')
        st.success(
            f"✅ **Provenance anchored and verified!**  \n"
            f"Transaction hash: `{_prov_tx}`  \n"
            "🔓 Export is unlocked."
        )
        _anchor_details = st.session_state.get('prov_anchor_info') or {}
        if _anchor_details:
            with st.expander("🔍 On-chain anchor details"):
                st.json({k: str(v) for k, v in _anchor_details.items()})

        # Standalone download (available immediately after anchoring)
        import json as _json
        _export_data = _json.dumps({
            "prediction": _stored_result.get('prediction'),
            "confidence": _stored_result.get('confidence'),
            "probabilities": _stored_result.get('probabilities'),
            "per_hospital": _stored_result.get('per_hospital'),
            "provenance": st.session_state.get('prov_bundle'),
            "tx_hash": st.session_state.get('prov_tx_hash'),
            "anchor_info": st.session_state.get('prov_anchor_info'),
        }, indent=2, default=str)
        st.download_button(
            label="⬇️ Download Result & Provenance (JSON)",
            data=_export_data,
            file_name="medrag_result_provenance.json",
            mime="application/json",
            use_container_width=True,
        )

    else:
        _rpc_default = os.environ.get("GANACHE_URL", "http://127.0.0.1:7545")
        _pk_env      = os.environ.get("BLOCKCHAIN_PRIVATE_KEY", "")

        _prov_col1, _prov_col2 = st.columns(2)
        with _prov_col1:
            _prov_rpc = st.text_input(
                "Ganache RPC URL",
                value=_rpc_default,
                key="prov_rpc_url",
                help="HTTP RPC endpoint of your Ganache node (e.g. http://127.0.0.1:7545).",
            )
            _prov_pk = st.text_input(
                "Private Key (0x + 64 hex chars)",
                value=_pk_env,
                type="password",
                key="prov_private_key",
                help=(
                    "Ganache account **private key** – copy from the 'Private Keys' "
                    "section of the Ganache terminal output.  "
                    "This is NOT the account address (which is only 40 hex chars)."
                ),
            )
        with _prov_col2:
            _prov_contract = st.text_input(
                "Contract Address (leave blank to auto-deploy)",
                value="",
                key="prov_contract_addr",
                help=(
                    "Address of a pre-deployed ProvenanceRegistry contract.  "
                    "Leave blank to compile ProvenanceRegistry.sol and deploy it automatically."
                ),
            )

        _prov_mock = st.checkbox(
            "🧪 Mock mode (demo / offline – no Ganache required)",
            value=False,
            key="prov_use_mock",
            help=(
                "In mock mode provenance is stored in-process only; "
                "no real blockchain transaction is submitted.  "
                "Useful for demos when Ganache is not available."
            ),
        )

        if st.button("🔗 Anchor Provenance On-Chain", type="primary", key="prov_anchor_btn"):
            _pk_val       = _prov_pk.strip()
            _rpc_val      = _prov_rpc.strip()
            _contract_val = _prov_contract.strip()

            # ── Input validation ────────────────────────────────────────────
            _err_msg = None
            if not _prov_mock:
                _pk_hex = _pk_val.removeprefix("0x")
                if not _pk_val:
                    _err_msg = (
                        "❌ **Private key is required** in real mode.  \n"
                        "Enter the private key from your Ganache terminal, or enable Mock mode."
                    )
                elif len(_pk_hex) != 64:
                    _err_msg = (
                        f"❌ **Invalid private key** – got **{len(_pk_hex)}** hex characters, "
                        "expected **64**.  \n"
                        "Make sure you copied the **Private Key** row from Ganache "
                        "(not the account address, which is only 40 chars).  \n"
                        "Example: `0x` followed by 64 hex digits."
                    )

            if _err_msg:
                st.error(_err_msg)
            else:
                with st.spinner("Anchoring provenance on-chain…"):
                    try:
                        import sys as _sys
                        _src_path = os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..', '..', 'src')
                        )
                        if _src_path not in _sys.path:
                            _sys.path.insert(0, _src_path)

                        from provenance import build_provenance_bundle
                        from provenance_integrator import ProvenanceIntegrator

                        # ── Build component hashes ──────────────────────────
                        _ver_id = st.session_state.get('last_selected_version', 'unknown')
                        _hashes = compute_inference_provenance_hashes(_stored_result, _ver_id)

                        # ── Build canonical provenance bundle ───────────────
                        _bundle = build_provenance_bundle(
                            knowledge_base_hash=_hashes['knowledge_base_hash'],
                            explanation_hash=_hashes['explanation_hash'],
                            retrieval_hash=_hashes['retrieval_hash'],
                            model_version_hash=_hashes['model_version_hash'],
                            prompt_hash=_hashes['prompt_hash'],
                            generation_params_hash=_hashes['generation_params_hash'],
                        )

                        # ── Create ProvenanceIntegrator ─────────────────────
                        if _prov_mock:
                            _integrator = ProvenanceIntegrator(use_mock=True)
                            _signer_addr = "0x" + "a" * 40
                        else:
                            _init_kwargs = {
                                "rpc_url": _rpc_val,
                                "private_key": _pk_val,
                            }
                            if _contract_val:
                                _init_kwargs["contract_address"] = _contract_val
                            _integrator = ProvenanceIntegrator(**_init_kwargs)
                            # Derive signer address from the private key
                            _signer_addr = _integrator.deployer_address

                        # ── Anchor on-chain ─────────────────────────────────
                        _tx_hash = _integrator.anchor_provenance(
                            bundle_hash=_bundle["bundle_hash"],
                            model_hash=_hashes["model_version_hash"],
                            kb_hash=_hashes["knowledge_base_hash"],
                            explanation_hash=_hashes["explanation_hash"],
                            signer_address=_signer_addr,
                        )

                        # ── Fetch anchor details ────────────────────────────
                        _anchor_info = _integrator.get_anchor(_bundle["bundle_hash"]) or {}

                        # ── Persist to session state ────────────────────────
                        st.session_state['prov_bundle']      = _bundle
                        st.session_state['prov_tx_hash']     = _tx_hash
                        st.session_state['prov_anchor_info'] = _anchor_info
                        st.session_state['prov_verified']    = True

                        st.success(
                            f"✅ **Provenance anchored successfully!**  \n"
                            f"Transaction hash: `{_tx_hash}`  \n"
                            f"Bundle hash: `{_bundle['bundle_hash'][:32]}…`  \n"
                            "🔓 **Export is now unlocked** – the download button has appeared above."
                        )
                        st.balloons()
                        st.rerun()

                    except ConnectionError as _ce:
                        st.error(
                            f"❌ **Cannot connect to Ganache** at `{_rpc_val}`:  \n{_ce}  \n\n"
                            "**Fixes:**  \n"
                            "- Ensure Ganache is running: `ganache --port 7545`  \n"
                            "- Check the RPC URL above matches the Ganache port  \n"
                            "- Or enable **Mock mode** for a demo without Ganache"
                        )
                    except ImportError as _ie:
                        _missing = str(_ie)
                        _solc_hint = (
                            "  \n- For solc: `pip install py-solc-x` then "
                            "`python -c \"import solcx; solcx.install_solc('0.8.23')\"`"
                            if "solc" in _missing.lower() else ""
                        )
                        st.error(
                            f"❌ **Missing dependency**: `{_missing}`  \n\n"
                            "Install with:  \n"
                            "`pip install web3 py-solc-x eth-tester py-evm`"
                            f"{_solc_hint}"
                        )
                    except Exception as _ex:
                        import traceback as _tb
                        st.error(f"❌ **Anchoring failed**: {_ex}")
                        with st.expander("📋 Full traceback (for debugging)"):
                            st.code(_tb.format_exc())

    # ── Admin Override (emergency bypass) ──────────────────────────────────
    with st.expander("🔑 Admin Override (emergency bypass)", expanded=False):
        st.warning(
            "⚠️ **Admin Override** bypasses the provenance gate and allows export "
            "without on-chain verification.  Use only for authorised demos or testing."
        )
        _admin_toggle = st.checkbox(
            "Enable Admin Override",
            value=st.session_state.get('admin_override', False),
            key="admin_override_toggle",
        )
        if _admin_toggle != st.session_state.get('admin_override', False):
            st.session_state['admin_override'] = _admin_toggle
            st.rerun()
