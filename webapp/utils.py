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
from inference import (
    load_inference_model,
    load_vfl_model,
    VFLInferenceEngine,
    load_multi_model_ensemble,
    MultiModelEnsembleEngine,
)


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
    Get inference engine. Prefers a VFLFramework checkpoint from
    ``outputs/checkpoints/`` (produced by train_multimodel.py) when
    available; otherwise falls back to the legacy MedRAGInference model.

    Args:
        _version_id: Ignored (kept for backwards-compat with existing callers).

    Returns:
        VFLInferenceEngine or MedRAGInference instance.
    """
    repo_root = Path(__file__).parent.parent
    ckpt_dir = repo_root / "outputs" / "checkpoints"

    # Try VFLFramework checkpoints first (newest by mtime)
    preferred_order = ["resnet18", "densenet121", "efficientnet_b0"]
    found_ckpt = None
    found_backbone = None
    for backbone in preferred_order:
        candidate = ckpt_dir / f"{backbone}_best.pth"
        if candidate.is_file():
            found_ckpt = str(candidate)
            found_backbone = backbone
            break

    if found_ckpt and found_backbone:
        try:
            engine = load_vfl_model(
                checkpoint_path=found_ckpt,
                backbone=found_backbone,
            )
            return engine
        except Exception as e:
            st.warning(f"Could not load VFL checkpoint ({found_ckpt}): {e}. Falling back.")

    # Legacy fallback
    try:
        inference = load_inference_model(use_rag=True, num_clients=4)
        return inference
    except Exception as e:
        st.error(f"Error loading inference engine: {e}")
        return None


@st.cache_resource
def get_ensemble_engine():
    """
    Load all available VFLFramework checkpoints and return a
    MultiModelEnsembleEngine (weighted average over resnet18/densenet121/
    efficientnet_b0).  Falls back gracefully if some checkpoints are absent.
    """
    repo_root = Path(__file__).parent.parent
    ckpt_dir = repo_root / "outputs" / "checkpoints"
    try:
        return load_multi_model_ensemble(
            checkpoints_dir=str(ckpt_dir),
            class_names=["Normal", "COVID", "Lung_Opacity", "Pneumonia"],
        )
    except Exception as exc:
        st.warning(f"Could not build ensemble engine: {exc}")
        return None


def generate_clinician_pdf(
    result: dict,
    patient_info: dict = None,
    image_path: str = None,
    model_version_info: dict = None,
) -> bytes:
    """
    Generate a clinician-grade PDF report from an inference result dict.

    Args:
        result           : Dict returned by MultiModelEnsembleEngine.predict()
                           or VFLInferenceEngine.predict().
        patient_info     : Optional {name, dob, study_date, referring_physician,
                           patient_id, notes}.
        image_path       : Optional path to the original X-ray image.
        model_version_info: Optional dict with audit/registry metadata.

    Returns:
        PDF bytes suitable for ``st.download_button``.
    """
    try:
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable,
        )
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io as _io
        from datetime import datetime as _dt

        # ── helpers ──────────────────────────────────────────────────────────
        def _pil_to_rl_image(pil_img, width_cm=7.5, height_cm=7.5):
            """Convert a PIL Image to an in-memory ReportLab Image."""
            buf = _io.BytesIO()
            pil_img.save(buf, format="PNG")
            buf.seek(0)
            return RLImage(buf, width=width_cm * cm, height=height_cm * cm)

        # ── document setup ────────────────────────────────────────────────────
        buf = _io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            title="MedRAG Clinician Report",
            author="MedRAG AI System",
        )

        styles = getSampleStyleSheet()
        style_normal  = styles["Normal"]
        style_h1      = styles["Heading1"]
        style_h2      = styles["Heading2"]
        style_h3      = styles["Heading3"]

        # Custom styles
        style_banner = ParagraphStyle(
            "Banner",
            parent=style_h1,
            fontSize=16,
            textColor=colors.white,
            backColor=colors.HexColor("#2c3e50"),
            spaceAfter=4,
            spaceBefore=0,
            leftIndent=-1 * cm,
            rightIndent=-1 * cm,
            leading=22,
            alignment=TA_CENTER,
        )
        style_review_flag = ParagraphStyle(
            "ReviewFlag",
            parent=style_normal,
            fontSize=12,
            textColor=colors.white,
            backColor=colors.HexColor("#c0392b"),
            spaceAfter=6,
            spaceBefore=4,
            leading=16,
            alignment=TA_CENTER,
        )
        style_impression = ParagraphStyle(
            "Impression",
            parent=style_normal,
            fontSize=13,
            textColor=colors.HexColor("#1a5276"),
            spaceAfter=4,
            spaceBefore=4,
            leading=18,
        )
        style_caption = ParagraphStyle(
            "Caption",
            parent=style_normal,
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER,
            spaceAfter=2,
        )

        story = []
        hr = HRFlowable(
            width="100%", thickness=1, color=colors.HexColor("#bdc3c7"),
            spaceAfter=6, spaceBefore=6,
        )

        # ── Banner ────────────────────────────────────────────────────────────
        story.append(Paragraph("🏥 MedRAG AI Radiology Report", style_banner))
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            f"Generated: {_dt.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  •  "
            f"System: MedRAG Weighted-Ensemble VFL",
            styles["Normal"]
        ))
        story.append(hr)

        # ── Patient Information ───────────────────────────────────────────────
        story.append(Paragraph("Patient Information", style_h2))
        pi = patient_info or {}
        pi_rows = [
            ["Patient Name", pi.get("name", "—")],
            ["Patient ID",   pi.get("patient_id", "—")],
            ["Date of Birth", pi.get("dob", "—")],
            ["Study Date",   pi.get("study_date", _dt.now().strftime("%Y-%m-%d"))],
            ["Referring Physician", pi.get("referring_physician", "—")],
            ["Notes",        pi.get("notes", "—")],
        ]
        pi_table = Table(pi_rows, colWidths=[5 * cm, 12 * cm])
        pi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf4fb")),
            ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7b8")),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(pi_table)
        story.append(hr)

        # ── Primary Impression ────────────────────────────────────────────────
        story.append(Paragraph("Primary Impression", style_h2))
        pred = result.get("prediction", "Unknown")
        conf = result.get("confidence", 0.0)
        story.append(Paragraph(
            f"<b>{pred.upper()}</b> — Confidence: <b>{conf:.1%}</b>",
            style_impression
        ))

        # Uncertainty / review flag
        if result.get("needs_review", False):
            reason = result.get("review_reason", "")
            story.append(Paragraph(
                f"⚠ NEEDS RADIOLOGIST REVIEW — {reason}",
                style_review_flag,
            ))
        story.append(hr)

        # ── Differential Diagnosis (top-3) ────────────────────────────────────
        story.append(Paragraph("Differential Diagnosis", style_h2))
        probs = result.get("probabilities", {})
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        dd_data = [["Rank", "Diagnosis", "Probability"]] + [
            [str(i + 1), cls.replace("_", " ").title(), f"{p:.2%}"]
            for i, (cls, p) in enumerate(sorted_probs)
        ]
        dd_table = Table(dd_data, colWidths=[2 * cm, 9 * cm, 6 * cm])
        dd_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7b8")),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(dd_table)
        story.append(hr)

        # ── Per-Hospital Predictions ──────────────────────────────────────────
        per_hospital = result.get("per_hospital")
        if per_hospital:
            story.append(Paragraph("Per-Hospital Predictions (Virtual Federated Network)", style_h2))
            ph_header = ["Hospital", "Backbone", "Prediction", "Confidence", "Weight"]
            ph_rows = [ph_header]
            for hosp, info in per_hospital.items():
                ph_rows.append([
                    hosp,
                    info.get("backbone", "—"),
                    info.get("prediction", "—").replace("_", " ").title(),
                    f"{info.get('confidence', 0):.1%}",
                    f"{info.get('weight', 0):.0%}",
                ])
            ph_table = Table(ph_rows, colWidths=[4 * cm, 4 * cm, 4 * cm, 3 * cm, 2 * cm])
            ph_table.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1a5276")),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 0), (-1, -1), 9),
                ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eaf4fb")]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7b8")),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(ph_table)
            story.append(hr)

        # ── Agreement / Uncertainty Analysis ──────────────────────────────────
        story.append(Paragraph("Agreement & Uncertainty Analysis", style_h2))
        needs_review = result.get("needs_review", False)
        review_reason = result.get("review_reason", "")
        num_models = result.get("num_models", 1)
        if per_hospital:
            top_preds = [v.get("prediction", "") for v in per_hospital.values()]
            unique_top = set(top_preds)
            agreement_text = (
                f"All {num_models} virtual hospitals agree on <b>{pred}</b>."
                if len(unique_top) == 1 else
                f"Hospitals predict different classes: "
                + ", ".join(
                    f"{h}→<b>{v.get('prediction','?')}</b>"
                    for h, v in per_hospital.items()
                )
            )
        else:
            agreement_text = "Single-model inference — no cross-hospital comparison available."

        story.append(Paragraph(agreement_text, style_normal))
        if needs_review:
            story.append(Paragraph(
                f"<b>Action Required:</b> {review_reason}",
                ParagraphStyle("ReviewText", parent=style_normal,
                               textColor=colors.HexColor("#c0392b"))
            ))
        else:
            story.append(Paragraph(
                "✓ Confidence and agreement are within acceptable thresholds.",
                ParagraphStyle("OKText", parent=style_normal,
                               textColor=colors.HexColor("#1e8449"))
            ))
        story.append(hr)

        # ── RAG Explanation ───────────────────────────────────────────────────
        story.append(Paragraph("RAG Explanation & Clinical Guidance", style_h2))
        exp_text = (
            result.get("explanation_text")
            or result.get("rag_explanation", "")
            or "No AI explanation available."
        )
        story.append(Paragraph(exp_text, style_normal))
        story.append(Spacer(1, 0.3 * cm))

        citations = result.get("citations", [])
        if citations:
            story.append(Paragraph("Retrieved Guideline Citations:", style_h3))
            cit_data = [["#", "Source", "Snippet"]]
            _MAX_SNIPPET_LEN = 120
            for idx, cit in enumerate(citations, 1):
                snippet = cit.get("snippet", "—")
                cit_data.append([
                    str(idx),
                    cit.get("source", "—"),
                    snippet[:_MAX_SNIPPET_LEN] + ("…" if len(snippet) > _MAX_SNIPPET_LEN else ""),
                ])
            cit_table = Table(
                cit_data,
                colWidths=[0.8 * cm, 6 * cm, 10.2 * cm],
                repeatRows=1,
            )
            cit_table.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1b4f72")),
                ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE",    (0, 0), (-1, -1), 8),
                ("ALIGN",       (0, 0), (0, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#d6eaf8")]),
                ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7b8")),
                ("VALIGN",      (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",  (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("WORDWRAP",    (1, 0), (-1, -1), True),
            ]))
            story.append(cit_table)
        story.append(hr)

        # ── Grad-CAM Visuals ──────────────────────────────────────────────────
        gradcam_images = result.get("gradcam_images", {})
        valid_gcams = {k: v for k, v in gradcam_images.items() if v is not None}
        if valid_gcams:
            story.append(Paragraph("Grad-CAM Activation Maps", style_h2))
            story.append(Paragraph(
                "Heat maps show regions that most influenced each backbone's prediction. "
                "Warmer colours (red/yellow) indicate stronger activation.",
                style_normal,
            ))
            story.append(Spacer(1, 0.3 * cm))

            gcam_row = []
            for backbone, pil_img in valid_gcams.items():
                try:
                    rl_img = _pil_to_rl_image(pil_img, width_cm=5.5, height_cm=5.5)
                    hospital = {"resnet18": "Hospital_A", "densenet121": "Hospital_B",
                                "efficientnet_b0": "Hospital_C"}.get(backbone, backbone)
                    gcam_row.append([
                        rl_img,
                        Paragraph(
                            f"<b>{hospital}</b><br/>{backbone}",
                            style_caption,
                        ),
                    ])
                except Exception:
                    pass

            if gcam_row:
                # Lay out max 3 images side-by-side
                inner_cols = min(3, len(gcam_row))
                row_imgs   = [[g[0] for g in gcam_row[:inner_cols]]]
                row_labels = [[g[1] for g in gcam_row[:inner_cols]]]
                col_w = 6 * cm
                gcam_table = Table(
                    row_imgs + row_labels,
                    colWidths=[col_w] * inner_cols,
                )
                gcam_table.setStyle(TableStyle([
                    ("ALIGN",   (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID",    (0, 0), (-1, -1), 0.3, colors.HexColor("#bdc3c7")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                story.append(gcam_table)
            story.append(hr)
        elif image_path:
            # Embed the original X-ray if no Grad-CAM available
            try:
                from PIL import Image as _PI
                orig = _PI.open(image_path).convert("RGB")
                orig.thumbnail((400, 400))
                story.append(Paragraph("Original X-Ray Image", style_h2))
                story.append(_pil_to_rl_image(orig, width_cm=7, height_cm=7))
                story.append(hr)
            except Exception:
                pass

        # ── Audit Metadata ────────────────────────────────────────────────────
        story.append(Paragraph("Audit Metadata", style_h2))
        mvi = model_version_info or {}
        _HASH_DISPLAY_LEN = 32
        _model_hash_raw = mvi.get("model_hash", mvi.get("sha256", "N/A"))
        _model_hash_display = (
            _model_hash_raw[:_HASH_DISPLAY_LEN] + "…"
            if len(_model_hash_raw) > _HASH_DISPLAY_LEN
            else _model_hash_raw
        )
        audit_rows = [
            ["Field", "Value"],
            ["Model Type",       result.get("model_type", "VFL Ensemble")],
            ["Inference Time",   f"{result.get('inference_time', 0):.3f} s"],
            ["Backbones Used",   ", ".join(result.get("ensemble_weights", {}).keys()) or "N/A"],
            ["Version ID",       mvi.get("version_id", "N/A")],
            ["SHA-256 Hash",     _model_hash_display],
            ["Ledger Timestamp", mvi.get("timestamp", "N/A")],
            ["Report Generated", _dt.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]
        audit_table = Table(audit_rows, colWidths=[5 * cm, 12 * cm])
        audit_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND",  (0, 1), (0, -1), colors.HexColor("#eaecee")),
            ("FONTNAME",    (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",    (1, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#aab7b8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9f9")]),
            ("TOPPADDING",  (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(audit_table)
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "<i>This report was generated by an AI system and is intended to assist "
            "qualified radiologists. It does not replace professional clinical judgment. "
            "Always correlate with patient history and other investigations.</i>",
            ParagraphStyle("Disclaimer", parent=style_normal, fontSize=8,
                           textColor=colors.gray),
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )



def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp for display."""
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(timestamp_str)


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
