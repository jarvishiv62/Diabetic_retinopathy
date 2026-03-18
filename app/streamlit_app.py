"""
app/streamlit_app.py
---------------------
Streamlit web application for Diabetic Retinopathy Classification.

Features:
  - Upload a retinal fundus image (JPG/PNG)
  - Display predicted DR stage with clinical description
  - Show confidence scores for all 5 classes (bar chart)
  - Grad-CAM visualization of model attention regions
  - Clinical recommendation based on predicted grade

Usage:
    streamlit run app/streamlit_app.py -- --checkpoint outputs/checkpoints/best_fold1.pth
    streamlit run app/streamlit_app.py  (will prompt for checkpoint path in UI)
"""

import sys
import io
import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_model
from src.dataset import get_val_transforms, CLASS_NAMES, IMAGE_SIZE
from src.gradcam import GradCAM, overlay_heatmap
from src.utils import load_checkpoint, get_device


# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="DR Classifier | DRNet",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────
# Clinical Descriptions
# ─────────────────────────────────────────────

CLASS_DESCRIPTIONS = {
    0: {
        "name": "No DR",
        "description": "No signs of diabetic retinopathy detected.",
        "recommendation": "Continue regular annual eye exams.",
        "color": "#2ecc71",
        "urgency": "Routine",
    },
    1: {
        "name": "Mild NPDR",
        "description": (
            "Microaneurysms only — small balloon-like bulges in the retina's "
            "blood vessels. Early stage non-proliferative DR."
        ),
        "recommendation": "Rescreen in 9–12 months. Optimize glycemic and blood pressure control.",
        "color": "#f1c40f",
        "urgency": "Monitor",
    },
    2: {
        "name": "Moderate NPDR",
        "description": (
            "More than just microaneurysms but less severe than severe NPDR. "
            "May include dot/blot hemorrhages, hard exudates."
        ),
        "recommendation": "Refer to ophthalmologist. Rescreen in 6 months.",
        "color": "#e67e22",
        "urgency": "Refer",
    },
    3: {
        "name": "Severe NPDR",
        "description": (
            "Extensive retinal hemorrhages in all 4 quadrants, venous beading, "
            "or intraretinal microvascular abnormalities (IRMA). High risk of progression."
        ),
        "recommendation": "⚠ Urgent ophthalmology referral required. Treat within 2–4 weeks.",
        "color": "#e74c3c",
        "urgency": "Urgent",
    },
    4: {
        "name": "Proliferative DR",
        "description": (
            "New blood vessel formation (neovascularization) on the retina or optic disc. "
            "High risk of blindness without immediate treatment."
        ),
        "recommendation": "🚨 EMERGENCY referral. Laser photocoagulation or anti-VEGF treatment needed immediately.",
        "color": "#8e44ad",
        "urgency": "Emergency",
    },
}


# ─────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading DRNet model...")
def load_model(checkpoint_path: str):
    """
    Load DRNet model from checkpoint. Cached after first load so
    subsequent inference calls don't reload weights from disk.
    """
    device = get_device()
    model  = build_model().to(device)

    try:
        ckpt = load_checkpoint(model, checkpoint_path, device=device)
        model.eval()
        return model, device, ckpt
    except Exception as e:
        st.error(f"Failed to load checkpoint: {e}")
        return None, device, None


# ─────────────────────────────────────────────
# Inference Function
# ─────────────────────────────────────────────

def predict(
    model: torch.nn.Module,
    pil_image: Image.Image,
    device: torch.device,
    image_size: int = IMAGE_SIZE,
):
    """
    Run inference on a PIL image and return prediction + probabilities.

    Returns:
        (predicted_class, probabilities_array)
    """
    transform = get_val_transforms(image_size)
    tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    predicted_class = probs.argmax()
    return int(predicted_class), probs


# ─────────────────────────────────────────────
# Grad-CAM Generation
# ─────────────────────────────────────────────

def generate_gradcam(
    model: torch.nn.Module,
    pil_image: Image.Image,
    device: torch.device,
    target_class: int,
    image_size: int = IMAGE_SIZE,
):
    """
    Generate Grad-CAM overlay for the uploaded image.

    Returns:
        overlay_rgb : Numpy array (H, W, 3) with heatmap overlaid
        cam         : Raw CAM (H, W), values in [0, 1]
    """
    transform   = get_val_transforms(image_size)
    tensor      = transform(pil_image).unsqueeze(0).to(device)
    original_np = np.array(pil_image.resize((image_size, image_size)))

    gradcam_gen   = GradCAM(model)
    cam, _, _     = gradcam_gen.generate(tensor, target_class=target_class)
    overlay_rgb   = overlay_heatmap(original_np, cam, alpha=0.45)

    return overlay_rgb, cam


# ─────────────────────────────────────────────
# Probability Bar Chart
# ─────────────────────────────────────────────

def plot_probability_chart(probs: np.ndarray, predicted_class: int) -> plt.Figure:
    """Horizontal bar chart of class probabilities."""
    class_names = [CLASS_NAMES[i] for i in range(len(probs))]
    colors = [
        CLASS_DESCRIPTIONS[i]["color"] if i == predicted_class else "#bdc3c7"
        for i in range(len(probs))
    ]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(class_names, probs * 100, color=colors,
                   edgecolor="#555", linewidth=0.5, height=0.6)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Model Confidence per DR Grade", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    for bar, prob in zip(bars, probs):
        ax.text(
            min(prob * 100 + 1.5, 100), bar.get_y() + bar.get_height() / 2,
            f"{prob * 100:.1f}%", va="center", ha="left", fontsize=9, fontweight="bold"
        )

    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar(default_checkpoint: str = "") -> tuple:
    """Render sidebar controls. Returns (checkpoint_path, show_gradcam, target_class)."""
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24701-nature-natural-beauty.jpg/1200px-24701-nature-natural-beauty.jpg",
        use_column_width=True,
        caption="Retinal Fundus Imaging",
    ) if False else None  # Placeholder — remove in production

    st.sidebar.title("⚙️ Configuration")

    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value=default_checkpoint or "outputs/checkpoints/best_fold1.pth",
        help="Path to a trained DRNet .pth checkpoint file",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Grad-CAM Settings")

    show_gradcam = st.sidebar.checkbox(
        "Show Grad-CAM Explanation", value=True,
        help="Visualize which retinal regions influenced the prediction"
    )

    target_class = st.sidebar.selectbox(
        "Explain for Class",
        options=list(range(5)),
        format_func=lambda x: f"{x} — {CLASS_NAMES[x]}",
        index=0,
        help="Override the class to explain (default: predicted class)",
    )
    use_predicted = st.sidebar.checkbox(
        "Use predicted class for Grad-CAM", value=True
    )
    if use_predicted:
        target_class = None

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**About DRNet**\n"
        "Custom CNN with SE-Block attention, trained with Focal Loss "
        "using K-Fold Cross Validation on 400 fundus images.\n\n"
        "*For research use only. Not a clinical diagnostic tool.*"
    )

    return checkpoint_path, show_gradcam, target_class


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

def main(default_checkpoint: str = ""):
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("👁️ Diabetic Retinopathy Classifier")
    st.markdown(
        "Upload a **retinal fundus image** to detect the severity of "
        "Diabetic Retinopathy using DRNet — a custom CNN with SE-Block attention."
    )
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    checkpoint_path, show_gradcam, target_class_override = render_sidebar(default_checkpoint)

    # ── Load Model ────────────────────────────────────────────────────────────
    if not checkpoint_path or not Path(checkpoint_path).exists():
        st.warning(
            f"⚠️ Checkpoint not found at `{checkpoint_path}`. "
            "Please update the path in the sidebar after training."
        )
        st.stop()

    model, device, ckpt = load_model(checkpoint_path)
    if model is None:
        st.stop()

    st.sidebar.success(
        f"✅ Model loaded\n"
        f"Fold {ckpt.get('fold', '?')} | Epoch {ckpt.get('epoch', '?')}"
    )

    # ── Image Upload ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a fundus image",
        type=["jpg", "jpeg", "png"],
        help="Upload a retinal fundus photograph (JPG or PNG)",
    )

    if uploaded_file is None:
        st.info("👆 Upload a retinal fundus image above to begin analysis.")
        return

    # ── Load Image ────────────────────────────────────────────────────────────
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not read image: {e}")
        return

    # ── Run Inference ─────────────────────────────────────────────────────────
    with st.spinner("Analyzing image..."):
        predicted_class, probs = predict(model, pil_image, device)

    effective_target = target_class_override if target_class_override is not None else predicted_class

    # ── Layout: 3 columns ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1.2, 1.3])

    with col1:
        st.subheader("📷 Input Image")
        st.image(pil_image, caption=uploaded_file.name, use_column_width=True)

    with col2:
        st.subheader("🔬 Prediction")

        info = CLASS_DESCRIPTIONS[predicted_class]
        confidence = probs[predicted_class] * 100

        # Urgency badge
        urgency_colors = {
            "Routine":   "🟢",
            "Monitor":   "🟡",
            "Refer":     "🟠",
            "Urgent":    "🔴",
            "Emergency": "🟣",
        }
        badge = urgency_colors.get(info["urgency"], "⚪")

        st.markdown(
            f"""
            <div style="
                background-color:{info['color']}22;
                border-left: 6px solid {info['color']};
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 12px;
            ">
                <h3 style="color:{info['color']}; margin:0;">{badge} {info['name']}</h3>
                <p style="font-size:14px; margin:6px 0;">{info['description']}</p>
                <p style="font-size:13px; color:#555;"><b>Confidence:</b> {confidence:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clinical recommendation
        st.markdown(f"**📋 Recommendation:** {info['recommendation']}")

        # Probability chart
        fig = plot_probability_chart(probs, predicted_class)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col3:
        if show_gradcam:
            st.subheader("🗺️ Grad-CAM Explanation")
            st.caption(
                f"Highlighting retinal regions influencing "
                f"**{CLASS_NAMES[effective_target]}** prediction"
            )

            with st.spinner("Generating Grad-CAM..."):
                try:
                    overlay_rgb, cam = generate_gradcam(
                        model, pil_image, device, effective_target
                    )
                    st.image(overlay_rgb, caption="Grad-CAM Overlay", use_column_width=True)
                    st.caption(
                        "🔴 Red = high attention | 🔵 Blue = low attention\n"
                        "Warm regions are where the model focused most."
                    )
                except Exception as e:
                    st.warning(f"Grad-CAM generation failed: {e}")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "> ⚠️ **Disclaimer:** This tool is for research and educational purposes only. "
        "It is **not** a certified medical device and should **not** be used as a substitute "
        "for professional ophthalmic examination."
    )


# ─────────────────────────────────────────────
# CLI Entry
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Streamlit doesn't support argparse directly; read sys.argv manually
    default_ckpt = ""
    if "--checkpoint" in sys.argv:
        idx = sys.argv.index("--checkpoint")
        if idx + 1 < len(sys.argv):
            default_ckpt = sys.argv[idx + 1]

    main(default_checkpoint=default_ckpt)