import streamlit as st
from PIL import Image
import json
from pathlib import Path

from preprocess import preprocess
from pipeline import run_ocr, draw_results, results_to_text
from trainer import OCRCorrectionModel, align_words, save_model, load_model, PICKLE_PATH

SAMPLE_DIR = Path(__file__).parent / "ocr"

st.set_page_config(page_title="Local OCR", layout="wide")
st.title("Local Image Text Identifier")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
    st.subheader("Preprocessing")
    apply_denoise = st.checkbox("Denoise", value=True)
    apply_deskew = st.checkbox("Deskew", value=False)
    apply_sharpen = st.checkbox("Sharpen", value=False)
    st.caption("Green = high confidence · Orange = medium · Red = low")

    st.divider()
    st.subheader("Correction Model")
    correction_model = load_model()
    if correction_model:
        st.success(f"Model loaded — {correction_model.correction_count} corrections")
        uploaded_pkl = st.file_uploader("Replace model (.pkl)", type=["pkl"], key="sidebar_pkl")
        if uploaded_pkl:
            import pickle
            correction_model = pickle.loads(uploaded_pkl.read())
            save_model(correction_model)
            st.success("Model replaced.")
            st.rerun()
    else:
        st.info("No correction model — train one in the Train tab")
        uploaded_pkl = st.file_uploader("Load existing model (.pkl)", type=["pkl"], key="sidebar_pkl")
        if uploaded_pkl:
            import pickle
            correction_model = pickle.loads(uploaded_pkl.read())
            save_model(correction_model)
            st.success("Model loaded.")
            st.rerun()

tab_ocr, tab_train = st.tabs(["OCR", "Train Correction Model"])

# ─── OCR Tab ────────────────────────────────────────────────────────────────
with tab_ocr:
    source = st.radio("Image source", ["Upload image", "Use sample images"], horizontal=True)

    images: dict[str, Image.Image] = {}

    if source == "Upload image":
        uploaded = st.file_uploader(
            "Upload one or more images",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            accept_multiple_files=True,
        )
        for f in uploaded:
            images[f.name] = Image.open(f)
    else:
        samples = sorted(SAMPLE_DIR.glob("*.png")) + sorted(SAMPLE_DIR.glob("*.jpg"))
        selected = st.multiselect(
            "Select sample images",
            [p.name for p in samples],
            default=[p.name for p in samples],
        )
        for name in selected:
            images[name] = Image.open(SAMPLE_DIR / name)

    if not images:
        st.info("Select or upload an image to get started.")
    else:
        for name, img in images.items():
            st.divider()
            st.subheader(name)

            with st.spinner(f"Running OCR on {name}..."):
                processed = preprocess(
                    img,
                    apply_denoise=apply_denoise,
                    apply_deskew=apply_deskew,
                    apply_sharpen=apply_sharpen,
                )
                regions = run_ocr(processed)

            if correction_model:
                for r in regions:
                    r.text = correction_model.correct_text(r.text)

            annotated = draw_results(processed, regions, conf_threshold=conf_threshold)

            col1, col2 = st.columns(2)
            with col1:
                st.caption("Annotated image")
                st.image(annotated, use_container_width=True)
            with col2:
                st.caption("Extracted text")
                text_out = results_to_text(regions, conf_threshold=conf_threshold)
                st.text_area("", value=text_out, height=300, key=f"text_{name}")

            with st.expander("Detection details"):
                rows = [
                    {"text": r.text, "confidence": f"{r.confidence:.2%}", "bbox": r.bbox}
                    for r in regions
                ]
                st.dataframe(rows, use_container_width=True)

            ecol1, ecol2 = st.columns(2)
            with ecol1:
                st.download_button(
                    "Download as .txt",
                    data=text_out,
                    file_name=f"{Path(name).stem}_ocr.txt",
                    mime="text/plain",
                    key=f"dl_txt_{name}",
                )
            with ecol2:
                json_out = json.dumps(
                    [{"text": r.text, "confidence": r.confidence, "bbox": r.bbox} for r in regions],
                    indent=2,
                )
                st.download_button(
                    "Download as .json",
                    data=json_out,
                    file_name=f"{Path(name).stem}_ocr.json",
                    mime="application/json",
                    key=f"dl_json_{name}",
                )

# ─── Train Tab ──────────────────────────────────────────────────────────────
with tab_train:
    st.write(
        "Upload images and provide correct ground-truth text. "
        "The app learns word-level corrections and saves them as a pickle model "
        "that is automatically applied during OCR."
    )

    if "training_pairs" not in st.session_state:
        st.session_state.training_pairs = []

    train_files = st.file_uploader(
        "Upload training images",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        accept_multiple_files=True,
        key="train_uploader",
    )

    if train_files:
        for f in train_files:
            img = Image.open(f)
            st.divider()
            st.subheader(f.name)

            with st.spinner(f"Running OCR on {f.name}..."):
                processed = preprocess(
                    img,
                    apply_denoise=apply_denoise,
                    apply_deskew=apply_deskew,
                    apply_sharpen=apply_sharpen,
                )
                regions = run_ocr(processed)

            ocr_text = results_to_text(regions)

            col1, col2 = st.columns(2)
            with col1:
                st.caption("OCR output (raw)")
                st.text_area("OCR text", value=ocr_text, height=200, key=f"ocr_{f.name}", disabled=True)
            with col2:
                st.caption("Correct text (ground truth)")
                correct_text = st.text_area(
                    "Enter correct text here",
                    value="",
                    height=200,
                    key=f"correct_{f.name}",
                )

            if st.button("Add to training data", key=f"add_{f.name}"):
                if not correct_text.strip():
                    st.warning("Enter the correct text first.")
                else:
                    pairs = align_words(ocr_text, correct_text)
                    if pairs:
                        st.session_state.training_pairs.extend(pairs)
                        st.success(
                            f"Added {len(pairs)} correction pair(s). "
                            f"Total collected: {len(st.session_state.training_pairs)}"
                        )
                    else:
                        st.info("No differences found — OCR output already matches the correct text.")

    st.divider()
    col_info, col_action = st.columns([2, 1])

    with col_info:
        st.metric("Training pairs collected", len(st.session_state.training_pairs))
        if st.session_state.training_pairs:
            with st.expander("Preview collected corrections"):
                st.dataframe(
                    [{"OCR word": ocr, "Correct word": cor} for ocr, cor in st.session_state.training_pairs],
                    use_container_width=True,
                )
        if PICKLE_PATH.exists():
            st.metric(
                "Saved model corrections",
                load_model().correction_count if load_model() else 0,
            )

    with col_action:
        if st.session_state.training_pairs:
            if st.button("Train & Save Model", type="primary", use_container_width=True):
                model_to_update = load_model() or OCRCorrectionModel()
                model_to_update.fit(st.session_state.training_pairs)
                save_model(model_to_update)
                count = len(st.session_state.training_pairs)
                st.session_state.training_pairs = []
                st.success(
                    f"Saved `{PICKLE_PATH.name}` — "
                    f"{model_to_update.correction_count} total corrections."
                )
                st.rerun()

        if PICKLE_PATH.exists():
            with open(PICKLE_PATH, "rb") as pkl_f:
                st.download_button(
                    "Download model (.pkl)",
                    data=pkl_f.read(),
                    file_name="ocr_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            if st.button("Clear saved model", type="secondary", use_container_width=True):
                PICKLE_PATH.unlink()
                st.success("Model deleted.")
                st.rerun()
