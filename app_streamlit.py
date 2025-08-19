import io
import streamlit as st
from PIL import Image
import numpy as np
from utils import get_device
from vision import caption_image
from ocr import ocr_image, preprocess_for_ocr
from translate import translate_text
from summarize import summarize_text
from tts import speak
from vqa import answer_question
from pypdf import PdfReader
from pdf2image import convert_from_bytes

st.set_page_config(page_title="Visual Aid Learning Platform", page_icon="🎓", layout="wide")
st.title("🎓 Visual Aid Learning Platform")
st.caption("All-in-one visual assist: caption, OCR, translate, summarize, TTS, VQA, and more.")

with st.sidebar:
    st.header("Models & Settings")
    device = get_device()
    st.write(f"**Device:** {device}")
    max_caption_tokens = st.slider("Caption max tokens", 12, 80, 40)
    ocr_langs = st.multiselect("OCR languages", ["en", "hi", "ta", "te", "kn", "mr", "bn", "gu", "pa"], default=["en"])
    src_lang = st.text_input("Translate: source (ISO-2)", "en")
    tgt_lang = st.text_input("Translate: target (ISO-2)", "hi")

tabs = st.tabs(["Image Describe", "OCR (Image/PDF)", "Translate", "Summarize", "Text-to-Speech", "VQA (Ask Image)", "Alt-Text Batcher"])

# --- Image Describe ---
with tabs[0]:
    st.subheader("🖼️ Image Describe")
    up = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp","bmp"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, use_column_width=True)
        if st.button("Generate Caption"):
            with st.spinner("Captioning..."):
                cap = caption_image(img, max_caption_tokens)
            st.success(cap)
            if st.button("Speak"):
                speak(cap)

# --- OCR ---
with tabs[1]:
    st.subheader("📄 OCR (Image or PDF)")
    col1, col2 = st.columns(2)
    with col1:
        up_img = st.file_uploader("Upload image for OCR", type=["png","jpg","jpeg","webp","bmp"], key="ocr_img")
        if up_img:
            img = Image.open(up_img).convert("RGB")
            show = st.checkbox("Enhance for readability", True)
            if show:
                img = preprocess_for_ocr(img)
            st.image(img, use_column_width=True)
            if st.button("Run OCR", key="btn_ocr_img"):
                with st.spinner("Extracting..."):
                    text, _ = ocr_image(img, tuple(ocr_langs) if ocr_langs else ("en",))
                st.text_area("Extracted Text", text, height=250)
    with col2:
        up_pdf = st.file_uploader("Upload PDF for OCR", type=["pdf"], key="ocr_pdf")
        if up_pdf:
            with st.spinner("Converting pages to images..."):
                pages = convert_from_bytes(up_pdf.read(), dpi=200, fmt="png")
            st.write(f"Pages converted: {len(pages)}")
            page_index = st.number_input("Page index", min_value=0, max_value=len(pages)-1, value=0, step=1)
            page_img = pages[page_index]
            st.image(page_img, use_column_width=True)
            if st.button("OCR this page", key="btn_ocr_pdf"):
                with st.spinner("Extracting..."):
                    text, _ = ocr_image(page_img, tuple(ocr_langs) if ocr_langs else ("en",))
                st.text_area("Extracted Text", text, height=250)

# --- Translate ---
with tabs[2]:
    st.subheader("🌐 Translate (Offline)")
    text = st.text_area("Enter text to translate")
    if st.button("Translate"):
        with st.spinner("Translating..."):
            out = translate_text(text, src_lang, tgt_lang)
        st.success(out)

# --- Summarize ---
with tabs[3]:
    st.subheader("🧠 Summarize")
    text = st.text_area("Paste long text to summarize", height=220)
    max_len = st.slider("Max tokens", 32, 256, 130)
    min_len = st.slider("Min tokens", 10, 100, 30)
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            out = summarize_text(text, max_len, min_len)
        st.success(out)

# --- TTS ---
with tabs[4]:
    st.subheader("🔊 Text-to-Speech (Offline)")
    text = st.text_area("Text to speak", height=150)
    if st.button("Speak now"):
        speak(text)
        st.info("If you didn't hear anything, ensure system audio is allowed and pyttsx3 works on your OS.")

# --- VQA ---
with tabs[5]:
    st.subheader("❓ Visual Question Answering")
    up = st.file_uploader("Upload an image", type=["png","jpg","jpeg","webp","bmp"], key="vqa_img")
    q = st.text_input("Ask a question about this image", "What is in the picture?")
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, use_column_width=True)
        if st.button("Answer"):
            with st.spinner("Thinking..."):
                ans = answer_question(img, q)
            st.success(ans)

# --- Alt-Text Batcher ---
with tabs[6]:
    st.subheader("🧩 Alt-Text Batcher")
    ups = st.file_uploader("Upload multiple images", type=["png","jpg","jpeg","webp","bmp"], accept_multiple_files=True)
    if ups:
        rows = []
        for uf in ups:
            img = Image.open(uf).convert("RGB")
            cap = caption_image(img, max_caption_tokens=32)
            rows.append((uf.name, cap))
        st.write("Generated alt text:")
        for name, cap in rows:
            st.write(f"- **{name}** → {cap}")
