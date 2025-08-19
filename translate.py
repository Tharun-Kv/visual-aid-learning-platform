import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer

# Map a few common directions to MarianMT models (expandable)
MARIAN_MODELS = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "ta"): "Helsinki-NLP/opus-mt-en-ta",
    ("ta", "en"): "Helsinki-NLP/opus-mt-ta-en",
}

@st.cache_resource
def load_translator(src="en", tgt="fr"):
    model_name = MARIAN_MODELS.get((src, tgt))
    if not model_name:
        st.warning("Language pair not configured; defaulting to en->fr")
        model_name = "Helsinki-NLP/opus-mt-en-fr"
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tok, model

def translate_text(text, src="en", tgt="fr"):
    tok, model = load_translator(src, tgt)
    batch = tok([text], return_tensors="pt", padding=True)
    with torch.no_grad():
        gen = model.generate(**batch, max_new_tokens=180)
    return tok.decode(gen[0], skip_special_tokens=True)
