import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_summarizer(model_name="sshleifer/distilbart-cnn-12-6"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mdl

def summarize_text(text, max_len=130, min_len=30):
    tok, mdl = load_summarizer()
    inputs = tok([text], return_tensors="pt", truncation=True)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_len, min_length=min_len, length_penalty=1.0)
    return tok.decode(out[0], skip_special_tokens=True)
