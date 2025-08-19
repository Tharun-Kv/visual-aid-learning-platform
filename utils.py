import streamlit as st
import torch

@st.cache_resource
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def bytesio_from_uploaded(uploaded_file):
    return uploaded_file.getvalue()

def safe_import_tts():
    try:
        import pyttsx3
        eng = pyttsx3.init()
        return eng
    except Exception as e:
        return None
