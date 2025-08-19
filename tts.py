import streamlit as st
from utils import safe_import_tts

def speak(text):
    eng = safe_import_tts()
    if not eng:
        st.error("TTS engine not available on this system.")
        return
    eng.say(text)
    eng.runAndWait()
