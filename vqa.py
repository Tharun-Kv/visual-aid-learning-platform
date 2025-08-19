import streamlit as st
from PIL import Image
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering

@st.cache_resource
def load_vqa(model_name="dandelin/vilt-b32-finetuned-vqa"):
    processor = ViltProcessor.from_pretrained(model_name)
    model = ViltForQuestionAnswering.from_pretrained(model_name)
    return processor, model

def answer_question(pil_image, question: str):
    processor, model = load_vqa()
    enc = processor(pil_image, question, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]
