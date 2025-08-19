import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from utils import get_device

@st.cache_resource
def load_blip(model_name="Salesforce/blip-image-captioning-large"):
    device = get_device()
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    return processor, model, device

def caption_image(pil_image, max_new_tokens=40):
    processor, model, device = load_blip()
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True)
