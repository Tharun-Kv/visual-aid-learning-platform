import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import cv2

@st.cache_resource
def load_reader(lang_list=("en",)):
    return easyocr.Reader(lang_list, gpu=False)

def ocr_image(pil_image, languages=("en",)):
    reader = load_reader(languages)
    img = np.array(pil_image)
    result = reader.readtext(img, detail=1)
    # Combine lines sorted by top-left y
    result_sorted = sorted(result, key=lambda r: r[0][0][1])
    lines = [r[1] for r in result_sorted]
    return "\n".join(lines), result

def preprocess_for_ocr(pil_image, contrast=1.2, brightness=20, denoise=True):
    img = np.array(pil_image)
    # Apply brightness/contrast
    img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
    # Optional denoise
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    # Convert back to PIL
    return Image.fromarray(img)
