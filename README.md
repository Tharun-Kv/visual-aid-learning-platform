# Visual Aid Learning Platform (Streamlit)

A feature-rich, accessibility-first visual learning assistant built with Python and many popular libraries.

## Key Features
- **Image Describe**: Auto-captions images (BLIP).
- **OCR (Image & PDF)**: Extracts text using EasyOCR; optional layout detection with OpenCV.
- **Translate**: Offline translation using MarianMT (Helsinki-NLP) or NLLB (optional, heavier).
- **Summarize**: TL;DR with DistilBART or T5.
- **Text-to-Speech**: Offline TTS (pyttsx3).
- **Visual Filters**: Contrast/brightness, grayscale, edge emphasis for low-vision users.
- **VQA (Ask about an Image)**: Visual question answering (ViLT-based).
- **Alt-Text Batcher**: Generate and export alt text for multiple images.

## Quickstart
```bash
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# (Optional) Download a small spaCy model for tokenization
python -m spacy download en_core_web_sm

streamlit run app_streamlit.py
```

## Notes
- First run will download model weights (one-time) to your cache.
- If GPU is available, PyTorch will accelerate the models automatically.
- Tesseract is **not required** (we use EasyOCR).
- You can switch models in the sidebar > "Models & Settings".
