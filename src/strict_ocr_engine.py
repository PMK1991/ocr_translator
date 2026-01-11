import os
import google.generativeai as genai
from PIL import Image

class OCREngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        
        # User requested specific model
        self.model = genai.GenerativeModel(
            'gemini-3-pro-image-preview',
            generation_config=genai.GenerationConfig(temperature=0.0)
        )

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text from an image using Gemini PRO with strict settings.
        """
        prompt = """
        # Task
        Transcribe the text in this image.

        # Guidelines
        1. **Verbatim**: Transcribe exactly what you see. 
        2. **No Description**: Do not describe the image.
        3. **No Correction**: Do not correct spelling errors in the source.
        4. **Completeness**: Read all visible text.
        5. **Strictness**: If you cannot read a word, write [ILLEGIBLE]. Do not guess.

        Output ONLY the text.
        """
        
        try:
            response = self.model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as e:
            return f"Error during OCR: {e}"
