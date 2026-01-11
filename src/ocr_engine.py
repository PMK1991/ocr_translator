import os
import google.generativeai as genai
from PIL import Image

class OCREngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text from an image using Gemini 1.5 Flash.
        Prompts the model to transcribe exact text.
        """
        prompt = """
        You are an expert OCR engine. 
        Please transcribe the text in this image EXACTLY as it appears. 
        Do not translate it. 
        Do not describe the image. 
        Output only the text found in the image.
        If the image contains columns, transcribe them in reading order.
        """
        
        try:
            response = self.model.generate_content([prompt, image])
            return response.text
        except Exception as e:
            return f"Error during OCR: {e}"
