import os
from google import genai
from google.genai import types
from PIL import Image

# Configuration Constants
DEFAULT_MODEL_NAME = "gemini-1.5-pro"

class OCREngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text using Google GenAI Vision.
        """
        prompt = """
        You are a highly accurate OCR engine.
        Transcribe the text in this image EXACTLY as it appears.
        Do not describe the image. Do not add conversational text.
        If the text is illegible, write [ILLEGIBLE].
        Preserve the layout structure where possible.
        """

        try:
            # The new SDK handles PIL images directly in contents
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    temperature=1.0,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE",
                        ),
                    ]
                )
            )
            return response.text.strip()
        except Exception as e:
            return f"Error during Google GenAI OCR: {e}"
