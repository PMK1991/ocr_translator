import ollama
from io import BytesIO
from PIL import Image

class OCREngine:
    def __init__(self, api_key: str = None):
        # User confirmed 'deepseek-ocr:latest' is pulled.
        self.model_name = "deepseek-ocr:latest"

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text using DeepSeek Vision (local).
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        prompt = "OCR: Extract all text from this image."

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error during DeepSeek OCR (ensure '{self.model_name}' is pulled): {e}"
