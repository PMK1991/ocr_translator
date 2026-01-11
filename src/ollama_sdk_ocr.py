import os
import ollama
from ollama import Client
from io import BytesIO
from PIL import Image

class OCREngine:
    def __init__(self, api_key: str = None):
        self.model_name = "gemini-3-pro-preview"
        
        # Check for OLLAMA_API_KEY to use the Authenticated Cloud Client
        self.ollama_key =  os.getenv("OLLAMA_API_KEY")
        
        if self.ollama_key:
            self.client = Client(
                host="https://ollama.com",
                headers={'Authorization': 'Bearer ' + self.ollama_key}
            )
        else:
            # Fallback to default local client if no key is found
            # (Note: This will likely fail for gemini-cloud models with 403)
            self.client = ollama

    def extract_text(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()

        prompt = """
        You are an OCR tool.
        Look at this image.
        Extract and output ONLY the text visible in the image.
        Do not describe the image.
        If there is no text, say [NO TEXT].
        """

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }]
            )
            # Handle both object (local) and dict (some client versions) response types
            if isinstance(response, dict):
                return response['message']['content'].strip()
            return response.message.content.strip()
        except Exception as e:
            return f"Error during Authenticated Ollama Cloud OCR: {e}"
