import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

class OCREngine:
    def __init__(self, api_key: str = "ollama"):
        # Ollama runs locally, usually no key needed, but client requires one.
        self.api_key = api_key
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://localhost:11434/v1"
        )
        
        # User requested specific model via Ollama
        self.model_name = os.getenv("OLLAMA_VISION_MODEL", "Gemini-3-pro-preview:cloud")

    def encode_image(self, image: Image.Image) -> str:
        # Convert to RGB to ensure compatibility
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text using a local Vision model via Ollama.
        """
        base64_image = self.encode_image(image)
        
        prompt = """
        You are an OCR tool.
        Look at this image.
        Extract and output ONLY the text visible in the image.
        Do not describe the image.
        If there is no text, say [NO TEXT].
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during Ollama OCR (ensure '{self.model_name}' is pulled): {e}"
