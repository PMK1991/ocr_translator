import os
import base64
from io import BytesIO
from PIL import Image
from openai import AzureOpenAI

# Configuration Constants
DEFAULT_API_VERSION = "2024-02-15-preview"
DEFAULT_DEPLOYMENT = "gpt-4o-mini"

class OCREngine:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", DEFAULT_DEPLOYMENT) 
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)

        if not self.api_key or not self.endpoint:
            raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set.")

        self.client = AzureOpenAI(
            api_key=self.api_key,  
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

    def encode_image(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def extract_text(self, image: Image.Image) -> str:
        """
        Extracts text using Azure OpenAI (GPT-4o) Vision capabilities.
        """
        base64_image = self.encode_image(image)
        
        prompt = """
        You are an advanced OCR engine.
        Transcribe the text in this image EXACTLY as it appears.
        Do not describe the image. Do not add conversational text.
        If there is no text, say [NO TEXT].
        """

        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=4096
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during Azure GPT-4o OCR: {e}"
