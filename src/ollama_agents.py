import os
from openai import OpenAI

class AgentManager:
    def __init__(self, api_key: str = "ollama"):
        self.api_key = api_key
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="http://localhost:11434/v1"
        )
        
        # User requested specific model via Ollama
        self.model_name = os.getenv("OLLAMA_TEXT_MODEL", "Gemini-3-pro-preview:cloud")

    def translate(self, text: str, source_lang_hint: str = None) -> str:
        prompt = f"""
        Translate the following extracted text to English.
        Source hint: {source_lang_hint or "Unknown"}.
        Output ONLY the translation.
        
        Text:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during Ollama Translation (ensure '{self.model_name}' is pulled): {e}"

    def refine(self, text: str) -> str:
        prompt = f"""
        Refine the following English text to be professional and clear.
        Fix grammar and OCR errors.
        Output ONLY the refined text.
        
        Text:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during Ollama Refinement: {e}"
