import os
import ollama

# Configuration Constants
DEFAULT_MODEL_NAME = "deepseek-r1:latest" # Defaulting to local model as safer fallback

class AgentManager:
    def __init__(self, api_key: str = None):
        # Allow override of model via env var
        self.model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL_NAME)
        
        # Initialize client (defaults to localhost:11434)
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=host)

    def process_text_single_shot(self, text: str, source_lang_hint: str = None) -> str:
        prompt = f"""
        Translate the following text to professional English and refine it for clarity.
        Source Hint: {source_lang_hint or "Unknown"}
        Output ONLY the final translation.
        
        Text:
        {text}
        """
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            # Handle potential dict or object response from SDK updates
            if isinstance(response, dict):
                return response['message']['content'].strip()
            return response.message.content.strip()
        except Exception as e:
            return f"Error during Ollama Processing: {e}"

    def translate(self, text: str) -> str:
        return self.process_text_single_shot(text)
    
    def refine(self, text: str) -> str:
        return text

    def format_text(self, text: str) -> str:
        prompt = f"""
        Format the following text into structured Markdown.
        - Use # for the Main Title.
        - Use ## for Section Headings.
        - Use bullet points (-) for lists.
        - Use **bold** for key terms or names.
        
        Text:
        {text}
        """
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            if isinstance(response, dict):
                return response['message']['content'].strip()
            return response.message.content.strip()
        except Exception as e:
            return f"Error during Ollama Formatting: {e}\n\n{text}"
