import os
from openai import AzureOpenAI

# Configuration Constants
DEFAULT_API_VERSION = "2024-02-15-preview"
DEFAULT_DEPLOYMENT = "gpt-4o"

class AgentManager:
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

    def process_text_single_shot(self, text: str, source_lang_hint: str = None) -> str:
        prompt = f"""
        Translate the following text to professional English and refine it for clarity.
        Source Hint: {source_lang_hint or "Unknown"}
        Output ONLY the final translation.
        
        Text:
        {text}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "You are a professional academic translator. You are translating historical or literary texts. Maintain the original meaning accurately, even if it contains sensitive topics."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during Azure Agent Processing: {e}"

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
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error during Azure Formatting: {e}\n\n{text}"
