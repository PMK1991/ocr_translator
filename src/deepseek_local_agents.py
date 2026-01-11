import ollama

class AgentManager:
    def __init__(self, api_key: str = None):
        # User confirmed 'deepseek-r1:latest' is pulled.
        self.model_name = "deepseek-r1:latest"

    def process_text_single_shot(self, text: str, source_lang_hint: str = None) -> str:
        prompt = f"""
        Translate the following text to English and refine it for clarity.
        Source Hint: {source_lang_hint or "Unknown"}
        Output ONLY the final translation.
        
        Text:
        {text}
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error during DeepSeek Agent Processing: {e}"

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
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            return response['message']['content'].strip()
        except Exception as e:
            return f"Error during DeepSeek Formatting: {e}\n\n{text}"
