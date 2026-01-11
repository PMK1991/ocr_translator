import os
import google.generativeai as genai

class AgentManager:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        
        # User requested specific model
        self.model = genai.GenerativeModel(
            'gemini-3-pro-preview',
            generation_config=genai.GenerationConfig(temperature=0.0)
        )

    def translate(self, text: str, source_lang_hint: str = None) -> str:
        """
        Translates text to English using STRICT guidelines.
        """
        lang_context = f"The source language is identified as: {source_lang_hint}." if source_lang_hint else "The source language is an Indian language."
        
        prompt = f"""
        # Task
        Translate the following text to English.

        # Strict Guidelines (DO NOT VIOLATE)
        1. **NO Hallucinations**: Do not contain ANY information that is not present in the source text. 
        2. **Literal over Creative**: If the text is fragmented, translate the fragments exactly. Do not try to make a coherent story if the source is not coherent.
        3. **Missing Text**: If parts are illegible, mark them as [ILLEGIBLE]. Do not guess.
        4. **Source Fidelity**: {lang_context}

        # Source Text
        \"\"\"
        {text}
        \"\"\"
        
        # Output
        English translation only.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error during translation: {e}"

    def refine(self, text: str) -> str:
        """
        Refines English text but forbids adding new meaning.
        """
        prompt = f"""
        # Task
        Fix grammatical errors in the following text.

        # Strict Guidelines
        1. **Do NOT add information**: You are a proofreader, not a co-author. Do not expand on concepts.
        2. **Do NOT hallucinate**: If the input text is Nonsense, output "Original text appears to be nonsense/garbled."
        3. **Consistency**: Keep the tone of the original input.

        # Input Text
        \"\"\"
        {text}
        \"\"\"
        
        # Output
        Refined text only.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error during refinement: {e}"
