import os
import google.generativeai as genai

class AgentManager:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def translate(self, text: str, source_lang_hint: str = None) -> str:
        """
        Translates text to English.
        """
        lang_context = f"The source language is likely {source_lang_hint}." if source_lang_hint else ""
        
        prompt = f"""
        You are an expert translator.
        Translate the following text to English.
        {lang_context}
        Maintain the original meaning and nuances.
        
        Text to translate:
        ---
        {text}
        ---
        
        Output only the English translation.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error during translation: {e}"

    def refine(self, text: str) -> str:
        """
        Refines English text for clarity and flow.
        """
        prompt = f"""
        You are an expert editor and language refiner.
        The following text is a raw translation from another language (possibly via OCR).
        It may contain grammatical errors, awkward phrasing, or OCR artifacts.
        
        Your task is to refine the text to be clear, professional, and natural-sounding in English.
        Do not change the underlying meaning.
        
        Text to refine:
        ---
        {text}
        ---
        
        Output only the refined text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error during refinement: {e}"
