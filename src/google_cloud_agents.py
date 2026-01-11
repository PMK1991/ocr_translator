import os
from typing import List, Any
from google import genai
from google.genai import types

# Optional AutoGen Support (Assumed installed based on requirements)
try:
    from autogen_core.models import (
        ChatCompletionClient, CreateResult, LLMMessage, SystemMessage, 
        UserMessage, AssistantMessage, ModelCapabilities, RequestUsage, 
        ModelInfo, ModelFamily
    )
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Define dummy base class to avoid NameError if not installed
    ChatCompletionClient = object 

# Configuration Constants
DEFAULT_MODEL_NAME = "gemini-1.5-pro"

class AgentManager:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
            
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL_NAME)

    def process_text_single_shot(self, text: str, source_lang_hint: str = None) -> str:
        """
        Combines Translation and Refinement into a single call using Gemini.
        """
        prompt = f"""
        You are an expert Translator and Editor.
        
        Task:
        1. Translate the following text to professional English.
        2. Simultaneously refine the grammar, flow, and fix potential OCR artifacts.
        
        Source Hint: {source_lang_hint or "Unknown"}
        
        Input Text:
        \"\"\"
        {text}
        \"\"\"
        
        Output ONLY the final, polished English translation.
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text.strip()
        except Exception as e:
            return f"Error during Google GenAI Processing: {e}"

    # Legacy method support
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
        - Do not change the content, just structure it for a document.

        Text:
        {text}
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text.strip()
        except Exception as e:
            return f"Error during Google GenAI Formatting: {e}\n\n{text}"

if AUTOGEN_AVAILABLE:
    class GoogleGenAIClient(ChatCompletionClient):
        def __init__(self, api_key: str, model_name: str):
            self.client = genai.Client(api_key=api_key)
            self.model_name = model_name
            self._capabilities = ModelCapabilities(vision=True, function_calling=True, json_output=False)
            self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        @property
        def capabilities(self) -> ModelCapabilities:
            return self._capabilities

        @property
        def model_info(self) -> ModelInfo:
            return ModelInfo(vision=True, function_calling=True, json_output=False, family=ModelFamily.UNKNOWN)

        def actual_usage(self) -> RequestUsage:
            return self._usage

        def total_usage(self) -> RequestUsage:
            return self._usage

        def count_tokens(self, messages: List[LLMMessage], tools: List[Any] = []) -> int:
            return 0 

        def remaining_tokens(self, messages: List[LLMMessage], tools: List[Any] = []) -> int:
            return 100000 

        def close(self):
            pass

        async def create_stream(self, messages: List[LLMMessage], **kwargs):
            result = await self.create(messages, **kwargs)
            yield result

        async def create(self, messages: List[LLMMessage], **kwargs) -> CreateResult:
            contents = []
            system_instruction = None
            
            for m in messages:
                if isinstance(m, SystemMessage):
                    system_instruction = m.content
                elif isinstance(m, UserMessage):
                    if isinstance(m.content, str):
                        contents.append(types.Content(role="user", parts=[types.Part(text=m.content)]))
                elif isinstance(m, AssistantMessage):
                    if isinstance(m.content, str):
                        contents.append(types.Content(role="model", parts=[types.Part(text=m.content)]))

            config = types.GenerateContentConfig(
                temperature=kwargs.get('temperature', 0.1),
                system_instruction=system_instruction
            )
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                return CreateResult(
                    finish_reason="stop",
                    content=response.text or "",
                    usage=RequestUsage(prompt_tokens=0, completion_tokens=0),
                    cached=False
                )
            except Exception as e:
                raise RuntimeError(f"Google GenAI Error: {e}")
