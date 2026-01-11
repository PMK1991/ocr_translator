import os
import asyncio
import importlib
from typing import List, Optional, Any
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_core.models import UserMessage, ModelInfo, ModelFamily, ChatCompletionClient, CreateResult, LLMMessage, SystemMessage, UserMessage, AssistantMessage, ModelCapabilities, RequestUsage
from src.docx_saver import save_markdown_to_docx
from src.input_handler import load_document

# --- Configuration Constants ---
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# System Prompt Configuration - Agentic OCR Decision Making
PROCESSOR_SYSTEM_MESSAGE = """You are an expert document processor and translator.

STEP 1 - Analyze File:
Call `get_file_info` to understand the file type and characteristics.

STEP 2 - Extract Text (Choose ONE based on file info):
- If file_type is "image" → Call `perform_ocr`
- If file_type is "pdf" AND has_native_text is True → Call `extract_native_text` (faster, more accurate)
- If file_type is "pdf" AND has_native_text is False → Call `perform_ocr` (for scanned documents)

STEP 3 - Translate:
After receiving the extracted text, you MUST translate the ENTIRE document to professional English.
- CRITICAL: You must output the ACTUAL translation, not a description or placeholder.
- Do NOT write things like "[translation here]" or "[Full translation...]"
- You MUST translate every single sentence from the source language to English.
- Maintain the original structure and meaning.

STEP 4 - Save:
Call `save_document` with the ACTUAL TRANSLATED TEXT as the content parameter.
- The content parameter MUST contain your real English translation.
- Do NOT pass placeholders, descriptions, or summaries.
- Pass the FULL translated text.

Say TERMINATE when complete."""

# --- Agentic Tools ---

async def get_file_info(filepath: str) -> str:
    """
    Analyzes a file and returns metadata to help decide extraction method.
    Returns: file_type (pdf/image), page_count, has_native_text
    """
    import fitz  # PyMuPDF
    from PIL import Image
    
    print(f"[Tool] get_file_info called for: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    result = {
        "filepath": filepath,
        "file_type": "unknown",
        "page_count": 0,
        "has_native_text": False,
        "recommendation": ""
    }
    
    # Check if image
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']:
        result["file_type"] = "image"
        result["page_count"] = 1
        result["has_native_text"] = False
        result["recommendation"] = "Use perform_ocr for image files"
        return str(result)
    
    # Check if PDF
    if ext == '.pdf':
        result["file_type"] = "pdf"
        try:
            doc = fitz.open(filepath)
            result["page_count"] = len(doc)
            
            # Check if PDF has extractable text (sample first few pages)
            total_text_chars = 0
            pages_to_check = min(3, len(doc))
            for i in range(pages_to_check):
                page = doc[i]
                text = page.get_text().strip()
                total_text_chars += len(text)
            
            doc.close()
            
            # If average text per page > 100 chars, assume native text exists
            avg_chars = total_text_chars / pages_to_check if pages_to_check > 0 else 0
            result["has_native_text"] = avg_chars > 100
            
            if result["has_native_text"]:
                result["recommendation"] = "Use extract_native_text for faster, more accurate extraction"
            else:
                result["recommendation"] = "Use perform_ocr - this appears to be a scanned document"
                
        except Exception as e:
            result["error"] = str(e)
            result["recommendation"] = "Error analyzing PDF, try perform_ocr"
        
        return str(result)
    
    result["recommendation"] = f"Unknown file type: {ext}"
    return str(result)


async def extract_native_text(filepath: str) -> str:
    """
    Extracts native/selectable text from a PDF document.
    Use this for digital PDFs that have embedded text layers.
    This is faster and more accurate than OCR for such documents.
    """
    import fitz  # PyMuPDF
    
    print(f"[Tool] extract_native_text called for: {filepath}")
    
    try:
        doc = fitz.open(filepath)
        all_text = []
        
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                all_text.append(f"--- Page {i+1} ---\n{text}")
            else:
                all_text.append(f"--- Page {i+1} ---\n[No text found on this page - may need OCR]")
        
        doc.close()
        
        full_text = "\n\n".join(all_text)
        print(f"[Tool] Extracted {len(full_text)} characters from {len(all_text)} pages using native text extraction")
        return full_text
        
    except Exception as e:
        return f"Error extracting native text: {e}. Try using perform_ocr instead."


async def perform_ocr(filepath: str, engine_name: str) -> str:
    """
    Performs OCR (Optical Character Recognition) on a file.
    Automatically preprocesses images for better accuracy (sharpening, contrast, denoising).
    
    Use this for:
    - Image files (PNG, JPG, etc.)
    - Scanned PDF documents that don't have selectable text
    
    Args:
        filepath: Path to the image or PDF file
        engine_name: OCR engine to use (azure, google, ollama, deepseek)
    """
    from src.image_preprocessing import preprocess_for_ocr
    
    print(f"[Tool] perform_ocr called for: {filepath} with engine: {engine_name}")
    
    # Determine module based on engine name
    mod_name = None
    if 'azure' in engine_name:
        mod_name = 'src.azure_ocr'
    elif 'deepseek' in engine_name:
        mod_name = 'src.deepseek_local_ocr'
    elif 'ollama' in engine_name:
        mod_name = 'src.ollama_sdk_ocr'
    else:
        mod_name = 'src.google_cloud_ocr'
            
    try:
        mod = importlib.import_module(mod_name)
        ocr_engine = mod.OCREngine()
    except ImportError as e:
        return f"Error loading OCR module '{mod_name}': {e}"
    except Exception as e:
        return f"Error initializing OCR engine: {e}"
    
    # Load Document
    try:
        pages = load_document(filepath)
    except Exception as e:
        return f"Error loading document: {e}"

    full_text = []
    print(f"[Tool] OCR processing {len(pages)} pages with preprocessing...")
    
    for i, p in enumerate(pages):
        if p['type'] == 'text':
            # Even in OCR mode, if we get text, use it
            full_text.append(f"--- Page {i+1} ---\n{p['content']}")
        else:
            try:
                # Apply preprocessing for better OCR
                print(f"[Tool] Preprocessing page {i+1}...")
                preprocessed_image = preprocess_for_ocr(
                    p['content'],
                    auto_detect_blur=True,
                    force_contrast=True  # Always enhance contrast for OCR
                )
                
                txt = ocr_engine.extract_text(preprocessed_image)
                full_text.append(f"--- Page {i+1} ---\n{txt}")
                print(f"[Tool] OCR completed page {i+1}")
            except Exception as e:
                full_text.append(f"--- Page {i+1} ---\n[OCR Error: {e}]")
    
    result = "\n\n".join(full_text)
    print(f"[Tool] OCR extracted {len(result)} total characters")
    return result


async def save_document(content: str, filename_base: str) -> str:
    """Saves translated text to a DOCX file."""
    path = filename_base + "_autogen.docx"
    try:
        save_markdown_to_docx(content, path)
        print(f"[Tool] Document saved to: {path}")
        return f"Successfully saved to {path}"
    except Exception as e:
        return f"Error saving document: {e}"


# --- Google GenAI Client Adapter ---
from src.google_cloud_agents import GoogleGenAIClient

# --- Orchestrator ---
class AutoGenOrchestrator:
    def __init__(self, engine_name: str, file_path: str):
        self.engine_name = engine_name
        self.file_path = file_path
        
    async def _run_async(self):
        base_name = os.path.splitext(os.path.basename(self.file_path))[0]
        
        # 1. Select Client based on Engine
        if 'google' in self.engine_name:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            
            api_key = os.getenv("GEMINI_API_KEY")
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # Use stable 2.0, not 3.x preview
            if not api_key:
                print("Error: GEMINI_API_KEY missing.")
                return
            
            # Use Google's OpenAI-compatible endpoint for FULL tool calling support
            model_client = OpenAIChatCompletionClient(
                model=model,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=api_key,
                model_info=ModelInfo(
                    vision=True,
                    function_calling=True,
                    json_output=True,
                    family=ModelFamily.UNKNOWN,
                    structured_output=False  # Add this to suppress warning
                )
            )
            print(f"AutoGen: Using Google Gemini ({model}) via OpenAI-compatible API - FULL AGENTIC MODE")
            
            # Create Agent WITH all agentic tools (same as Azure!)
            participant = AssistantAgent(
                name="Processor_Agent",
                model_client=model_client,
                tools=[get_file_info, extract_native_text, perform_ocr, save_document],
                system_message=PROCESSOR_SYSTEM_MESSAGE
            )

            task_prompt = (
                f"Process the file: '{self.file_path}'.\n"
                f"Engine for OCR (if needed): '{self.engine_name}'.\n"
                f"Save output as: '{base_name}'.\n\n"
                f"Follow the steps in your instructions to analyze, extract, translate, and save."
            )

        else:
            # Default to Azure (With Full Agentic Tools)
            if not AZURE_API_KEY or not AZURE_ENDPOINT:
                print("Error: Azure OpenAI credentials missing.")
                return

            model_client = AzureOpenAIChatCompletionClient(
                azure_deployment=AZURE_DEPLOYMENT,
                model=AZURE_DEPLOYMENT,
                api_version=AZURE_API_VERSION,
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                model_info=ModelInfo(
                    vision=True,
                    function_calling=True,
                    json_output=True,
                    family=ModelFamily.GPT_4
                )
            )
            print(f"AutoGen: Using Azure Client ({AZURE_DEPLOYMENT}) - FULL AGENTIC MODE")
            
            # Create Agent WITH all agentic tools
            participant = AssistantAgent(
                name="Processor_Agent",
                model_client=model_client,
                tools=[get_file_info, extract_native_text, perform_ocr, save_document],
                system_message=PROCESSOR_SYSTEM_MESSAGE
            )

            task_prompt = (
                f"Process the file: '{self.file_path}'.\n"
                f"Engine for OCR (if needed): '{self.engine_name}'.\n"
                f"Save output as: '{base_name}'.\n\n"
                f"Follow the steps in your instructions to analyze, extract, translate, and save."
            )

        # 3. Create Team
        termination = TextMentionTermination("TERMINATE")
        team = RoundRobinGroupChat([participant], termination_condition=termination)
        
        print(f"Starting AutoGen Team for task...")
        
        # 4. Run with Console output
        await Console(team.run_stream(task=task_prompt))

    def run(self):
        """Entry point called by main.py (Sync wrapper for Async)"""
        asyncio.run(self._run_async())
