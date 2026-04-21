import argparse
import concurrent.futures
import importlib
import os
from typing import Tuple, Any
from dotenv import load_dotenv

# Local Imports
from src.input_handler import load_document
from src.docx_saver import save_markdown_to_docx

# Load environment variables
load_dotenv()

# Constants
ENGINE_MAP = {
    'google': {'ocr': 'src.engines.google.ocr', 'agents': 'src.engines.google.agents'},
    'ollama': {'ocr': 'src.engines.ollama.ocr', 'agents': 'src.engines.ollama.agents'},
    'deepseek': {'ocr': 'src.engines.deepseek.ocr', 'agents': 'src.engines.deepseek.agents'},
    'azure': {'ocr': 'src.engines.azure.ocr', 'agents': 'src.engines.azure.agents'},
}

SUFFIX_MAP = {
    'google': '_gcp',
    'azure': '_az',
    'ollama': '_ollama',
    'deepseek': '_ds'
}

DEFAULT_WORKERS = 5

def get_engine_classes(engine_name: str) -> Tuple[Any, Any]:
    """Dynamically imports OCR and Agent classes based on engine name."""
    if engine_name not in ENGINE_MAP:
        raise ValueError(f"Unknown engine: {engine_name}. Available: {list(ENGINE_MAP.keys())}")
    
    paths = ENGINE_MAP[engine_name]
    try:
        ocr_module = importlib.import_module(paths['ocr'])
        agents_module = importlib.import_module(paths['agents'])
        return ocr_module.OCREngine, agents_module.AgentManager
    except ImportError as e:
        raise ImportError(f"Failed to load engine '{engine_name}': {e}")

def process_page(ocr_engine: Any, page_data: dict, page_num: int) -> str:
    """Processes a single page (Native Text or OCR)."""
    print(f"      [Start] Processing page {page_num}...")
    try:
        if page_data['type'] == 'text':
            # Native text found, skip OCR
            text = page_data['content']
            print(f"      [Done]  Page {page_num} used Native PDF Text ({len(text)} chars).")
            return text
        else:
            # Image, run OCR
            image = page_data['content']
            text = ocr_engine.extract_text(image)
            print(f"      [Done]  OCR on page {page_num} ({len(text)} chars).")
            return text
    except Exception as e:
        return f"[Error on page {page_num}: {e}]"

def main():
    parser = argparse.ArgumentParser(description="OCR -> Translate -> Refine Agentic System")
    parser.add_argument("file", help="Path to the image or PDF file to process")
    parser.add_argument("--lang", help="Source language name or code (optional hint for translator)", default=None)
    parser.add_argument("--engine", choices=list(ENGINE_MAP.keys()), default='google', 
                        help="Select backend engine.")
    parser.add_argument("--framework", choices=['simple', 'autogen'], default='simple',
                        help="Select orchestration framework.")
    
    args = parser.parse_args()
    file_path = args.file
    
    print(f"--- Processing: {file_path} using engine: [{args.engine.upper()}] ---")

    # Delegate to AutoGen Framework if selected
    if args.framework == 'autogen':
        print(">>> STARTING MICROSOFT AUTOGEN FRAMEWORK <<<")
        from src.autogen_flow import AutoGenOrchestrator
        orchestrator = AutoGenOrchestrator(args.engine, file_path)
        orchestrator.run()
        print(">>> AUTOGEN WORKFLOW COMPLETE <<<")
        return

    # --- Simple Framework Flow ---
    
    # 1. Load Document
    print("[1/4] Loading Document...")
    try:
        pages = load_document(file_path)
        print(f"      Loaded {len(pages)} page(s).")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Initialize Engine and Agents
    try:
        OCREngine, AgentManager = get_engine_classes(args.engine)
        ocr_engine = OCREngine()
        agent_manager = AgentManager()
    except Exception as e:
        print(f"ERROR initializing engines: {e}")
        return

    # 2. Parallel Processing (Hybrid OCR/Text)
    print(f"[2/4] Processing {len(pages)} page(s) in parallel...")
    full_transcription = ""
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as executor:
        future_to_page = {executor.submit(process_page, ocr_engine, page_data, i+1): i for i, page_data in enumerate(pages)}
        
        results: list[str] = [""] * len(pages)
        for future in concurrent.futures.as_completed(future_to_page):
            page_idx = future_to_page[future]
            results[page_idx] = future.result()
            
    full_transcription = "\n\n".join(results)

    # 3. Single-Shot Translate & Refine
    print("[3/4] Translating & Refining...")
    try:
        final_text = agent_manager.process_text_single_shot(full_transcription, args.lang)
    except AttributeError:
        # Fallback for older agent implementations
        print("Warning: Single-shot method not found, falling back to legacy pipeline.")
        translated = agent_manager.translate(full_transcription, args.lang)
        final_text = agent_manager.refine(translated)
    
    # 4. Formatter Agent & Save
    print("[4/4] Formatting & Saving to Docx...")
    
    if hasattr(agent_manager, 'format_text'):
        formatted_text = agent_manager.format_text(final_text)
    else:
        print("Formatting agent not available for this engine. Saving raw text.")
        formatted_text = final_text

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    suffix = SUFFIX_MAP.get(args.engine, f"_{args.engine}")
    
    # Save Text
    out_txt = f"{base_name}{suffix}_refined.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(formatted_text)
        
    # Save Docx (Translated)
    out_docx = f"{base_name}{suffix}_refined.docx"
    try:
        save_markdown_to_docx(formatted_text, out_docx)
        print(f"\nSUCCESS: Output saved to:\n  - {out_txt}\n  - {out_docx}")
    except Exception as e:
        print(f"\nWarning: Failed to create Docx ({e}). Text file saved.")

    # 5. Format & Save Original Text
    print("[Optional] Formatting & Saving Original Text...")
    try:
        if hasattr(agent_manager, 'format_text'):
            original_structured = agent_manager.format_text(full_transcription)
            out_orig_docx = f"{base_name}{suffix}_original.docx"
            save_markdown_to_docx(original_structured, out_orig_docx)
            print(f"  - {out_orig_docx}")
        else:
            out_orig_txt = f"{base_name}{suffix}_original.txt"
            with open(out_orig_txt, "w", encoding="utf-8") as f:
                f.write(full_transcription)
            print(f"  - {out_orig_txt} (Raw text only)")
    except Exception as e:
        print(f"Warning: Failed to save original docx ({e})")

if __name__ == "__main__":
    main()