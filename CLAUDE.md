# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

OCR Translator: an agentic pipeline that extracts text from images/PDFs (via OCR or native text extraction), translates it to English, refines the translation, formats it as Markdown, and saves output as `.docx` and `.txt` files.

## Running

```bash
# Simple framework (default) with Google engine
python main.py <file> --engine google

# With AutoGen multi-agent framework
python main.py <file> --engine azure --framework autogen

# Engines: google, azure, ollama, deepseek
# Frameworks: simple (default), autogen
```

No test suite exists. No build step. No linter configured.

## Environment Variables

Configured via `.env` (loaded by python-dotenv):

- **Google**: `GEMINI_API_KEY`, `GEMINI_MODEL` (default: `gemini-1.5-pro`)
- **Azure**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` (default: `gpt-4o`), `AZURE_OPENAI_API_VERSION`
- **Ollama**: `OLLAMA_API_KEY` (if set, uses cloud endpoint `https://ollama.com`; otherwise falls back to local)
- **DeepSeek**: check `src/deepseek_local_ocr.py` and `src/deepseek_local_agents.py`

## Architecture

### Two Orchestration Frameworks

1. **Simple** (`main.py` directly): Linear pipeline — load document, parallel OCR across pages, single-shot translate+refine, format, save. Uses `concurrent.futures.ThreadPoolExecutor` (5 workers).

2. **AutoGen** (`src/autogen_flow.py`): Microsoft AutoGen `RoundRobinGroupChat` with a single `Processor_Agent` that has tool-calling access to `get_file_info`, `extract_native_text`, `perform_ocr`, and `save_document`. The agent decides whether to use native text extraction or OCR based on file analysis.

### Engine Abstraction

Each engine backend provides two classes loaded dynamically via `importlib`:
- `OCREngine` with `extract_text(image: PIL.Image) -> str`
- `AgentManager` with `process_text_single_shot(text, lang)`, `format_text(text)`, and legacy `translate()`/`refine()` methods

Engine backends live under `src/engines/{google,azure,ollama,deepseek}/`, each with `ocr.py` and `agents.py`. They are loaded dynamically via `ENGINE_MAP` in `main.py`.

### Key Modules

- `src/input_handler.py` — Loads PDFs (via PyMuPDF/fitz) and images. For PDFs, uses a 50-char heuristic to decide between native text extraction and image rendering for OCR.
- `src/image_preprocessing.py` — OpenCV-based preprocessing (sharpening, CLAHE contrast, denoising, binarization, blur detection). Used by the AutoGen `perform_ocr` tool.
- `src/docx_saver.py` — Converts simple Markdown (headers, bold, bullets) to `.docx` via python-docx.
- `src/engines/google/agents.py` — Also contains `GoogleGenAIClient`, an AutoGen `ChatCompletionClient` adapter for Google GenAI, used by `autogen_flow.py`.

### Dependencies (no requirements.txt — install manually)

Key packages: `google-genai`, `openai`, `ollama`, `python-dotenv`, `Pillow`, `PyMuPDF` (fitz), `python-docx`, `opencv-python`, `numpy`, `autogen-agentchat`, `autogen-ext`, `autogen-core`

## Output Naming Convention

Output files use engine-specific suffixes: `_gcp` (google), `_az` (azure), `_ollama`, `_ds` (deepseek). AutoGen outputs use `_autogen` suffix.
