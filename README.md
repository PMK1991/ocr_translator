# OCR Translator

Agentic pipeline that extracts text from images and PDFs, translates it to English, and saves polished output as `.docx` and `.txt`.

## How It Works

1. **Load** — PDFs are split into pages; native text is extracted directly, scanned pages are rendered as images
2. **OCR** — Pages are processed in parallel (5 workers) using a vision LLM as the OCR engine
3. **Translate & Refine** — Full text is translated to English and cleaned of OCR artifacts in a single LLM call
4. **Format & Save** — Output is structured as Markdown, then saved as both `.docx` and `.txt`

## Setup

```bash
pip install google-genai openai ollama python-dotenv Pillow PyMuPDF python-docx opencv-python numpy
```

For the AutoGen framework:
```bash
pip install autogen-agentchat autogen-ext autogen-core
```

Create a `.env` file with credentials for your chosen engine:

```env
# Google Gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-1.5-pro        # optional

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o     # optional

# Ollama (cloud)
OLLAMA_API_KEY=...                  # omit for local Ollama

# Ollama / DeepSeek (local)
OLLAMA_HOST=http://localhost:11434  # optional
OLLAMA_MODEL=deepseek-r1:latest    # optional
```

## Usage

```bash
# Basic — OCR + translate an image or PDF
python main.py document.pdf

# Specify source language hint and engine
python main.py scan.jpg --lang Sanskrit --engine azure

# Use the AutoGen multi-agent framework
python main.py document.pdf --engine google --framework autogen
```

### Engines

| Engine     | Provider         | Requires              |
|------------|------------------|-----------------------|
| `google`   | Google Gemini    | `GEMINI_API_KEY`      |
| `azure`    | Azure OpenAI     | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| `ollama`   | Ollama (cloud/local) | `OLLAMA_API_KEY` or local Ollama running |
| `deepseek` | DeepSeek via Ollama | Local Ollama with models pulled |

### Frameworks

- **`simple`** (default) — Linear pipeline orchestrated directly in `main.py`
- **`autogen`** — Microsoft AutoGen agent that autonomously decides whether to use native text extraction or OCR based on file analysis

## Project Structure

```
main.py                        # Entry point and simple framework pipeline
src/
  engines/
    google/  ocr.py, agents.py # Google Gemini backend
    azure/   ocr.py, agents.py # Azure OpenAI backend
    ollama/  ocr.py, agents.py # Ollama SDK backend
    deepseek/ocr.py, agents.py # DeepSeek (local Ollama) backend
  autogen_flow.py              # AutoGen multi-agent orchestration
  input_handler.py             # PDF/image loading (PyMuPDF)
  image_preprocessing.py       # OpenCV preprocessing for OCR
  docx_saver.py                # Markdown-to-DOCX converter
```

## Supported Formats

**Input:** PDF, JPG, JPEG, PNG, BMP, TIFF, WebP

**Output:** `.docx` and `.txt` (named with engine suffix, e.g. `document_gcp_refined.docx`)
