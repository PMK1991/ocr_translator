import os
from typing import List
from PIL import Image
import fitz  # PyMuPDF

def load_document(file_path: str):
    """
    Loads a document.
    - If PDF: Checks each page for native text. 
      - If text found (>50 chars), returns {'type': 'text', 'content': text}.
      - Else, renders to image and returns {'type': 'image', 'content': PIL_Image}.
    - If Image: Returns {'type': 'image', 'content': PIL_Image}.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == '.pdf':
        try:
            doc = fitz.open(file_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Check for native text
                text = page.get_text().strip()
                
                # Heuristic: If meaningful text exists (e.g. > 50 chars), use it.
                # Adjust threshold as needed. Sometimes OCR artifacts or watermarks appear as text.
                if len(text) > 50:
                    pages_content.append({'type': 'text', 'content': text})
                else:
                    # Render to image for OCR
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pages_content.append({'type': 'image', 'content': img})
            
            return pages_content
            
        except Exception as e:
            raise RuntimeError(f"Failed to process PDF with PyMuPDF. Error: {e}")
    
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        try:
            image = Image.open(file_path)
            return [{'type': 'image', 'content': image}]
        except Exception as e:
            raise RuntimeError(f"Failed to open image. Error: {e}")
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")
