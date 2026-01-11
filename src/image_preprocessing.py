"""
Image Preprocessing Module for OCR Enhancement

Applies various image processing techniques to improve OCR accuracy:
- Sharpening
- Contrast Enhancement (CLAHE)
- Denoising
- Binarization (for text documents)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional

# Configuration
DEFAULT_SHARPEN_STRENGTH = 1.5
DEFAULT_DENOISE_STRENGTH = 10
ENABLE_SUPER_RESOLUTION = False  # Requires additional models, disabled by default


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format (BGR)."""
    rgb = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to PIL Image."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def sharpen_image(image: np.ndarray, strength: float = DEFAULT_SHARPEN_STRENGTH) -> np.ndarray:
    """
    Apply unsharp masking to sharpen the image.
    Higher strength = more sharpening.
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    for local contrast enhancement.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return result


def denoise_image(image: np.ndarray, strength: int = DEFAULT_DENOISE_STRENGTH) -> np.ndarray:
    """
    Apply Non-local Means Denoising.
    Good for removing noise while preserving edges (important for text).
    """
    # For color images
    denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    return denoised


def binarize_for_text(image: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale and apply adaptive thresholding.
    Useful for text-heavy documents with uneven lighting.
    Returns a color image (3-channel) for compatibility.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    # Convert back to 3-channel for compatibility
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def detect_blur(image: np.ndarray) -> tuple[bool, float]:
    """
    Detect if image is blurry using Laplacian variance.
    Returns (is_blurry, blur_score).
    Lower score = more blurry.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Threshold for blur detection (tune based on your images)
    BLUR_THRESHOLD = 100.0
    is_blurry = variance < BLUR_THRESHOLD
    
    return is_blurry, variance


def preprocess_for_ocr(
    pil_image: Image.Image,
    auto_detect_blur: bool = True,
    force_sharpen: bool = False,
    force_denoise: bool = False,
    force_contrast: bool = False,
    binarize: bool = False
) -> Image.Image:
    """
    Main preprocessing function for OCR enhancement.
    
    Args:
        pil_image: Input PIL Image
        auto_detect_blur: Automatically detect and fix blur
        force_sharpen: Always apply sharpening
        force_denoise: Always apply denoising
        force_contrast: Always apply contrast enhancement
        binarize: Convert to binary (black/white) - good for old documents
    
    Returns:
        Preprocessed PIL Image
    """
    cv_img = pil_to_cv2(pil_image)
    original_shape = cv_img.shape
    
    print(f"[Preprocess] Input image size: {original_shape[1]}x{original_shape[0]}")
    
    # Detect blur
    is_blurry, blur_score = detect_blur(cv_img)
    print(f"[Preprocess] Blur detection - Score: {blur_score:.2f}, Is Blurry: {is_blurry}")
    
    # Apply preprocessing based on detection or forced flags
    if auto_detect_blur and is_blurry:
        print("[Preprocess] Applying sharpening (blur detected)")
        cv_img = sharpen_image(cv_img, strength=2.0)  # Stronger for blurry images
        print("[Preprocess] Applying denoising")
        cv_img = denoise_image(cv_img)
    
    if force_sharpen and not (auto_detect_blur and is_blurry):
        print("[Preprocess] Applying sharpening (forced)")
        cv_img = sharpen_image(cv_img)
    
    if force_denoise and not (auto_detect_blur and is_blurry):
        print("[Preprocess] Applying denoising (forced)")
        cv_img = denoise_image(cv_img)
    
    if force_contrast:
        print("[Preprocess] Applying contrast enhancement (CLAHE)")
        cv_img = enhance_contrast(cv_img)
    
    if binarize:
        print("[Preprocess] Applying binarization")
        cv_img = binarize_for_text(cv_img)
    
    # Always apply mild contrast enhancement for OCR
    if not force_contrast and not binarize:
        print("[Preprocess] Applying mild contrast enhancement")
        cv_img = enhance_contrast(cv_img)
    
    result = cv2_to_pil(cv_img)
    print(f"[Preprocess] Output image size: {result.size[0]}x{result.size[1]}")
    
    return result


# Simple super-resolution using OpenCV (basic upscaling with enhancement)
def upscale_image(pil_image: Image.Image, scale: float = 2.0) -> Image.Image:
    """
    Simple upscaling with sharpening for better OCR on small text.
    For true super-resolution, consider using ESRGAN or similar models.
    """
    cv_img = pil_to_cv2(pil_image)
    
    # Upscale using INTER_CUBIC (good balance of speed and quality)
    height, width = cv_img.shape[:2]
    new_size = (int(width * scale), int(height * scale))
    upscaled = cv2.resize(cv_img, new_size, interpolation=cv2.INTER_CUBIC)
    
    # Apply sharpening to compensate for upscale blur
    sharpened = sharpen_image(upscaled, strength=1.0)
    
    print(f"[Preprocess] Upscaled from {width}x{height} to {new_size[0]}x{new_size[1]}")
    
    return cv2_to_pil(sharpened)
