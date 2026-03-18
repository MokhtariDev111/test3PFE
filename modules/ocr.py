"""
ocr.py  —  Step 3: OCR Integration
==================================
Receives: list[DocumentPage] from Step 2.
Action:   Finds all pages where type == "image" (or text == ""),
          runs OCR to extract text, and populates the `text` field.
Returns:  The same list[DocumentPage], now fully text-populated.
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path

import sys
from pathlib import Path

# Add project root to sys.path so we can import modules.*
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from modules.ingestion import DocumentPage
except ImportError:
    pass

from modules.config_loader import CONFIG

log = logging.getLogger("ocr")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


class OCREngine:
    def __init__(self):
        self.engine_type = CONFIG["ocr"]["engine"].lower()  # "easyocr" | "tesseract"
        self.languages = CONFIG["ocr"]["languages"]         # e.g., ["fr", "en"]
        self._reader = None
        log.info(f"Initializing OCR engine: {self.engine_type.upper()} for {self.languages}")
        self._initialize()

    def _initialize(self):
        """Loads the required OCR engine into memory."""
        if self.engine_type == "easyocr":
            try:
                import easyocr
                # GPU is automatically used if available via PyTorch
                self._reader = easyocr.Reader(self.languages)
                log.info("  ✔ EasyOCR loaded successfully.")
            except ImportError:
                log.error("EasyOCR not installed. Run: pip install easyocr")
                raise
        elif self.engine_type == "tesseract":
            try:
                import pytesseract
                self._reader = pytesseract
                log.info("  ✔ Tesseract binding loaded (make sure Tesseract-OCR is installed on your OS).")
            except ImportError:
                log.error("pytesseract not installed. Run: pip install pytesseract")
                raise
        else:
            raise ValueError(f"Unknown OCR engine: {self.engine_type}")

    def process_image(self, pil_image) -> str:
        """Runs the chosen OCR engine on a PIL Image and returns the extracted text."""
        if self.engine_type == "easyocr":
            # EasyOCR expects a numpy array
            img_np = np.array(pil_image)
            # detail=0 returns a simple list of strings instead of bounding boxes
            results = self._reader.readtext(img_np, detail=0)
            return "\n".join(results)
            
        elif self.engine_type == "tesseract":
            # Map ["fr", "en"] to tesseract format "fra+eng"
            lang_map = {"fr": "fra", "en": "eng"}
            tess_langs = "+".join([lang_map.get(l, l) for l in self.languages])
            return self._reader.image_to_string(pil_image, lang=tess_langs).strip()


def run_ocr(pages: list[DocumentPage]) -> list[DocumentPage]:
    """
    Scans the list of DocumentPage objects.
    If a page is an image (or has no text but has an image object), runs OCR.
    """
    needs_ocr = [p for p in pages if (p.type == "image" or not p.text) and p.image is not None]
    
    if not needs_ocr:
        log.info("No images/scans found. Skipping OCR.")
        return pages

    log.info(f"Found {len(needs_ocr)} page(s) requiring OCR.")
    engine = OCREngine()
    
    processed = 0
    for page in pages:
        if (page.type == "image" or not page.text) and page.image is not None:
            text = engine.process_image(page.image)
            page.text = text.strip()
            processed += 1
            log.debug(f"  OCR output for {page.source}: {len(page.text)} chars")
            
    log.info(f"OCR complete. {processed} images processed.")
    return pages


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    
    from modules.ingestion import ingest_directory
    
    raw_dir = Path(CONFIG["paths"]["data_raw"])
    print("\n[Step 2] Ingesting data...")
    pages = ingest_directory(raw_dir)
    
    print("\n[Step 3] Running OCR...")
    processed_pages = run_ocr(pages)
    
    print("\n── OCR Verification ──────────────────────────")
    img_pages = [p for p in processed_pages if p.type == "image"]
    if img_pages:
        for p in img_pages[:3]:
            preview = p.text[:100].replace('\n', ' ') + "…" if len(p.text) > 100 else p.text.replace('\n', ' ')
            print(f"[{p.source}] OCR Text: {preview}")
    else:
        print("No images were found in data/raw/ to OCR.")
        
    print("\n✅ Step 3 (OCR) logic executed.")
