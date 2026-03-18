"""
ingestion.py  —  Step 2: Data Ingestion
Loads PDFs (text) and images from data/raw/.
Produces a list of DocumentPage dicts for downstream steps.
"""

from pathlib import Path
from dataclasses import dataclass, field
import fitz          # PyMuPDF
from PIL import Image

PDF_EXT   = {".pdf"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
TXT_EXT   = {".txt"}


@dataclass
class DocumentPage:
    source: str
    page:   int          # 1-based for PDFs, 0 for images/txt
    type:   str          # "pdf" | "image" | "txt"
    text:   str = ""     # empty for images → filled by OCR (Step 3)
    image:  object = field(default=None, repr=False)


def load_txt(path: Path) -> DocumentPage:
    """Reads a plain text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read().strip()
    print(f"  [TXT]  {path.name}  →  {len(text)} chars")
    return DocumentPage(source=path.name, page=0, type="txt", text=text)


def load_pdf(path: Path) -> list[DocumentPage]:
    doc = fitz.open(str(path))
    pages = []
    
    for i in range(len(doc)):
        page = doc[i]
        
        # 1. Extract text (if any)
        text = page.get_text("text").strip()
        pages.append(
            DocumentPage(source=path.name, page=i + 1, type="pdf", text=text)
        )
        
        # 2. Extract embedded images on this page
        for img_idx, img_info in enumerate(page.get_images(full=True)):
            xref = img_info[0]
            base_img = doc.extract_image(xref)
            if base_img:
                from io import BytesIO
                try:
                    pil_img = Image.open(BytesIO(base_img["image"])).convert("RGB")
                    pages.append(
                        DocumentPage(
                            source=f"{path.name} (page {i+1} img {img_idx+1})", 
                            page=i + 1, 
                            type="image", 
                            image=pil_img
                        )
                    )
                except Exception as e:
                    print(f"  [Warn] Failed to load image {img_idx} on page {i+1}: {e}")
                    
    doc.close()
    
    img_count = sum(1 for p in pages if p.type == "image")
    txt_count = len(pages) - img_count
    print(f"  [PDF]  {path.name}  →  {txt_count} text pg(s), {img_count} embedded img(s)")
    return pages


def load_image(path: Path) -> DocumentPage:
    img = Image.open(path).convert("RGB")
    print(f"  [IMG]  {path.name}  →  {img.size[0]}×{img.size[1]} px")
    return DocumentPage(source=path.name, page=0, type="image", image=img)


def ingest_directory(raw_dir: str | Path) -> list[DocumentPage]:
    raw_dir = Path(raw_dir)
    results = []
    files   = sorted(raw_dir.iterdir())
    print(f"\nIngesting {len(files)} file(s) from '{raw_dir}' …")
    for f in files:
        ext = f.suffix.lower()
        if ext in PDF_EXT:
            results.extend(load_pdf(f))
        elif ext in IMAGE_EXT:
            results.append(load_image(f))
        elif ext in TXT_EXT:
            results.append(load_txt(f))
    print(f"\nTotal pages/images loaded: {len(results)}\n")
    return results


# ── Quick run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modules.config_loader import CONFIG

    pages = ingest_directory(CONFIG["paths"]["data_raw"])
    pdfs  = [p for p in pages if p.type == "pdf"]
    imgs  = [p for p in pages if p.type == "image"]
    empty = [p for p in pdfs  if not p.text]

    print(f"PDF pages : {len(pdfs)}")
    print(f"Images    : {len(imgs)}")
    print(f"Empty(OCR): {len(empty)}")
