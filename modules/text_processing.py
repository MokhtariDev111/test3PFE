"""
text_processing.py  —  Step 4: Text Cleaning & Chunking
======================================================
Receives: list[DocumentPage] from Step 3 (all with text populated).
Action:   Cleans the text (removes extra whitespace, fixes hyphens) and 
          splits it into overlapping TextChunks.
Returns:  list[TextChunk] ready for embedding (Step 5).
"""

import re
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

# Setup imports so it can run standalone
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from modules.ingestion import DocumentPage
except ImportError:
    pass

from modules.config_loader import CONFIG

log = logging.getLogger("text_processing")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


@dataclass
class TextChunk:
    """A smaller piece of text ready to be embedded."""
    text: str
    source: str
    page: int
    chunk_id: int


def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not text:
        return ""
    # 1. Fix broken hyphenation at the end of lines (e.g. "intelli-\ngence" -> "intelligence")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # 2. Replace all remaining newlines with spaces (keeps chunks contiguous)
    text = text.replace("\n", " ")
    # 3. Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits text into chunks of approximately `chunk_size` characters,
    with `chunk_overlap` characters overlapping between chunks.
    
    This is a simple character-based sliding window approach.
    For production, consider `langchain.text_splitter.RecursiveCharacterTextSplitter`.
    """
    if not text:
        return []

    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1  # +1 for the space
        if current_length + word_len > chunk_size and current_chunk:
            # Chunk is full, save it
            chunks.append(" ".join(current_chunk))
            
            # Keep the last few words for overlap
            overlap_words = []
            overlap_len = 0
            for w in reversed(current_chunk):
                if overlap_len + len(w) + 1 > chunk_overlap:
                    break
                overlap_words.insert(0, w)
                overlap_len += len(w) + 1
            
            current_chunk = overlap_words
            current_length = overlap_len

        current_chunk.append(word)
        current_length += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_pages(pages: list[DocumentPage]) -> list[TextChunk]:
    """
    Main pipeline for Step 4.
    Cleans DocumentPages and flattens them into a list of TextChunks.
    """
    log.info(f"Processing {len(pages)} pages (cleaning & chunking)...")
    
    chunk_size = CONFIG["text_processing"].get("chunk_size", 512)
    chunk_overlap = CONFIG["text_processing"].get("chunk_overlap", 64)
    
    all_chunks: list[TextChunk] = []
    chunk_counter = 0

    for page in pages:
        if not page.text:
            continue
            
        cleaned = clean_text(page.text)
        text_pieces = chunk_text(cleaned, chunk_size, chunk_overlap)
        
        for piece in text_pieces:
            all_chunks.append(TextChunk(
                text=piece,
                source=page.source,
                page=page.page,
                chunk_id=chunk_counter
            ))
            chunk_counter += 1

    log.info(f"Created {len(all_chunks)} chunks (size={chunk_size}, overlap={chunk_overlap}).")
    return all_chunks


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.ingestion import ingest_directory
    try:
        from modules.ocr import run_ocr
    except ImportError:
        run_ocr = lambda p: p  # Mock OCR if missing
    
    raw_dir = Path(CONFIG["paths"]["data_raw"])
    
    print("\n[Step 2] Ingesting data...")
    pages = ingest_directory(raw_dir)
    
    print("\n[Step 3] Running/Skipping OCR...")
    try:
        pages = run_ocr(pages)
    except Exception as e:
        print(f"Skipping OCR due to error: {e}")
    
    print("\n[Step 4] Processing text...")
    chunks = process_pages(pages)
    
    print(f"\n── First 3 Chunks ──────────────────────────")
    for chunk in chunks[:3]:
        print(f"\n[{chunk.source} | p.{chunk.page} | ID {chunk.chunk_id}]")
        print(f"Text ({len(chunk.text)} chars): {chunk.text}")
        
    print(f"\n✅ Step 4 (Text Proc) generated {len(chunks)} chunks.")
