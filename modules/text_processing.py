"""
text_processing.py — Semantic-Aware Text Chunking
==================================================
Receives: list[DocumentPage] from ingestion/OCR
Action:   Cleans text and splits into semantic chunks
Returns:  list[TextChunk] ready for embedding

Improvements over basic chunking:
- Respects paragraph and section boundaries
- Keeps lists and bullet points together
- Sentence-boundary aware splitting
- Preserves context with smart overlap
"""

import re
import logging
from dataclasses import dataclass
from pathlib import Path
import sys

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S"
    )


@dataclass
class TextChunk:
    """A piece of text ready for embedding."""
    text: str
    source: str
    page: int
    chunk_id: int


def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not text:
        return ""
    # Fix broken hyphenation at line breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Normalize whitespace but preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
    text = re.sub(r"[ \t]+", " ", text)      # Collapse spaces
    return text.strip()


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double newline separated)."""
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common edge cases."""
    # Handle abbreviations and decimals
    text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)  # Preserve decimals
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|etc|vs|e\.g|i\.e)\.", r"\1<DOT>", text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore dots
    sentences = [s.replace("<DOT>", ".") for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]


def _is_list_item(text: str) -> bool:
    """Check if text starts with a list marker."""
    patterns = [
        r"^\s*[\-\*\•]\s",           # Bullet points
        r"^\s*\d+[\.\)]\s",          # Numbered lists
        r"^\s*[a-zA-Z][\.\)]\s",     # Lettered lists
        r"^\s*\([a-zA-Z0-9]+\)\s",   # Parenthetical lists
    ]
    return any(re.match(p, text) for p in patterns)


def _find_list_block(paragraphs: list[str], start_idx: int) -> int:
    """Find the end index of a list block starting at start_idx."""
    if start_idx >= len(paragraphs):
        return start_idx
    
    if not _is_list_item(paragraphs[start_idx]):
        return start_idx
    
    end_idx = start_idx
    for i in range(start_idx, len(paragraphs)):
        if _is_list_item(paragraphs[i]):
            end_idx = i
        else:
            break
    
    return end_idx + 1


def semantic_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_size: int = 100,
) -> list[str]:
    """
    Split text into semantic chunks.
    
    Strategy:
    1. Split into paragraphs first
    2. Keep list items together
    3. Split large paragraphs at sentence boundaries
    4. Merge small chunks with neighbors
    5. Add overlap from previous chunk for context
    """
    if not text or not text.strip():
        return []
    
    text = clean_text(text)
    paragraphs = _split_into_paragraphs(text)
    
    if not paragraphs:
        return []
    
    # Phase 1: Group paragraphs semantically
    semantic_blocks = []
    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        
        # Check if this starts a list — keep list together
        if _is_list_item(para):
            list_end = _find_list_block(paragraphs, i)
            list_block = "\n".join(paragraphs[i:list_end])
            semantic_blocks.append(list_block)
            i = list_end
        else:
            semantic_blocks.append(para)
            i += 1
    
    # Phase 2: Split large blocks, merge small ones
    chunks = []
    current_chunk = ""
    
    for block in semantic_blocks:
        block_len = len(block)
        current_len = len(current_chunk)
        
        # If block fits in current chunk
        if current_len + block_len + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + block
            else:
                current_chunk = block
        
        # If block is too large, split it
        elif block_len > chunk_size:
            # Save current chunk first
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # Split large block by sentences
            sentences = _split_into_sentences(block)
            sentence_chunk = ""
            
            for sentence in sentences:
                if len(sentence_chunk) + len(sentence) + 1 <= chunk_size:
                    sentence_chunk = (sentence_chunk + " " + sentence).strip()
                else:
                    if sentence_chunk and len(sentence_chunk) >= min_chunk_size:
                        chunks.append(sentence_chunk)
                    sentence_chunk = sentence
            
            # Handle remaining sentences
            if sentence_chunk:
                current_chunk = sentence_chunk
        
        # Block doesn't fit, start new chunk
        else:
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk)
            current_chunk = block
    
    # Don't forget the last chunk
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk)
    
    # Phase 3: Add overlap for context continuity
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # Get last N characters from previous chunk
            overlap_text = prev_chunk[-chunk_overlap:] if len(prev_chunk) > chunk_overlap else prev_chunk
            
            # Find a clean break point (sentence or word boundary)
            space_idx = overlap_text.find(" ")
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            
            # Prepend overlap
            if overlap_text and not curr_chunk.startswith(overlap_text[:20]):
                curr_chunk = f"...{overlap_text}\n\n{curr_chunk}"
            
            overlapped_chunks.append(curr_chunk)
        
        chunks = overlapped_chunks
    
    log.debug(f"Semantic chunking: {len(paragraphs)} paragraphs → {len(chunks)} chunks")
    return chunks


def process_pages(pages: list) -> list[TextChunk]:
    """
    Main pipeline: Clean and chunk all pages.
    """
    log.info(f"Processing {len(pages)} pages (semantic chunking)...")
    
    chunk_size = CONFIG["text_processing"].get("chunk_size", 512)
    chunk_overlap = CONFIG["text_processing"].get("chunk_overlap", 64)
    
    all_chunks: list[TextChunk] = []
    chunk_counter = 0
    
    for page in pages:
        if not page.text:
            continue
        
        text_pieces = semantic_chunk(
            page.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        for piece in text_pieces:
            all_chunks.append(TextChunk(
                text=piece,
                source=page.source,
                page=page.page,
                chunk_id=chunk_counter,
            ))
            chunk_counter += 1
    
    log.info(f"Created {len(all_chunks)} semantic chunks (size={chunk_size}, overlap={chunk_overlap})")
    return all_chunks


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test semantic chunking
    test_text = """
Machine Learning Fundamentals

Machine learning is a subset of artificial intelligence that enables systems to learn from data.

There are three main types of machine learning:
- Supervised learning: Uses labeled training data
- Unsupervised learning: Finds patterns in unlabeled data
- Reinforcement learning: Learns through trial and error

Key Concepts

Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information.

Deep learning is a subset of machine learning that uses neural networks with many layers. It has revolutionized fields like computer vision and natural language processing.
"""
    
    chunks = semantic_chunk(test_text, chunk_size=300, chunk_overlap=50)
    
    print(f"\n── Semantic Chunking Test ──")
    print(f"Input: {len(test_text)} chars")
    print(f"Output: {len(chunks)} chunks\n")
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
        print()