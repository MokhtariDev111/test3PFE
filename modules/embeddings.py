"""
embeddings.py  —  Step 5: Embeddings & Vector DB
================================================
Receives: list[TextChunk] from Step 4.
Action:   Converts chunk text into high-dimensional vectors (embeddings)
          using a HuggingFace model, and stores them in a FAISS index.
Returns:  The path to the saved FAISS index and the chunk datastore.
"""

import os
import pickle
import logging
from pathlib import Path
import numpy as np
import sys

# Setup imports so it can run standalone
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from modules.text_processing import TextChunk
except ImportError:
    pass

from modules.config_loader import CONFIG

log = logging.getLogger("embeddings")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


class VectorDB:
    def __init__(self):
        self.model_name = CONFIG["embeddings"]["model"]
        self.device = CONFIG["embeddings"]["device"]
        self.index_path = Path(CONFIG["vector_db"]["index_path"])
        
        self.model = None
        self.index = None
        self.chunks_store: list[TextChunk] = []

    def _load_model(self):
        if self.model is None:
            log.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
            from sentence_transformers import SentenceTransformer
            # If CUDA is available, sentence-transformers will use it automatically via the device flag
            self.model = SentenceTransformer(self.model_name, device=self.device)
            log.info("  ✔ Model loaded successfully.")

    def embed_and_store(self, chunks: list[TextChunk]):
        """Creates vectors for the chunks and builds the FAISS index."""
        if not chunks:
            log.warning("No chunks to embed! Skipping.")
            return

        self._load_model()
        import faiss

        log.info(f"Embedding {len(chunks)} chunks...")
        texts = [c.text for c in chunks]
        
        # model.encode returns a numpy array of vectors
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        # FAISS requires float32
        embeddings = np.array(embeddings).astype("float32")
        
        dimension = embeddings.shape[1]
        
        # Create a simple flat (L2 distance) FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        self.chunks_store = chunks
        
        log.info(f"FAISS index built with {self.index.ntotal} vectors of dimension {dimension}.")
        self._save_to_disk()

    def _save_to_disk(self):
        """Saves the FAISS index and the corresponding chunks text metadata."""
        import faiss
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss_file = str(self.index_path.with_suffix(".index"))
        store_file = str(self.index_path.with_suffix(".pkl"))
        
        faiss.write_index(self.index, faiss_file)
        with open(store_file, "wb") as f:
            pickle.dump(self.chunks_store, f)
            
        log.info(f"Saved Vector DB to '{self.index_path.parent}'.")


def build_vector_db(chunks: list[TextChunk]):
    """Wrapper function to perform Step 5 inside the pipeline."""
    db = VectorDB()
    db.embed_and_store(chunks)

# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.ingestion import ingest_directory
    from modules.ocr import run_ocr
    from modules.text_processing import process_pages

    raw_dir = Path(CONFIG["paths"]["data_raw"])
    print("\n[Step 2] Ingesting data...")
    pages = ingest_directory(raw_dir)
    
    print("\n[Step 3] OCR Integration...")
    pages = run_ocr(pages)
    
    print("\n[Step 4] Processing text...")
    chunks = process_pages(pages)

    print("\n[Step 5] Building Vector DB...")
    build_vector_db(chunks)
    
    print("\n✅ Step 5 (Embeddings) executed successfully.")
