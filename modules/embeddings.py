"""
embeddings.py  —  Step 5: Embeddings & Vector DB
================================================
Receives: list[TextChunk] from Step 4.
Action:   Converts chunk text into high-dimensional vectors (embeddings)
          using a HuggingFace model, and stores them in a FAISS index.
Returns:  The path to the saved FAISS index and the chunk datastore.
"""

import os
import json
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
    def __init__(self, index_path: str | Path = None):
        self.model_name = CONFIG["embeddings"]["model"]
        self.device = CONFIG["embeddings"]["device"]
        # Use provided path (for isolated API requests) or fallback to global config
        self.index_path = Path(index_path) if index_path else Path(CONFIG["vector_db"]["index_path"])
        
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
        faiss.normalize_L2(embeddings)
        
        # --- IDEA 7: Semantic Deduplication ---
        # Calculate cosine similarity of all pairs (since vectors are normalized L2, dot product is cosine sim)
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        keep_indices = []
        dropped_count = 0
        for i in range(len(embeddings)):
            if not keep_indices:
                keep_indices.append(i)
                continue
                
            # Check similarity against ALREADY KEPT chunks only
            max_sim_to_kept = np.max(sim_matrix[i, keep_indices])
            if max_sim_to_kept < 0.95:  # Threshold for deduplication
                keep_indices.append(i)
            else:
                dropped_count += 1
                
        if dropped_count > 0:
            log.info(f"Semantic Deduplication dropped {dropped_count} near-duplicate chunks.")
            embeddings = embeddings[keep_indices]
            chunks = [chunks[i] for i in keep_indices]
        # --- End Idea 7 ---
        
        dimension = embeddings.shape[1]
        
        # Create an inner product (cosine similarity) FAISS index
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        self.chunks_store = chunks
        
        log.info(f"FAISS index built with {self.index.ntotal} vectors of dimension {dimension}.")
        self._save_to_disk()

    def _save_to_disk(self):
        """Saves the FAISS index and the corresponding chunks text metadata."""
        import faiss
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss_file = str(self.index_path.with_suffix(".index"))
        store_file = str(self.index_path.with_suffix(".json"))
        
        faiss.write_index(self.index, faiss_file)
        
        # Serialize chunks using JSON (safe) instead of generic Pickle (insecure)
        chunk_dicts = [c.__dict__ for c in self.chunks_store]
        with open(store_file, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, ensure_ascii=False)
            
        log.info(f"Saved Vector DB to '{self.index_path.parent}'.")


def build_vector_db(chunks: list[TextChunk], index_path: str | Path = None):
    """Wrapper function to perform Step 5 inside the pipeline."""
    db = VectorDB(index_path=index_path)
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
