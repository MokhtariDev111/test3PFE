"""
retrieval.py  —  Step 6: Retrieval / RAG Core
=============================================
Receives: A user query string.
Action:   Embeds the query using the same SentenceTransformer model, 
          searches the FAISS index on disk, and retrieves the matching chunks.
Returns:  list[TextChunk] containing the top `k` most relevant chunks.
"""

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

log = logging.getLogger("retrieval")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


class Retriever:
    def __init__(self, index_path: str | Path = None):
        self.model_name = CONFIG["embeddings"]["model"]
        self.device = CONFIG["embeddings"]["device"]
        self.index_path = Path(index_path) if index_path else Path(CONFIG["vector_db"]["index_path"])
        
        self.model = None
        self.index = None
        self.chunks_store: list[TextChunk] = []
        
        self._load_db()

    def _load_db(self):
        import faiss
        faiss_file = str(self.index_path.with_suffix(".index"))
        store_file = str(self.index_path.with_suffix(".json"))
        
        if not Path(faiss_file).exists() or not Path(store_file).exists():
            log.warning(f"Vector DB files not found at {self.index_path.parent}. Has Step 5 run?")
            return
            
        self.index = faiss.read_index(faiss_file)
        with open(store_file, "r", encoding="utf-8") as f:
            chunk_dicts = json.load(f)
            self.chunks_store = [TextChunk(**d) for d in chunk_dicts]
            
        log.info(f"Loaded Vector DB mappings: {self.index.ntotal} vectors.")

    def _load_model(self):
        if self.model is None:
            log.info(f"Loading embedding model for retrieval: '{self.model_name}'...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def search(self, query: str, top_k: int = 5) -> list[TextChunk]:
        """Searches the FAISS index for the query."""
        if not self.index:
            log.error("Cannot search: FAISS index is not loaded.")
            return []
            
        self._load_model()
        
        # Embed the query
        query_vector = self.model.encode([query], convert_to_numpy=True)
        query_vector = np.array(query_vector).astype("float32")
        
        import faiss
        faiss.normalize_L2(query_vector)
        
        # Search FAISS (now uses dot product / cosine similarity)
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks_store):
                results.append(self.chunks_store[idx])
                
        log.info(f"Retrieved {len(results)} chunks for query: '{query}'")
        return results

# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = Retriever()
    query = "What is the Transformer architecture?"
    print(f"\n[Step 6] Searching for: '{query}'")
    
    results = retriever.search(query, top_k=3)
    
    print("\n── Top Results ──────────────────────────")
    for i, res in enumerate(results, 1):
        print(f"\nResult {i}: [{res.source} | p.{res.page}]")
        print(f"{res.text[:200]}...")
