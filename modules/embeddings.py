"""
embeddings.py — v2: saves BM25 tokenized corpus alongside FAISS index
======================================================================
Changes from v1:
  1. After building the FAISS index, saves a tokenized BM25 corpus to
     <index_path>.bm25.json so retrieval.py can load it without re-processing.
  2. SHA-256 cache and semantic deduplication unchanged.
  3. All existing API (build_vector_db, VectorDB) unchanged — no breaking changes.
"""

import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.config_loader import CONFIG

log = logging.getLogger("embeddings")
if not log.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

_GLOBAL_EMBED_MODEL = None

# In-process index cache  { content_hash: (faiss_index, chunks_list) }
_INDEX_CACHE: dict = {}


def _chunks_hash(chunks: list) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.text.encode("utf-8"))
    return h.hexdigest()


class VectorDB:
    def __init__(self, index_path=None):
        self.model_name = CONFIG["embeddings"]["model"]
        self.device     = CONFIG["embeddings"]["device"]
        self.index_path = Path(index_path) if index_path else Path(CONFIG["vector_db"]["index_path"])
        self.model      = None
        self.index      = None
        self.chunks_store: list = []

    def _load_model(self):
        global _GLOBAL_EMBED_MODEL
        if _GLOBAL_EMBED_MODEL is None:
            log.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
            from sentence_transformers import SentenceTransformer
            _GLOBAL_EMBED_MODEL = SentenceTransformer(self.model_name, device=self.device)
            log.info("  ✔ Embedding model loaded.")
        self.model = _GLOBAL_EMBED_MODEL

    def embed_and_store(self, chunks: list):
        if not chunks:
            log.warning("No chunks to embed — skipping.")
            return

        import faiss

        # Cache check
        key = _chunks_hash(chunks)
        if key in _INDEX_CACHE:
            log.info("  ✔ FAISS index cache HIT — skipping rebuild.")
            cached_index, cached_chunks = _INDEX_CACHE[key]
            self.index        = cached_index
            self.chunks_store = cached_chunks
            if not self.index_path.with_suffix(".index").exists():
                self._save_to_disk()
            return

        log.info(f"  FAISS index cache MISS — building for {len(chunks)} chunks...")
        self._load_model()

        texts      = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)

        # Semantic deduplication (skip O(n²) for large sets)
        if len(embeddings) <= 500:
            sim_matrix   = np.dot(embeddings, embeddings.T)
            keep_indices = []
            for i in range(len(embeddings)):
                if not keep_indices:
                    keep_indices.append(i)
                    continue
                if np.max(sim_matrix[i, keep_indices]) < 0.95:
                    keep_indices.append(i)
            dropped = len(embeddings) - len(keep_indices)
            if dropped:
                log.info(f"  Deduplication dropped {dropped} near-duplicate chunks.")
                embeddings = embeddings[keep_indices]
                chunks     = [chunks[i] for i in keep_indices]

        dim        = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks_store = chunks

        _INDEX_CACHE[key] = (self.index, self.chunks_store)
        log.info(f"  ✔ FAISS index built ({self.index.ntotal} vectors) and cached.")
        self._save_to_disk()

    def _save_to_disk(self):
        import faiss
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS binary index
        faiss.write_index(self.index, str(self.index_path.with_suffix(".index")))

        # Save chunk metadata (text + source + page)
        chunk_dicts = [c.__dict__ for c in self.chunks_store]
        with open(self.index_path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, ensure_ascii=False)

        # NEW: Save BM25 tokenized corpus so retrieval.py can build BM25 without
        # re-loading the embedding model or re-reading the original text.
        # Each entry is a list of lowercase tokens from the chunk text.
        bm25_corpus = [c.text.lower().split() for c in self.chunks_store]
        bm25_path   = self.index_path.with_suffix(".bm25.json")
        with open(bm25_path, "w", encoding="utf-8") as f:
            json.dump(bm25_corpus, f, ensure_ascii=False)

        log.info(f"  Saved Vector DB (FAISS + BM25 corpus) to '{self.index_path.parent}'.")


def build_vector_db(chunks: list, index_path=None):
    db = VectorDB(index_path=index_path)
    db.embed_and_store(chunks)
