"""
evaluation.py — RAG Pipeline Evaluation Metrics
=================================================
Provides quantitative metrics to measure RAG quality:
- Retrieval relevance (semantic similarity)
- Answer groundedness (is answer based on context?)
- Chunk utilization (how much context is used?)
- Diversity score (are results varied?)

Usage:
    from modules.evaluation import RAGEvaluator
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate_retrieval(query, chunks)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.config_loader import CONFIG

log = logging.getLogger("evaluation")

# Cache embedding model
_EVAL_MODEL = None


def _get_eval_model():
    """Load embedding model for evaluation (cached)."""
    global _EVAL_MODEL
    if _EVAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        model_name = CONFIG["embeddings"]["model"]
        device = CONFIG["embeddings"]["device"]
        log.info(f"Loading evaluation model: {model_name}")
        _EVAL_MODEL = SentenceTransformer(model_name, device=device)
    return _EVAL_MODEL


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    query: str
    num_chunks: int
    avg_relevance: float      # Average semantic similarity to query
    max_relevance: float      # Best chunk similarity
    min_relevance: float      # Worst chunk similarity
    diversity_score: float    # How diverse are the chunks (0-1)
    coverage_estimate: float  # Estimated topic coverage
    
    def to_dict(self) -> dict:
        return {
            "query": self.query[:50],
            "num_chunks": self.num_chunks,
            "avg_relevance": round(self.avg_relevance, 3),
            "max_relevance": round(self.max_relevance, 3),
            "min_relevance": round(self.min_relevance, 3),
            "diversity_score": round(self.diversity_score, 3),
            "coverage_estimate": round(self.coverage_estimate, 3),
        }
    
    def summary(self) -> str:
        grade = "🟢" if self.avg_relevance > 0.5 else "🟡" if self.avg_relevance > 0.3 else "🔴"
        return (
            f"{grade} Retrieval: {self.avg_relevance:.2f} avg relevance | "
            f"{self.diversity_score:.2f} diversity | "
            f"{self.num_chunks} chunks"
        )


@dataclass 
class GenerationMetrics:
    """Metrics for generation quality."""
    groundedness: float       # How much of answer is based on context
    relevance: float          # How relevant is answer to query
    completeness: float       # Estimated completeness
    
    def to_dict(self) -> dict:
        return {
            "groundedness": round(self.groundedness, 3),
            "relevance": round(self.relevance, 3),
            "completeness": round(self.completeness, 3),
        }
    
    def summary(self) -> str:
        avg = (self.groundedness + self.relevance + self.completeness) / 3
        grade = "🟢" if avg > 0.6 else "🟡" if avg > 0.4 else "🔴"
        return (
            f"{grade} Generation: {self.groundedness:.2f} grounded | "
            f"{self.relevance:.2f} relevant | "
            f"{self.completeness:.2f} complete"
        )


class RAGEvaluator:
    """
    Evaluates RAG pipeline quality with quantitative metrics.
    """
    
    def __init__(self):
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            self.model = _get_eval_model()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def _pairwise_diversity(self, embeddings: np.ndarray) -> float:
        """
        Calculate diversity as 1 - average pairwise similarity.
        Higher = more diverse chunks.
        """
        n = len(embeddings)
        if n < 2:
            return 1.0
        
        total_sim = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += self._cosine_similarity(embeddings[i], embeddings[j])
                count += 1
        
        avg_similarity = total_sim / count if count > 0 else 0
        return 1.0 - avg_similarity  # Invert: high similarity = low diversity
    
    def evaluate_retrieval(
        self,
        query: str,
        chunks: list,
        return_details: bool = False,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.
        
        Args:
            query: The search query
            chunks: List of retrieved TextChunk objects
            return_details: Include per-chunk scores
        
        Returns:
            RetrievalMetrics with quality scores
        """
        self._load_model()
        
        if not chunks:
            return RetrievalMetrics(
                query=query,
                num_chunks=0,
                avg_relevance=0.0,
                max_relevance=0.0,
                min_relevance=0.0,
                diversity_score=0.0,
                coverage_estimate=0.0,
            )
        
        # Get chunk texts
        chunk_texts = [getattr(c, 'text', str(c)) for c in chunks]
        
        # Embed query and chunks
        query_emb = self.model.encode(query, convert_to_numpy=True)
        chunk_embs = self.model.encode(chunk_texts, convert_to_numpy=True)
        
        # Calculate relevance scores
        relevance_scores = [
            self._cosine_similarity(query_emb, chunk_emb)
            for chunk_emb in chunk_embs
        ]
        
        # Calculate diversity
        diversity = self._pairwise_diversity(chunk_embs)
        
        # Estimate coverage (unique information density)
        total_chars = sum(len(t) for t in chunk_texts)
        unique_words = len(set(" ".join(chunk_texts).lower().split()))
        coverage = min(1.0, unique_words / 200)  # Normalize to ~200 unique words = full coverage
        
        metrics = RetrievalMetrics(
            query=query,
            num_chunks=len(chunks),
            avg_relevance=float(np.mean(relevance_scores)),
            max_relevance=float(np.max(relevance_scores)),
            min_relevance=float(np.min(relevance_scores)),
            diversity_score=diversity,
            coverage_estimate=coverage,
        )
        
        log.info(f"Retrieval eval: {metrics.summary()}")
        return metrics
    
    def evaluate_generation(
        self,
        query: str,
        context: str,
        generated_text: str,
    ) -> GenerationMetrics:
        """
        Evaluate generation quality.
        
        Args:
            query: Original query
            context: Retrieved context used for generation
            generated_text: The generated output (slide content)
        
        Returns:
            GenerationMetrics with quality scores
        """
        self._load_model()
        
        if not generated_text:
            return GenerationMetrics(
                groundedness=0.0,
                relevance=0.0,
                completeness=0.0,
            )
        
        # Embed all texts
        query_emb = self.model.encode(query, convert_to_numpy=True)
        context_emb = self.model.encode(context[:2000], convert_to_numpy=True)  # Truncate long context
        generated_emb = self.model.encode(generated_text, convert_to_numpy=True)
        
        # Groundedness: similarity between generated and context
        groundedness = self._cosine_similarity(generated_emb, context_emb)
        
        # Relevance: similarity between generated and query
        relevance = self._cosine_similarity(generated_emb, query_emb)
        
        # Completeness: rough estimate based on length and query coverage
        query_words = set(query.lower().split())
        generated_words = set(generated_text.lower().split())
        word_coverage = len(query_words & generated_words) / len(query_words) if query_words else 0
        length_score = min(1.0, len(generated_text) / 500)  # Expect ~500 chars
        completeness = (word_coverage + length_score) / 2
        
        metrics = GenerationMetrics(
            groundedness=groundedness,
            relevance=relevance,
            completeness=completeness,
        )
        
        log.info(f"Generation eval: {metrics.summary()}")
        return metrics
    
    def evaluate_full_pipeline(
        self,
        query: str,
        chunks: list,
        generated_text: str,
    ) -> dict:
        """
        Full pipeline evaluation combining retrieval and generation.
        
        Returns:
            Combined metrics dictionary
        """
        context = " ".join(getattr(c, 'text', str(c)) for c in chunks)
        
        retrieval = self.evaluate_retrieval(query, chunks)
        generation = self.evaluate_generation(query, context, generated_text)
        
        # Overall score (weighted average)
        overall = (
            retrieval.avg_relevance * 0.3 +
            retrieval.diversity_score * 0.1 +
            generation.groundedness * 0.3 +
            generation.relevance * 0.2 +
            generation.completeness * 0.1
        )
        
        return {
            "query": query[:50],
            "retrieval": retrieval.to_dict(),
            "generation": generation.to_dict(),
            "overall_score": round(overall, 3),
            "grade": "A" if overall > 0.6 else "B" if overall > 0.45 else "C" if overall > 0.3 else "D",
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.retrieval import Retriever
    
    print("\n── RAG Evaluation Test ──\n")
    
    evaluator = RAGEvaluator()
    retriever = Retriever()
    
    # Test query
    query = "What is machine learning?"
    
    # Get chunks
    chunks = retriever.search(query, top_k=5)
    
    if chunks:
        # Evaluate retrieval
        ret_metrics = evaluator.evaluate_retrieval(query, chunks)
        print(f"Retrieval: {ret_metrics.summary()}")
        print(f"  Details: {ret_metrics.to_dict()}\n")
        
        # Simulate generation (use chunk text as "generated" for testing)
        fake_generated = " ".join(c.text[:100] for c in chunks[:2])
        gen_metrics = evaluator.evaluate_generation(
            query, 
            " ".join(c.text for c in chunks),
            fake_generated
        )
        print(f"Generation: {gen_metrics.summary()}")
        print(f"  Details: {gen_metrics.to_dict()}\n")
        
        # Full evaluation
        full = evaluator.evaluate_full_pipeline(query, chunks, fake_generated)
        print(f"Overall Grade: {full['grade']} ({full['overall_score']:.2f})")
    else:
        print("No chunks found. Make sure you have indexed some documents.")
        print("Run: python -m modules.embeddings")