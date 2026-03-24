"""
health.py — System Health & Status Dashboard
==============================================
Provides comprehensive system status for monitoring and debugging.
"""

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.config_loader import CONFIG

log = logging.getLogger("health")


def check_faiss_index() -> dict:
    """Check FAISS index status."""
    index_path = Path(CONFIG["vector_db"]["index_path"])
    faiss_file = index_path.with_suffix(".index")
    json_file = index_path.with_suffix(".json")
    bm25_file = index_path.with_suffix(".bm25.json")
    
    result = {
        "status": "not_found",
        "vectors": 0,
        "chunks": 0,
        "bm25_ready": False,
        "path": str(index_path.parent),
    }
    
    if faiss_file.exists() and json_file.exists():
        try:
            import faiss
            import json
            
            index = faiss.read_index(str(faiss_file))
            result["vectors"] = index.ntotal
            
            with open(json_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            result["chunks"] = len(chunks)
            
            result["bm25_ready"] = bm25_file.exists()
            result["status"] = "ready"
            
        except Exception as e:
            result["status"] = f"error: {e}"
    
    return result


def check_embedding_model() -> dict:
    """Check embedding model status."""
    model_name = CONFIG["embeddings"]["model"]
    device = CONFIG["embeddings"]["device"]
    
    result = {
        "model": model_name,
        "device": device,
        "status": "not_loaded",
        "dimensions": 0,
    }
    
    try:
        from modules.embeddings import _GLOBAL_EMBED_MODEL
        if _GLOBAL_EMBED_MODEL is not None:
            result["status"] = "loaded"
            result["dimensions"] = _GLOBAL_EMBED_MODEL.get_sentence_embedding_dimension()
        else:
            result["status"] = "not_loaded"
    except Exception as e:
        result["status"] = f"error: {e}"
    
    return result


def check_llm_backend() -> dict:
    """Check LLM backend status."""
    result = {
        "backend": "unknown",
        "model": "",
        "status": "unknown",
        "groq_configured": False,
        "ollama_url": CONFIG["llm"]["api_url"],
    }
    
    try:
        groq_key = os.getenv("GROQ_API_KEY", "")
        result["groq_configured"] = bool(groq_key)
        
        if groq_key:
            result["backend"] = "groq"
            result["model"] = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            result["status"] = "configured"
        else:
            result["backend"] = "ollama"
            result["model"] = CONFIG["llm"]["model"]
            
            # Quick Ollama connectivity check
            import httpx
            try:
                resp = httpx.get(
                    CONFIG["llm"]["api_url"].replace("/api/generate", "/api/tags"),
                    timeout=2.0
                )
                if resp.status_code == 200:
                    result["status"] = "connected"
                else:
                    result["status"] = "error"
            except:
                result["status"] = "unreachable"
                
    except Exception as e:
        result["status"] = f"error: {e}"
    
    return result


def check_reranker() -> dict:
    """Check reranker model status."""
    cfg = CONFIG.get("retrieval", {})
    enabled = cfg.get("use_reranker", False)
    model_name = cfg.get("reranker_model", "")
    
    result = {
        "enabled": enabled,
        "model": model_name,
        "status": "disabled" if not enabled else "not_loaded",
    }
    
    if enabled:
        try:
            from modules.retrieval import _GLOBAL_RERANKER
            if _GLOBAL_RERANKER is not None:
                result["status"] = "loaded"
        except:
            pass
    
    return result


def check_cache() -> dict:
    """Check cache status."""
    try:
        from modules.llm_cache import cache_stats
        stats = cache_stats()
        stats["status"] = "active"
        return stats
    except ImportError:
        return {"status": "not_configured"}
    except Exception as e:
        return {"status": f"error: {e}"}


def check_history() -> dict:
    """Check presentation history."""
    try:
        from modules.history_store import load_history, HISTORY_FILE
        
        history = load_history()
        
        return {
            "status": "ready",
            "total_presentations": len(history),
            "file_exists": HISTORY_FILE.exists(),
            "path": str(HISTORY_FILE),
        }
    except Exception as e:
        return {"status": f"error: {e}"}


def check_ocr() -> dict:
    """Check OCR engine status."""
    engine = CONFIG["ocr"]["engine"]
    languages = CONFIG["ocr"]["languages"]
    
    result = {
        "engine": engine,
        "languages": languages,
        "status": "not_loaded",
    }
    
    try:
        from modules.ocr import _EASYOCR_CACHE
        if _EASYOCR_CACHE is not None:
            result["status"] = "loaded"
        else:
            result["status"] = "not_loaded"
    except:
        pass
    
    return result


def get_system_info() -> dict:
    """Get system information."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
        cuda_memory = None
        if cuda_available:
            mem = torch.cuda.get_device_properties(0).total_memory
            cuda_memory = f"{mem / 1024**3:.1f} GB"
    except ImportError:
        cuda_available = False
        cuda_device = None
        cuda_memory = None
    
    return {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "cuda_memory": cuda_memory,
    }


def get_config_summary() -> dict:
    """Get key configuration values."""
    return {
        "embedding_model": CONFIG["embeddings"]["model"],
        "embedding_device": CONFIG["embeddings"]["device"],
        "chunk_size": CONFIG["text_processing"]["chunk_size"],
        "chunk_overlap": CONFIG["text_processing"]["chunk_overlap"],
        "llm_model": CONFIG["llm"]["model"],
        "use_bm25": CONFIG.get("retrieval", {}).get("use_bm25", False),
        "use_reranker": CONFIG.get("retrieval", {}).get("use_reranker", False),
        "judge_enabled": CONFIG["llm"].get("judge_enabled", False),
    }


def full_health_check() -> dict:
    """
    Comprehensive system health check.
    Returns status of all components.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",  # Will be updated based on checks
        "version": CONFIG.get("project", {}).get("version", "unknown"),
        "project": CONFIG.get("project", {}).get("name", "AI Teaching Assistant"),
        
        "system": get_system_info(),
        "config": get_config_summary(),
        
        "components": {
            "faiss_index": check_faiss_index(),
            "embedding_model": check_embedding_model(),
            "llm_backend": check_llm_backend(),
            "reranker": check_reranker(),
            "cache": check_cache(),
            "history": check_history(),
            "ocr": check_ocr(),
        },
    }


# ── Quick Status (for frequent polling) ──────────────────────────────────────
def quick_status() -> dict:
    """Lightweight status check for frequent polling."""
    faiss = check_faiss_index()
    llm = check_llm_backend()
    
    all_ok = (
        faiss["status"] == "ready" and
        llm["status"] in ("configured", "connected")
    )
    
    return {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "index_vectors": faiss["vectors"],
        "llm_backend": llm["backend"],
    }


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    
    print("── System Health Check ──\n")
    health = full_health_check()
    print(json.dumps(health, indent=2, default=str))