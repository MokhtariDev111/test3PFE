"""
llm_cache.py — LLM Response Caching
====================================
Two-level caching for LLM responses:
- L1: In-memory (fast, session-scoped)
- L2: Disk-based (persistent across restarts)

Cache key = hash(model + prompt)
TTL = 24 hours for disk cache
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional
import threading

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "data" / "cache" / "llm"

log = logging.getLogger("llm_cache")

# L1: In-memory cache (thread-safe)
_memory_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()

# Configuration
MAX_MEMORY_ENTRIES = 200  # Limit memory usage
DISK_TTL_SECONDS = 24 * 60 * 60  # 24 hours


def _cache_key(prompt: str, model: str) -> str:
    """Generate a unique cache key from prompt + model."""
    content = f"{model}::{prompt}"
    return hashlib.sha256(content.encode()).hexdigest()[:32]


def _disk_path(key: str) -> Path:
    """Get disk cache file path for a key."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def get_cached(prompt: str, model: str) -> Optional[str]:
    """
    Retrieve cached LLM response if available.
    Checks L1 (memory) first, then L2 (disk).
    
    Returns:
        Cached response string, or None if not found/expired
    """
    key = _cache_key(prompt, model)
    
    # L1: Check memory cache
    with _cache_lock:
        if key in _memory_cache:
            entry = _memory_cache[key]
            log.debug(f"Cache HIT (memory): {key[:8]}...")
            return entry["response"]
    
    # L2: Check disk cache
    disk_file = _disk_path(key)
    if disk_file.exists():
        try:
            data = json.loads(disk_file.read_text(encoding="utf-8"))
            
            # Check TTL
            age = time.time() - data.get("timestamp", 0)
            if age < DISK_TTL_SECONDS:
                response = data["response"]
                
                # Promote to memory cache
                with _cache_lock:
                    _memory_cache[key] = {"response": response}
                
                log.debug(f"Cache HIT (disk): {key[:8]}...")
                return response
            else:
                # Expired — delete
                disk_file.unlink(missing_ok=True)
                log.debug(f"Cache EXPIRED: {key[:8]}...")
                
        except (json.JSONDecodeError, KeyError) as e:
            log.warning(f"Cache read error: {e}")
            disk_file.unlink(missing_ok=True)
    
    log.debug(f"Cache MISS: {key[:8]}...")
    return None


def set_cached(prompt: str, model: str, response: str) -> None:
    """
    Store LLM response in both memory and disk cache.
    
    Args:
        prompt: The input prompt
        model: Model identifier
        response: LLM response to cache
    """
    if not response or len(response) < 10:
        return  # Don't cache empty/error responses
    
    key = _cache_key(prompt, model)
    
    # L1: Store in memory
    with _cache_lock:
        # Evict oldest if at capacity
        if len(_memory_cache) >= MAX_MEMORY_ENTRIES:
            oldest_key = next(iter(_memory_cache))
            del _memory_cache[oldest_key]
        
        _memory_cache[key] = {"response": response}
    
    # L2: Store on disk
    try:
        disk_file = _disk_path(key)
        data = {
            "prompt_preview": prompt[:100],  # For debugging
            "model": model,
            "response": response,
            "timestamp": time.time(),
        }
        disk_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        log.debug(f"Cached response: {key[:8]}... ({len(response)} chars)")
        
    except Exception as e:
        log.warning(f"Cache write error: {e}")


def clear_cache() -> int:
    """
    Clear all cached responses (memory + disk).
    
    Returns:
        Number of entries cleared
    """
    count = 0
    
    # Clear memory
    with _cache_lock:
        count += len(_memory_cache)
        _memory_cache.clear()
    
    # Clear disk
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
            count += 1
    
    log.info(f"Cache cleared: {count} entries removed")
    return count


def cache_stats() -> dict:
    """Get cache statistics."""
    memory_count = len(_memory_cache)
    disk_count = len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0
    
    return {
        "memory_entries": memory_count,
        "disk_entries": disk_count,
        "memory_limit": MAX_MEMORY_ENTRIES,
        "disk_ttl_hours": DISK_TTL_SECONDS // 3600,
    }