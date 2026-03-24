"""Test LLM caching functionality."""
from modules.llm_cache import get_cached, set_cached, clear_cache, cache_stats

# Clear any existing cache
clear_cache()

# Test cache miss
result = get_cached("test prompt", "test-model")
assert result is None, "Should be cache miss"
print("✅ Cache miss works")

# Test cache set + hit
set_cached("test prompt", "test-model", "This is a test response")
result = get_cached("test prompt", "test-model")
assert result == "This is a test response", "Should return cached response"
print("✅ Cache hit works")

# Test different prompt = miss
result = get_cached("different prompt", "test-model")
assert result is None, "Different prompt should miss"
print("✅ Cache key isolation works")

# Test stats
stats = cache_stats()
print(f"Cache stats: {stats}")
assert stats["memory_entries"] >= 1, "Should have memory entries"
print("✅ Cache stats work")

# Cleanup
clear_cache()
print("\n✅ All cache tests passed!")