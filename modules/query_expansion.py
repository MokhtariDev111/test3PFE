"""
query_expansion.py — Multi-Query Retrieval
==========================================
Generates multiple search queries from a single user query
to improve retrieval coverage and accuracy.
"""

import json
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

log = logging.getLogger("query_expansion")


def expand_query_simple(query: str) -> list[str]:
    """
    Rule-based query expansion without LLM.
    Fast fallback that adds common query variations.
    """
    query_lower = query.lower().strip()
    expansions = [query]  # Always include original
    
    # Add definition-style query
    if not query_lower.startswith(("what is", "define", "explain")):
        expansions.append(f"what is {query}")
    
    # Add example-style query
    if "example" not in query_lower:
        expansions.append(f"examples of {query}")
    
    # Add concept-style query
    expansions.append(f"{query} concept explanation")
    
    return expansions


async def expand_query_llm(query: str, llm_engine) -> list[str]:
    """
    LLM-powered query expansion.
    Generates semantically diverse queries for better retrieval coverage.
    """
    prompt = f"""You are a search query optimizer. Generate 3 alternative search queries for the given topic.
Each query should approach the topic from a different angle to maximize retrieval coverage.

Original query: "{query}"

Rules:
1. Include synonyms and related technical terms
2. One query should be more specific/technical
3. One query should be broader/conceptual
4. Keep queries concise (3-8 words each)

Return ONLY a JSON object:
{{"queries": ["query1", "query2", "query3"]}}"""

    try:
        raw = await llm_engine.generate_async(
            query, 
            [], 
            prompt_override=prompt
        )
        data = json.loads(raw)
        expanded = data.get("queries", [])
        
        # Always include original query first
        result = [query] + [q for q in expanded if q.lower() != query.lower()]
        log.info(f"Query expanded: '{query}' → {len(result)} queries")
        return result[:4]  # Cap at 4 queries max
        
    except Exception as e:
        log.warning(f"LLM query expansion failed ({e}), using simple expansion")
        return expand_query_simple(query)


def deduplicate_results(all_results: list, top_k: int = 10) -> list:
    """
    Merge and deduplicate results from multiple queries.
    Preserves order (first occurrence wins) and limits to top_k.
    """
    seen_ids = set()
    unique_results = []
    
    for result in all_results:
        # Use chunk_id if available, otherwise hash the text
        chunk_id = getattr(result, 'chunk_id', None) or hash(result.text[:100])
        
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_results.append(result)
            
            if len(unique_results) >= top_k:
                break
    
    return unique_results