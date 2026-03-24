"""
llm.py — v3: Multi-backend support (Ollama + Groq) + Batch Generation
======================================================================
Optimizations:
  1. Groq backend for 10-20x faster inference
  2. Batch generation mode - all slides in ONE call
  3. Automatic fallback: Groq → Ollama
  4. Streaming support for progress feedback
"""

import json
import asyncio
import logging
import os
from pathlib import Path
import sys
from modules.llm_cache import get_cached, set_cached
from modules.retry_utils import retry_async, RateLimitHandler, RETRYABLE_STATUS_CODES

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.config_loader import CONFIG

log = logging.getLogger("llm")
if not log.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


class LLMEngine:
    def __init__(self):
        # Ollama config (fallback)
        self.ollama_url = CONFIG["llm"]["api_url"]
        self.ollama_model = CONFIG["llm"]["model"]
        
        # Groq config (primary if available)
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_url = "https://api.groq.com/openai/v1/chat/completions"
        
        # General config
        self.temp = CONFIG["llm"]["temperature"]
        self.ctx_len = min(CONFIG["llm"]["context_length"], 4096)
        self.judge_model = CONFIG["llm"].get("judge_model", self.ollama_model)
        self.judge_enabled = CONFIG["llm"].get("judge_enabled", False)
        
        # Backend selection
        self.backend = "groq" if self.groq_api_key else "ollama"
        log.info(f"LLM Backend: {self.backend.upper()}")
        # Rate limit handler
        self._rate_limiter = RateLimitHandler(base_delay=1.0)

    # ══════════════════════════════════════════════════════════════════════════
    # GROQ BACKEND (Primary - Fast)
    # ══════════════════════════════════════════════════════════════════════════
    
    async def _call_groq(self, prompt: str, model: str = None, json_mode: bool = True) -> str:
        """Call Groq API with caching and retry logic."""
        import httpx
        
        model = model or self.groq_model
        
        # Check cache first
        cached = get_cached(prompt, model)
        if cached:
            log.info(f"  ⚡ Cache HIT for Groq ({model})")
            return cached
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert AI teaching assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temp,
            "max_tokens": 4096,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        async def _make_request() -> str:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.groq_url, json=payload, headers=headers)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = float(response.headers.get("retry-after", 2))
                    delay = await self._rate_limiter.handle_rate_limit(retry_after)
                    await asyncio.sleep(delay)
                    raise ConnectionError("Rate limited, retrying...")
                
                # Handle other retryable errors
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise ConnectionError(f"Server error {response.status_code}")
                
                response.raise_for_status()
                self._rate_limiter.reset()  # Success — reset rate limiter
                
                data = response.json()
                result = data["choices"][0]["message"]["content"].strip()
                
                # Cache successful response
                set_cached(prompt, model, result)
                
                return result
        
        try:
            return await retry_async(
                _make_request,
                max_attempts=3,
                base_delay=1.0,
            )
        except Exception as e:
            log.warning(f"Groq failed after retries ({e}) - falling back to Ollama")
            return await self._call_ollama(prompt, self.ollama_model, json_mode)

    # ══════════════════════════════════════════════════════════════════════════
    # OLLAMA BACKEND (Fallback - Local)
    # ══════════════════════════════════════════════════════════════════════════
    
    async def _call_ollama(self, prompt: str, model: str, json_mode: bool = True) -> str:
        """Call local Ollama API with caching."""
        import httpx
        import asyncio
        
        # Check cache first
        cached = get_cached(prompt, model)
        if cached:
            log.info(f"  ⚡ Cache HIT for Ollama ({model})")
            return cached
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temp,
                "num_ctx": self.ctx_len,
                "num_predict": -1,
            },
        }
        if json_mode:
            payload["format"] = "json"
        
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(self.ollama_url, json=payload)
                    response.raise_for_status()
                    result = response.json().get("response", "").strip()
                    
                    # Cache successful response
                    if result:
                        set_cached(prompt, model, result)
                    
                    return result
            
            except httpx.ConnectError as e:
                raise RuntimeError(f"Cannot reach Ollama at {self.ollama_url}") from e
            
            except httpx.HTTPStatusError as e:
                if "CUDA error" in e.response.text and attempt == 0:
                    payload["options"]["num_ctx"] = max(512, self.ctx_len // 2)
                    await asyncio.sleep(3)
                    continue
                raise
        
        return ""

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED INTERFACE
    # ══════════════════════════════════════════════════════════════════════════
    
    async def generate_async(
        self,
        query: str,
        context_chunks: list,
        prompt_override: str = None,
        model_override: str = None,
    ) -> str:
        """Main generation call - routes to best available backend."""
        if not context_chunks and not prompt_override:
            return "{}"
        
        prompt = prompt_override or self._build_prompt(query, context_chunks)
        
        if self.backend == "groq":
            model = model_override if model_override in ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama-3.1-8b-instant"] else self.groq_model
            log.info(f"Generating with Groq ({model})")
            return await self._call_groq(prompt, model)
        else:
            model = model_override or self.ollama_model
            log.info(f"Generating with Ollama ({model})")
            return await self._call_ollama(prompt, model)

    # ══════════════════════════════════════════════════════════════════════════
    # BATCH GENERATION - All slides in ONE call
    # ══════════════════════════════════════════════════════════════════════════
    
    async def generate_all_slides_batch(
        self,
        query: str,
        context_text: str,
        num_slides: int,
        language: str = "English",
        available_images: list = None,
        image_contexts: dict = None,
    ) -> list[dict]:
        """
        Generate ALL slides in a single LLM call.
        Much faster than sequential generation (1 call vs N calls).
        """
        
        # Build image info for prompt
        image_block = ""
        if available_images:
            img_lines = []
            for img_id in available_images[:15]:
                ctx = (image_contexts or {}).get(img_id, "")
                if ctx:
                    img_lines.append(f'  - "{img_id}": {ctx[:80]}...')
                else:
                    img_lines.append(f'  - "{img_id}"')
            image_block = f"""
Available images (use image_id field to assign):
{chr(10).join(img_lines)}
"""
        
        prompt = f"""You are an expert AI teaching assistant creating an educational presentation.

Generate exactly {num_slides} slides about: {query}

Language: {language}

CONTEXT FROM DOCUMENTS:
{context_text}

{image_block}

OUTPUT FORMAT - Return a JSON object with this exact structure:
{{
  "slides": [
    {{
      "slide_type": "intro|definition|concept|example|comparison|summary",
      "title": "Slide Title",
      "bullets": [
        {{"text": "Specific fact or concept with concrete details", "source_id": "Page X"}},
        {{"text": "Second distinct point", "source_id": "Page Y"}},
        {{"text": "Third point with example or application", "source_id": "Page Z"}}
      ],
      "key_message": "One-sentence synthesis of this slide's main takeaway",
      "visual_hint": "flowchart|mindmap|timeline|comparison|process|hierarchy|none",
      "image_id": "IMG_001 or null",
      "speaker_notes": "Brief explanation for presenter"
    }}
  ]
}}

RULES:
1. Each slide MUST have 3-5 bullets with SPECIFIC facts, not vague statements
2. NO repetition between slides - each slide covers different aspects
3. visual_hint: Use variety - don't repeat the same type consecutively
4. image_id: Assign relevant images from the list above, or null if none fit
5. Slide types should follow a logical teaching arc:
   - Start with intro/definition
   - Middle slides: concept, example, comparison
   - End with summary
6. All content must be in {language}

Generate the {num_slides} slides now as valid JSON:"""

        try:
            raw = await self.generate_async(query, [], prompt_override=prompt)
            data = json.loads(raw)
            slides = data.get("slides", [])
            
            if len(slides) < num_slides:
                log.warning(f"Batch generation returned {len(slides)}/{num_slides} slides")
            
            log.info(f"✔ Batch generated {len(slides)} slides in one call")
            return slides
            
        except json.JSONDecodeError as e:
            log.error(f"Batch generation JSON parse failed: {e}")
            return []
        except Exception as e:
            log.error(f"Batch generation failed: {e}")
            return []

    # ══════════════════════════════════════════════════════════════════════════
    # JUDGE (Optional quality check)
    # ══════════════════════════════════════════════════════════════════════════
    
    async def judge_async(self, slide_json: dict) -> dict:
        """Rate slide quality. Returns {"score": 1-10, "feedback": str}."""
        if not self.judge_enabled:
            return {"score": 10, "feedback": "judge disabled"}
        
        judge_prompt = f"""Rate this teaching slide on specificity and accuracy (1-10).
Slide: {json.dumps(slide_json, ensure_ascii=False)}

Return JSON: {{"score": <1-10>, "feedback": "<one sentence>"}}"""
        
        try:
            if self.backend == "groq":
                raw = await self._call_groq(judge_prompt, "llama-3.1-8b-instant")
            else:
                raw = await self._call_ollama(judge_prompt, self.judge_model)
            
            result = json.loads(raw)
            return {"score": int(result.get("score", 7)), "feedback": result.get("feedback", "")}
        except Exception as e:
            log.warning(f"Judge failed: {e}")
            return {"score": 7, "feedback": "judge error"}

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_prompt(self, query: str, context_chunks: list) -> str:
        context_text = "".join(f"- {c.text}\n" for c in context_chunks)
        return f"""You are an expert AI Teaching Assistant.
Synthesize the context into a concise answer using ONLY the provided context.

=== CONTEXT ===
{context_text}
=== END CONTEXT ===

QUESTION: {query}

ANSWER (as JSON):"""

    # Sync fallback for CLI
    def generate(self, query: str, context_chunks: list, prompt_override: str = None, model_override: str = None) -> str:
        import asyncio
        return asyncio.run(self.generate_async(query, context_chunks, prompt_override, model_override))


# ══════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import asyncio
    
    async def test():
        engine = LLMEngine()
        print(f"Backend: {engine.backend}")
        print(f"Groq Key: {'✓' if engine.groq_api_key else '✗'}")
        
        # Test simple generation
        result = await engine.generate_async(
            "What is machine learning?",
            [],
            prompt_override='Return JSON: {"answer": "brief explanation"}'
        )
        print(f"Response: {result[:200]}")
    
    asyncio.run(test())
