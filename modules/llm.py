"""
llm.py — v2: JSON mode + model_override + judge support
=========================================================
Changes from v1:
  1. Adds `format: "json"` to every Ollama payload — forces valid JSON output,
     eliminates most cases where _repair_json was needed downstream.
  2. Adds `model_override` param — lets pedagogical_engine route different tasks
     (generation vs judging) to different models without instantiating new engines.
  3. Adds `judge_async()` convenience method — calls the judge_model from config
     with a simple scoring prompt, returns a dict {score, feedback}.
  4. CUDA OOM retry and sync fallback kept unchanged.
"""

import json
import logging
from pathlib import Path
import sys

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
        self.api_url    = CONFIG["llm"]["api_url"]
        self.model      = CONFIG["llm"]["model"]
        self.temp       = CONFIG["llm"]["temperature"]
        self.ctx_len    = min(CONFIG["llm"]["context_length"], 2048)  # Cap at 2048 for 6GB VRAM
        self.judge_model   = CONFIG["llm"].get("judge_model", self.model)
        self.judge_enabled = CONFIG["llm"].get("judge_enabled", False)

    # ── Internal HTTP call ────────────────────────────────────────────────────

    async def _call_async(
        self,
        prompt: str,
        model: str,
        use_json_mode: bool = True,
    ) -> str:
        """
        Core async HTTP call to Ollama.
        use_json_mode=True adds `format: "json"` which forces the model to
        produce syntactically valid JSON. Disable for free-text responses.
        """
        import httpx
        import asyncio

        payload: dict = {
            "model":  model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temp,
                "num_ctx":     self.ctx_len,
                "num_predict": -1,
            },
        }
        if use_json_mode:
            payload["format"] = "json"

        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(self.api_url, json=payload)
                    response.raise_for_status()
                    return response.json().get("response", "").strip()

            except httpx.ConnectError as e:
                msg = f"Cannot reach Ollama at {self.api_url}. Is it running? {e}"
                log.error(msg)
                raise RuntimeError(msg) from e

            except httpx.HTTPStatusError as e:
                err_body = e.response.text
                if "CUDA error" in err_body and attempt == 0:
                    reduced = max(512, payload["options"]["num_ctx"] // 2)
                    log.warning(
                        f"CUDA OOM — retrying with num_ctx={reduced} "
                        f"(was {payload['options']['num_ctx']})"
                    )
                    payload["options"]["num_ctx"] = reduced
                    await asyncio.sleep(3)
                    continue
                msg = f"Ollama HTTP {e.response.status_code}: {err_body}"
                log.error(msg)
                raise RuntimeError(msg) from e

            except Exception as e:
                log.error(f"Unexpected Ollama error: {e}")
                raise RuntimeError(f"Unexpected error: {e}") from e

        return ""

    # ── Public generation method ──────────────────────────────────────────────

    async def generate_async(
        self,
        query: str,
        context_chunks: list,
        prompt_override: str = None,
        model_override: str = None,
    ) -> str:
        """
        Main generation call.
        - prompt_override: full custom prompt (used by pedagogical_engine)
        - model_override: use a different Ollama model for this call only
        """
        if not context_chunks and not prompt_override:
            return "{}"

        prompt = prompt_override or self._build_prompt(query, context_chunks)
        model  = model_override or self.model

        log.info(f"Generating with model={model}, ctx={self.ctx_len}")
        return await self._call_async(prompt, model, use_json_mode=True)

    # ── Judge method ──────────────────────────────────────────────────────────

    async def judge_async(self, slide_json: dict) -> dict:
        """
        Asks the judge_model to rate a slide on specificity and accuracy.
        Returns: {"score": int 1-10, "feedback": str}

        If judge is disabled in config or the call fails, returns a
        passing score so generation is not blocked.
        """
        if not self.judge_enabled:
            return {"score": 10, "feedback": "judge disabled"}

        judge_prompt = f"""You are an expert educational content evaluator.
Rate the following teaching slide strictly on two criteria:
1. Specificity — does each bullet contain a concrete fact, number, or mechanism? (not vague)
2. Accuracy — does the content match the context and avoid hallucination?

Slide JSON:
{json.dumps(slide_json, ensure_ascii=False)}

Return ONLY a JSON object with this exact structure:
{{"score": <integer 1-10>, "feedback": "<one sentence explaining the main issue if score < 7, or 'Good' if score >= 7>"}}

Score 1-4: vague, repetitive, or incorrect content.
Score 5-6: acceptable but improvable.
Score 7-10: specific, accurate, and educational.
JSON:"""

        try:
            raw = await self._call_async(judge_prompt, self.judge_model, use_json_mode=True)
            result = json.loads(raw)
            score    = int(result.get("score", 7))
            feedback = str(result.get("feedback", ""))
            log.info(f"  Judge score: {score}/10 — {feedback[:60]}")
            return {"score": score, "feedback": feedback}
        except Exception as e:
            log.warning(f"  Judge call failed (non-fatal): {e} — passing slide through")
            return {"score": 7, "feedback": "judge error — passed by default"}

    # ── Default prompt builder (used when no prompt_override) ─────────────────

    def _build_prompt(self, query: str, context_chunks: list) -> str:
        context_text = "".join(f"- {c.text}\n" for c in context_chunks)
        return (
            "You are an expert AI Teaching Assistant.\n"
            "Synthesize the context into a concise answer. "
            "Answer using ONLY the provided context.\n\n"
            "=== CONTEXT ===\n"
            f"{context_text}\n"
            "=== END CONTEXT ===\n\n"
            f"QUESTION: {query}\n\nANSWER:\n"
        )

    # ── Sync fallback (CLI/tests only) ────────────────────────────────────────

    def generate(
        self,
        query: str,
        context_chunks: list,
        prompt_override: str = None,
        model_override: str = None,
    ) -> str:
        """Synchronous version for CLI tests only. Not used in the web server."""
        import urllib.request
        from urllib.error import URLError

        if not context_chunks and not prompt_override:
            return "{}"

        prompt = prompt_override or self._build_prompt(query, context_chunks)
        model  = model_override or self.model

        payload = {
            "model":   model,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",
            "options": {"temperature": self.temp, "num_ctx": self.ctx_len},
        }

        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            self.api_url, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read().decode()).get("response", "").strip()
        except URLError as e:
            msg = f"Failed to connect to Ollama at {self.api_url}: {e.reason}"
            log.error(msg)
            raise RuntimeError(msg) from e
