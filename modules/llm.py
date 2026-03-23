"""
llm.py — with CUDA OOM retry guard
====================================
If Ollama returns a CUDA error, automatically retries with
halved context window (num_ctx) after a short pause.
"""

import json
import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from modules.text_processing import TextChunk
except ImportError:
    pass

from modules.config_loader import CONFIG

log = logging.getLogger("llm")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%H:%M:%S")


class LLMEngine:
    def __init__(self):
        self.api_url = CONFIG["llm"]["api_url"]
        self.model   = CONFIG["llm"]["model"]
        self.temp    = CONFIG["llm"]["temperature"]
        # 2048 fits 6GB VRAM: ~925 input tokens + 1100 output tokens = 2025 total
        self.ctx_len = min(CONFIG["llm"]["context_length"], 3072)

    def _build_prompt(self, query: str, context_chunks: list) -> str:
        context_text = "".join(f"- {chunk.text}\n" for chunk in context_chunks)
        return (
            "You are an expert AI Teaching Assistant.\n"
            "Synthesize the context into a concise answer. "
            "Answer using ONLY the provided context.\n\n"
            "=== CONTEXT ===\n"
            f"{context_text}\n"
            "=== END CONTEXT ===\n\n"
            f"QUESTION: {query}\n\nANSWER:\n"
        )

    async def generate_async(self, query: str, context_chunks: list,
                             prompt_override: str = None) -> str:
        import httpx
        import asyncio

        if not context_chunks and not prompt_override:
            return "I don't have enough context to answer that question."

        prompt = prompt_override or self._build_prompt(query, context_chunks)
        log.info(f"Sending prompt to Ollama ({self.model}), ctx={self.ctx_len}...")

        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {
                "temperature": self.temp,
                "num_ctx":     self.ctx_len,
                "num_predict": -1,    # -1 = unlimited, let num_ctx be the only ceiling
            }
        }

        # Retry up to 2 times: on CUDA OOM halve the context and retry
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=600.0) as client:
                    response = await client.post(self.api_url, json=payload)
                    response.raise_for_status()
                    return response.json().get("response", "").strip()

            except httpx.ConnectError as e:
                msg = (f"Cannot reach Ollama at {self.api_url}. "
                       f"Is it running? Details: {e}")
                log.error(msg)
                raise RuntimeError(msg) from e

            except httpx.HTTPStatusError as e:
                err_body = e.response.text

                # CUDA OOM → halve context and retry once
                if "CUDA error" in err_body and attempt == 0:
                    reduced = max(512, payload["options"]["num_ctx"] // 2)
                    log.warning(
                        f"CUDA OOM detected — retrying with "
                        f"num_ctx={reduced} (was {payload['options']['num_ctx']})"
                    )
                    payload["options"]["num_ctx"] = reduced
                    await asyncio.sleep(3)  # give GPU time to free memory
                    continue

                msg = (f"Ollama returned HTTP "
                       f"{e.response.status_code}: {err_body}")
                log.error(msg)
                raise RuntimeError(msg) from e

            except Exception as e:
                log.error(f"Unexpected error calling Ollama: {e}")
                raise RuntimeError(f"Unexpected error: {e}") from e

    def generate(self, query: str, context_chunks: list,
                 prompt_override: str = None) -> str:
        """Sync version for CLI/standalone tests only."""
        import urllib.request
        from urllib.error import URLError

        if not context_chunks and not prompt_override:
            return "I don't have enough context to answer that question."

        prompt  = prompt_override or self._build_prompt(query, context_chunks)
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": self.temp, "num_ctx": self.ctx_len}
        }

        json_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url, data=json_data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except URLError as e:
            msg = f"Failed to connect to Ollama at {self.api_url}. Error: {e.reason}"
            log.error(msg)
            raise RuntimeError(msg) from e
