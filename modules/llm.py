"""
llm.py  —  Step 7: LLM Integration (Ollama)
===========================================
Receives: A user query and a list of retrieved TextChunks (from Step 6).
Action:   Builds a prompt containing the context chunks and sends it
          to the local Ollama API for generation.
Returns:  The generated string response from the LLM.
"""

import json
import logging
from pathlib import Path
import sys

# Setup imports so it can run standalone
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import urllib.request
    from urllib.error import URLError
    from modules.text_processing import TextChunk
except ImportError:
    pass

from modules.config_loader import CONFIG

log = logging.getLogger("llm")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


class LLMEngine:
    def __init__(self):
        self.api_url = CONFIG["llm"]["api_url"]
        self.model   = CONFIG["llm"]["model"]
        self.temp    = CONFIG["llm"]["temperature"]
        self.ctx_len = CONFIG["llm"]["context_length"]

    def _build_prompt(self, query: str, context_chunks: list[TextChunk]) -> str:
        """Formats the retrieved chunks and the user query into a single string."""
        
        context_text = ""
        for chunk in context_chunks:
            context_text += f"- {chunk.text}\n"
            
        prompt = (
            "You are an expert AI Teaching Assistant.\n"
            "Below are excerpts from various educational documents.\n"
            "Task: Synthesize the context into a single, cohesive, and concise answer to the user's question.\n"
            "CRITICAL INSTRUCTIONS:\n"
            " - Do NOT repeat the same information multiple times.\n"
            " - Do NOT answer document-by-document or chunk-by-chunk.\n"
            " - Answer using ONLY the provided context.\n"
            " - If the answer is not contained in the context, clearly state that you don't have enough information.\n\n"
            "=== CONTEXT ===\n"
            f"{context_text}\n"
            "=== END CONTEXT ===\n\n"
            f"USER QUESTION: {query}\n\n"
            "FINAL SYNTHESIZED ANSWER:\n"
        )
        return prompt

    def generate(self, query: str, context_chunks: list[TextChunk], prompt_override: str = None) -> str:
        """Sends the prompt to Ollama and returns the generated text."""
        
        if not context_chunks and not prompt_override:
            return "I don't have enough context to answer that question."

        prompt = prompt_override if prompt_override else self._build_prompt(query, context_chunks)
        log.info(f"Sending prompt to Ollama ({self.model})...")
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temp,
                "num_ctx": self.ctx_len
            }
        }
        
        # We use Python's built-in urllib to avoid requiring the 'requests' library
        json_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.api_url, 
            data=json_data, 
            headers={"Content-Type": "application/json"}
        )
        
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("response", "").strip()
                
        except URLError as e:
            err_msg = f"Failed to connect to Ollama at {self.api_url}. Error: {e.reason}"
            log.error(err_msg)
            raise RuntimeError(err_msg) from e
        except Exception as e:
            log.error(f"Unexpected error calling Ollama: {e}")
            raise RuntimeError(f"Unexpected error: {e}") from e


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.retrieval import Retriever
    
    query = "What is the Transformer architecture and what does it rely on?"
    print(f"\n[Step 6] Retrieving context for: '{query}'...")
    
    retriever = Retriever()
    chunks = retriever.search(query, top_k=3)
    
    print(f"\n[Step 7] Generating answer with Ollama ({CONFIG['llm']['model']})...")
    llm = LLMEngine()
    answer = llm.generate(query, chunks)
    
    print("\n── Final AI Answer ─────────────────────────")
    import textwrap
    print(textwrap.fill(answer, width=80))
    print("────────────────────────────────────────────")
