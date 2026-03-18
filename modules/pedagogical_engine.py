"""
pedagogical_engine.py  —  Step 8: Pedagogical Engine
====================================================
Receives: A user query and retrieved TextChunks (from Step 6).
Action:   Uses the LLM to produce a STRUCTURED JSON lesson plan
          instead of a raw text answer. The schema is:
          {
            "topic": "...",
            "slides": [
              {
                "title": "...",
                "bullets": ["...", "..."],
                "speaker_notes": "..."
              }
            ]
          }
Returns:  A validated Python dict matching the JSON schema above.
"""

import json
import logging
import re
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
from modules.llm import LLMEngine

log = logging.getLogger("pedagogical_engine")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


LESSON_SCHEMA = {
    "topic": "string — main concept being taught",
    "slides": [
        {
            "title": "string — slide title",
            "bullets": ["string — key point 1", "string — key point 2"],
            "speaker_notes": "string — full explanation for the teacher"
        }
    ]
}

JSON_INSTRUCTION = (
    "You are an expert AI Teaching Assistant and an instructional designer.\n"
    "Your task is to create a detailed, highly informative, and engaging lesson plan from the context provided.\n\n"
    "Output ONLY valid JSON matching this exact schema (no markdown, no prose, no ```json):\n"
    + json.dumps(LESSON_SCHEMA, indent=2)
    + "\n\nRULES:\n"
    "- Generate exactly NUM_SLIDES content slides (plus a title slide will be added automatically).\n"
    "- Make the slide content RICH and DETAILED. Do not just use a few words.\n"
    "- Each slide must have 3 to 5 detailed bullet points. Include key facts, examples, and deep context from the source material.\n"
    "- Speaker notes should be comprehensive (3-4 sentences) explaining the concepts deeply to the teacher.\n"
    "- The 'topic' field MUST be a short, descriptive phrase (max 4-5 words) summarizing the presentation (e.g., 'Transformer-Attention-Mechanism' or 'Intro-to-Photosynthesis'). We use this as the filename.\n"
    "- Do NOT include any text outside the JSON object.\n"
)


class PedagogicalEngine:
    def __init__(self):
        self.llm = LLMEngine()

    def _build_pedagogical_prompt(self, query: str, context_chunks: list[TextChunk], num_slides: int = 3) -> str:
        """Builds a structured prompt requesting JSON lesson output."""
        context_text = "\n".join(f"- {chunk.text}" for chunk in context_chunks)
        instruction = JSON_INSTRUCTION.replace("NUM_SLIDES", str(num_slides))

        return (
            f"{instruction}\n\n"
            "=== CONTEXT FROM DOCUMENTS ===\n"
            f"{context_text}\n"
            "=== END CONTEXT ===\n\n"
            f"USER TOPIC / QUESTION: {query}\n\n"
            "JSON LESSON PLAN:"
        )

    def _extract_json(self, raw: str) -> dict:
        """Tries to extract a valid JSON object from the model output."""
        # Strip any markdown fences if the model added them anyway
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # Try to find the JSON object using braces
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fall back: try parsing the whole string
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            log.warning(f"Could not parse JSON from LLM output: {e}")
            return {"topic": "Unknown", "slides": [], "_raw": raw}

    def generate_lesson(self, query: str, context_chunks: list[TextChunk], num_slides: int = 3) -> dict:
        """Main entry point for Step 8: produces a structured JSON lesson."""
        if not context_chunks:
            return {"topic": query, "slides": [], "_error": "No context retrieved."}

        prompt = self._build_pedagogical_prompt(query, context_chunks, num_slides=num_slides)
        log.info(f"Requesting lesson plan for: '{query}' ({num_slides} slides)...")

        raw_output = self.llm.generate(query, context_chunks, prompt_override=prompt)

        lesson = self._extract_json(raw_output)
        log.info(f"Lesson plan created: {len(lesson.get('slides', []))} slide(s).")
        return lesson


# ── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.retrieval import Retriever

    query = "Explain the Transformer attention mechanism"
    print(f"\n[Step 6] Retrieving context for: '{query}'...")
    retriever = Retriever()
    chunks = retriever.search(query, top_k=4)

    print("\n[Step 8] Generating pedagogical lesson plan...")
    engine = PedagogicalEngine()
    lesson = engine.generate_lesson(query, chunks)

    print("\n── Lesson Plan ──────────────────────────────")
    print(f"Topic: {lesson.get('topic', 'N/A')}")
    for i, slide in enumerate(lesson.get("slides", []), 1):
        print(f"\n  Slide {i}: {slide.get('title', '')}")
        for b in slide.get("bullets", []):
            print(f"    • {b}")
        print(f"  Notes: {slide.get('speaker_notes', '')}")
    print("─────────────────────────────────────────────")
