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
    "topic": "string — short filename-friendly topic (English)",
    "slides": [
        {
            "title": "string — clear, engaging slide title",

            "bullets": [
                {
                    "text": "string — concise but informative key point",
                    "source_id": "string — EXACT source ID or 'General Knowledge'"
                }
            ],

            "key_message": "string — the ONE main takeaway of the slide",

            "visual_hint": "string — what visual should be shown (diagram, flowchart, timeline, comparison, etc.)",
            
            "image_id": "string or null — the EXACT source ID of an available PDF image to feature, otherwise null",

            "speaker_notes": "string — 3-5 sentence deep explanation for the teacher, using analogies",

            "quality_score": 0,
            "quality_feedback": "string — short justification (clarity, depth, accuracy)"
        }
    ]
}

JSON_INSTRUCTION = (
    "You are an expert AI Teaching Assistant, instructional designer, and presentation expert.\n"
    "Your goal is to create a HIGH-QUALITY, ENGAGING, and PEDAGOGICALLY EFFECTIVE lesson.\n\n"

    "Output ONLY valid JSON matching this exact schema (no markdown, no prose):\n"
    + json.dumps(LESSON_SCHEMA, indent=2)
    + "\n\n"

    "STRICT RULES:\n"

    # STRUCTURE
    "- Generate exactly NUM_SLIDES slides.\n"
    "- Follow a LOGICAL TEACHING FLOW:\n"
    "  1. Introduction (what & why it matters)\n"
    "  2. Core concepts\n"
    "  3. Examples / applications\n"
    "  4. Key insights / summary\n"

    # LANGUAGE
    "- LANGUAGE: Write ALL slide content in SLIDE_LANGUAGE.\n"

    # CONTENT QUALITY & ENGAGEMENT
    "- Each slide must contain 3 to 5 HIGH-VALUE bullet points.\n"
    "- Act like a world-class TED speaker and instructional designer.\n"
    "- Use brilliant analogies, storytelling, and real-world examples to explain complex topics.\n"
    "- Each bullet must be clear, specific, and impactful.\n"

    # CITATIONS
    "- For EVERY bullet, you MUST include a correct 'source_id'.\n"
    "- Use EXACT source IDs from context or 'General Knowledge'.\n"

    # IMAGES & VISUALS
    "- You have access to raw images from the PDFs. They will be listed as 'AVAILABLE IMAGES'.\n"
    "- If you see a highly relevant image ID, set 'image_id' to that EXACT string to feature it on the slide.\n"
    "- If no image fits, set 'image_id' to null.\n"
    "- Also add a 'visual_hint' describing a diagram concept (even if you feature an image).\n"

    # TEACHING DEPTH
    "- Speaker notes must be 3–5 sentences deep, highly engaging, and use vivid analogies.\n"

    # SELF-EVALUATION
    "- Assign a 'quality_score' (1–10) based on clarity, depth, and usefulness.\n"
    "- Provide a 1-sentence 'quality_feedback'.\n"
    "- A score below 7 is NOT acceptable.\n"

    # TOPIC
    "- The 'topic' must be short (max 4–5 words), in English, filename-friendly.\n"

    # FINAL CONSTRAINT
    "- DO NOT output anything outside the JSON.\n"
)


class PedagogicalEngine:
    def __init__(self):
        self.llm = LLMEngine()

    def _build_pedagogical_prompt(self, query: str, context_chunks: list[TextChunk],
                                   num_slides: int = 3, language: str = "English",
                                   available_images: list[str] = None) -> str:
        """Builds a structured prompt requesting JSON lesson output."""
        context_text = "\n".join(f"[ID: {chunk.source} (page {chunk.page})] - {chunk.text}" for chunk in context_chunks)
        lang_label = "French" if language.lower() in ("fr", "french", "français") else "English"
        instruction = (
            JSON_INSTRUCTION
            .replace("NUM_SLIDES", str(num_slides))
            .replace("SLIDE_LANGUAGE", lang_label)
        )
        
        images_text = "AVAILABLE IMAGES:\n" + ("\n".join(f"- {img}" for img in available_images) if available_images else "None")

        return (
            f"{instruction}\n\n"
            "=== CONTEXT FROM DOCUMENTS ===\n"
            f"{context_text}\n"
            "=== END CONTEXT ===\n\n"
            f"{images_text}\n\n"
            f"USER TOPIC / QUESTION: {query}\n\n"
            "JSON LESSON PLAN:"
        )

    def _extract_json(self, raw: str) -> dict:
        """Tries to extract a valid JSON object from the model output."""
        # Strip any markdown fences if the model added them anyway
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # Some local LLMs hallucinate Python syntax instead of strict JSON
        raw = raw.replace('"image_id": None', '"image_id": null')
        raw = raw.replace(': None', ': null')
        raw = raw.replace(': False', ': false')
        raw = raw.replace(': True', ': true')

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
            log.warning(f"Could not parse JSON from LLM output: {e}\nRaw output snippet: {raw[:300]}")
            return {"topic": "Unknown", "slides": [], "_raw": raw}

    def generate_lesson(self, query: str, context_chunks: list[TextChunk],
                        num_slides: int = 3, language: str = "English",
                        available_images: list[str] = None) -> dict:
        """Main entry point for Step 8: produces a structured JSON lesson."""
        if not context_chunks:
            return {"topic": query, "slides": [], "_error": "No context retrieved."}

        prompt = self._build_pedagogical_prompt(query, context_chunks,
                                                num_slides=num_slides, language=language,
                                                available_images=available_images)
        log.info(f"Requesting lesson plan for: '{query}' ({num_slides} slides, lang={language})...")

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
        print(f"\n  Slide {i}: {slide.get('title', '')} [Score: {slide.get('quality_score', '?')}/10]")
        for b in slide.get("bullets", []):
            if isinstance(b, dict):
                print(f"    • {b.get('text', '')} (Source: {b.get('source_id', '')})")
            else:
                print(f"    • {b}")
        print(f"  Notes: {slide.get('speaker_notes', '')}")
    print("─────────────────────────────────────────────")
