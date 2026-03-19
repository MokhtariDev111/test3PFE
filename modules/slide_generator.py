"""
slide_generator.py  —  Step 9: Slide Generator (JSON → Slide Structure)
=======================================================================
Receives: The lesson plan dict from Step 8 (PedagogicalEngine).
Action:   Maps the JSON structure into a clean list of SlideData objects
          with a well-defined structure ready for PowerPoint rendering.
Returns:  list[SlideData]

SlideData is the contract between Step 9 and Step 10 (pptx_exporter.py).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

log = logging.getLogger("slide_generator")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


@dataclass
class SlideData:
    """Single slide ready for PowerPoint export."""
    slide_type:       str          # "title" | "content" | "section"
    title:            str
    bullets:          list[dict] = field(default_factory=list) # [{text: str, source_id: str}]
    speaker_notes:    str = ""
    image_id:         str = None
    quality_score:    int = 10
    quality_feedback: str = ""


def build_slides(lesson: dict) -> list[SlideData]:
    """
    Converts a lesson plan dict (from PedagogicalEngine) into
    a list of SlideData objects.

    1. A title slide is always generated first.
    2. Each "slide" in the lesson plan becomes a content slide.
    """
    topic = lesson.get("topic", "Lesson")
    raw_slides = lesson.get("slides", [])

    if not raw_slides:
        log.warning("No slides found in lesson plan. Returning empty list.")
        return []

    output: list[SlideData] = []

    # 1. Title slide
    output.append(SlideData(
        slide_type="title",
        title=topic,
        bullets=[],
        speaker_notes=f"Introduction to: {topic}"
    ))

    # 2. Content slides
    for raw in raw_slides:
        raw_bullets = raw.get("bullets", [])
        if not isinstance(raw_bullets, list):
            raw_bullets = [raw_bullets] if raw_bullets else []
            
        output.append(SlideData(
            slide_type="content",
            title=raw.get("title", ""),
            bullets=raw_bullets,
            speaker_notes=raw.get("speaker_notes", ""),
            image_id=raw.get("image_id"),
            quality_score=raw.get("quality_score", 0),
            quality_feedback=raw.get("quality_feedback", "")
        ))

    log.info(f"Generated {len(output)} slide(s) (1 title + {len(output)-1} content).")
    return output


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from modules.retrieval import Retriever
    from modules.pedagogical_engine import PedagogicalEngine

    query = "Explain the Transformer attention mechanism"
    print(f"\n[Step 6] Retrieving context...")
    retriever = Retriever()
    chunks = retriever.search(query, top_k=4)

    print("\n[Step 8] Generating lesson plan...")
    engine = PedagogicalEngine()
    lesson = engine.generate_lesson(query, chunks)

    print("\n[Step 9] Building slide structure...")
    slides = build_slides(lesson)

    print(f"\n── {len(slides)} Slide(s) Generated ───────────────────────")
    for i, s in enumerate(slides, 1):
        print(f"\n  [{s.slide_type.upper()}] Slide {i}: {s.title} (Score: {s.quality_score}/10)")
        for b in s.bullets:
            txt = b.get("text", "") if isinstance(b, dict) else str(b)
            src = b.get("source_id", "") if isinstance(b, dict) else ""
            print(f"    • {txt} [{src}]")
        if s.speaker_notes:
            print(f"  Notes: {s.speaker_notes[:80]}...")
        if s.quality_feedback:
            print(f"  Feedback: {s.quality_feedback}")
    print("────────────────────────────────────────────────────────")
