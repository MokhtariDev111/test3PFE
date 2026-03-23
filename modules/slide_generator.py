"""
slide_generator.py  — v2: Supports new slide_type variety
==========================================================
Maps the enriched lesson plan (definition, concept, example,
comparison, summary) into SlideData objects.
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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%H:%M:%S")

VALID_SLIDE_TYPES = {"title", "definition", "concept", "example",
                     "comparison", "summary", "content", "section"}


@dataclass
class SlideData:
    slide_type:       str
    title:            str
    bullets:          list = field(default_factory=list)
    speaker_notes:    str  = ""
    key_message:      str  = ""
    visual_hint:      str  = "none"
    image_id:         str  = None
    quality_score:    int  = 8
    quality_feedback: str  = ""


def build_slides(lesson: dict) -> list[SlideData]:
    topic      = lesson.get("topic", "Lesson")
    raw_slides = lesson.get("slides", [])

    if not raw_slides:
        log.warning("No slides in lesson plan.")
        return []

    output: list[SlideData] = []

    # Title slide
    output.append(SlideData(
        slide_type    = "title",
        title         = topic,
        speaker_notes = f"Introduction to: {topic}",
        key_message   = "",
    ))

    for raw in raw_slides:
        raw_bullets = raw.get("bullets", [])
        if not isinstance(raw_bullets, list):
            raw_bullets = [raw_bullets] if raw_bullets else []

        # Normalize slide_type
        st = raw.get("slide_type", "concept").lower().strip()
        if st not in VALID_SLIDE_TYPES:
            st = "concept"

        output.append(SlideData(
            slide_type    = st,
            title         = raw.get("title", ""),
            bullets       = raw_bullets,
            speaker_notes = raw.get("speaker_notes", ""),
            key_message   = raw.get("key_message", ""),
            visual_hint   = raw.get("visual_hint", "none"),
            image_id      = raw.get("image_id"),
            quality_score = raw.get("quality_score", 0),
            quality_feedback = raw.get("quality_feedback", ""),
        ))

    log.info(f"Built {len(output)} slides (1 title + {len(output)-1} content).")
    return output
