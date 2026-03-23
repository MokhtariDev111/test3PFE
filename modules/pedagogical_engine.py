"""
pedagogical_engine.py — v4: Batched generation
================================================
Instead of asking Mistral to generate all N slides at once
(which causes truncation), generates in batches of 2 slides
and merges the results. Each batch call is small enough to
complete fully within the context window.
"""

import json
import logging
import re
from pathlib import Path
import sys
import hashlib

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
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%H:%M:%S")

# Slide types to use in order for a good pedagogical arc
SLIDE_ARC = ["definition", "concept", "concept", "example", "comparison",
             "example", "summary", "concept", "comparison", "definition",
             "example", "summary", "concept", "comparison", "summary"]

# Minimal schema for ONE slide
_ONE_SLIDE_SCHEMA = '''{
  "slide_type": "concept",
  "title": "Slide Title Here",
  "bullets": [
    {"text": "Specific fact or mechanism with a concrete detail", "source_id": "source or General Knowledge"},
    {"text": "Second distinct point — no overlap with bullet 1", "source_id": "source or General Knowledge"},
    {"text": "Third point — real example or implication", "source_id": "source or General Knowledge"}
  ],
  "key_message": "One-sentence synthesis NOT present in the bullets",
  "visual_hint": "flowchart",
  "image_id": null,
  "speaker_notes": "One or two short sentences.",
  "quality_score": 8,
  "quality_feedback": "Brief reason."
}'''


def _build_single_slide_prompt(
    query: str,
    context_text: str,
    slide_type: str,
    slide_number: int,
    total: int,
    language: str,
    prior_titles: list[str],
    prior_hints: list[str] = None,
) -> str:
    prior = "\n".join(f"- {t}" for t in prior_titles[-6:]) if prior_titles else "(none)"
    used_hints = ", ".join(prior_hints[-4:]) if prior_hints else "(none)"
    # Suggest a hint that hasn't been used recently
    all_hints = ["flowchart", "mindmap", "timeline", "comparison", "process", "hierarchy"]
    suggested_hint = next((h for h in all_hints if not prior_hints or h not in prior_hints[-3:]), all_hints[slide_number % len(all_hints)])
    return f"""You are an expert AI teaching assistant. Output ONLY a valid JSON object for ONE slide. No text before or after the JSON.

Generate slide {slide_number} of {total} about: {query}
Slide type: {slide_type}
Language: {language}

JSON schema:
{_ONE_SLIDE_SCHEMA}

Rules:
1. slide_type must be: {slide_type}
2. bullets: exactly 3. Each must contain a specific fact, number, or mechanism. Never vague.
3. key_message: one sentence synthesis — must NOT repeat bullet text.
4. visual_hint: MUST be different from recently used hints ({used_hints}). Suggested: "{suggested_hint}". Choose from: flowchart, mindmap, timeline, comparison, process, hierarchy, none
5. speaker_notes: 1-2 short sentences only.
6. All content in {language}.
7. source_id: exact document source or "General Knowledge".
8. Start with {{ and end with }}. No markdown.
9. Do NOT repeat previous slide titles or bullets. Each slide must introduce new information.

Previous slide titles (avoid repeating):
{prior}

Context:
{context_text}

Topic: {query}

JSON:"""


def _repair_json(raw: str) -> str:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    raw = raw.replace(": None", ": null").replace(":None", ": null")
    raw = raw.replace(": True", ": true").replace(": False", ": false")
    # Remove trailing commas before } or ]
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    # Strip prose before {
    brace_start = raw.find("{")
    if brace_start > 0:
        raw = raw[brace_start:]
    # Strip prose after }
    brace_end = raw.rfind("}")
    if brace_end != -1 and brace_end < len(raw) - 1:
        raw = raw[:brace_end + 1]
    return raw


def _extract_slide_json(raw: str) -> dict | None:
    log.warning(f"Raw slide output ({len(raw)} chars): {raw[:400]!r}")
    if not raw or not raw.strip():
        return None
    repaired = _repair_json(raw)
    try:
        obj = json.loads(repaired)
        if "title" in obj and "bullets" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", repaired, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if "title" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    log.error(f"Could not extract slide JSON from: {raw[:200]!r}")
    return None


def _extract_topic(raw: str, fallback: str) -> str:
    """Try to extract topic from first slide output."""
    match = re.search(r'"topic"\s*:\s*"([^"]+)"', raw)
    return match.group(1) if match else fallback


def _norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _slide_fingerprint(slide: dict) -> str:
    """Stable-ish fingerprint to catch near-identical repeats."""
    title = _norm_title(slide.get("title", ""))
    bullets = slide.get("bullets", [])
    if not isinstance(bullets, list):
        bullets = [bullets] if bullets else []
    btxt = []
    for b in bullets:
        if isinstance(b, str):
            btxt.append(b.strip().lower())
        elif isinstance(b, dict):
            btxt.append(str(b.get("text", "")).strip().lower())
        else:
            btxt.append(str(b).strip().lower())
    key = title + "|" + "|".join(btxt[:3])
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()



# Sub-query templates per slide type to drive per-slide retrieval
_SLIDE_TYPE_SUBQUERY = {
    "definition":  "definition and meaning of {query}",
    "concept":     "key concepts and principles of {query}",
    "example":     "real-world examples and applications of {query}",
    "comparison":  "comparison and differences related to {query}",
    "summary":     "summary and main takeaways of {query}",
}


def _subquery_for_slide(query: str, slide_type: str, slide_number: int, prior_titles: list[str]) -> str:
    """Build a focused retrieval sub-query for a specific slide."""
    template = _SLIDE_TYPE_SUBQUERY.get(slide_type, "{query}")
    base = template.format(query=query)
    # After a few slides, steer away from already-covered titles
    if prior_titles:
        covered = ", ".join(prior_titles[-3:])
        base += f" (excluding: {covered})"
    return base


class PedagogicalEngine:
    def __init__(self, retriever=None):
        self.llm = LLMEngine()
        self.retriever = retriever  # optional: injected from api.py for per-slide retrieval

    def _prepare_context(self, context_chunks: list, max_chars: int = 2400) -> str:
        # Deduplicate chunks by text to avoid feeding the same passage twice
        seen = set()
        unique_chunks = []
        for c in context_chunks:
            key = c.text.strip()[:120]
            if key not in seen:
                seen.add(key)
                unique_chunks.append(c)

        context_text = "\n".join(
            f"[Source: {c.source} p{c.page}] {c.text}"
            for c in unique_chunks
        )
        if len(context_text) > max_chars:
            context_text = context_text[:max_chars] + "\n[truncated]"
        return context_text

    async def generate_lesson_async(self, query: str, context_chunks: list,
                                    num_slides: int = 5, language: str = "English",
                                    available_images: list = None) -> dict:
        if not context_chunks:
            return {"topic": query, "slides": [], "_error": "No context retrieved."}

        lang_label   = "French" if language.lower() in ("fr", "french", "français") else "English"
        slides       = []
        topic        = query[:40]
        prior_titles: list[str] = []
        prior_fps: set[str] = set()
        prior_hints: list[str] = []

        log.info(f"Generating {num_slides} slides one-by-one for: '{query}'")

        for i in range(num_slides):
            slide_type = SLIDE_ARC[i % len(SLIDE_ARC)]

            # Per-slide retrieval: fetch chunks relevant to this specific slide type
            if self.retriever is not None:
                sub_q = _subquery_for_slide(query, slide_type, i + 1, prior_titles)
                slide_chunks = self.retriever.search(sub_q, top_k=5)
                # Fall back to original chunks if retrieval returns nothing
                if not slide_chunks:
                    slide_chunks = context_chunks
                log.info(f"  Slide {i+1} sub-query: '{sub_q}' → {len(slide_chunks)} chunks")
            else:
                slide_chunks = context_chunks

            context_text = self._prepare_context(slide_chunks)

            last_err = None
            for attempt in range(3):
                prompt = _build_single_slide_prompt(
                    query, context_text, slide_type,
                    i + 1, num_slides, lang_label, prior_titles, prior_hints
                )
                if attempt > 0:
                    prompt += (
                        "\n\nCRITICAL: Your previous attempt was too similar to earlier slides. "
                        "You MUST change the title and ALL 3 bullets to be substantially different. "
                        "Focus on a different aspect or sub-topic that has NOT been covered yet.\n"
                    )

                try:
                    raw = await self.llm.generate_async(query, slide_chunks, prompt_override=prompt)
                    slide = _extract_slide_json(raw)
                    if not slide:
                        last_err = "extraction failed"
                        continue

                    fp = _slide_fingerprint(slide)
                    title_norm = _norm_title(slide.get("title", ""))
                    if fp in prior_fps or (title_norm and title_norm in {_norm_title(t) for t in prior_titles}):
                        last_err = "duplicate slide"
                        continue

                    # Extract topic from first slide if possible
                    if i == 0:
                        t = _extract_topic(raw, query[:40])
                        if t:
                            topic = t

                    slides.append(slide)
                    prior_fps.add(fp)
                    t = (slide.get("title") or "").strip()
                    if t:
                        prior_titles.append(t)
                    h = (slide.get("visual_hint") or "none").strip().lower()
                    prior_hints.append(h)
                    log.info(f"  ✔ Slide {i+1}/{num_slides}: '{slide.get('title', '?')}' [hint={h}]")
                    break
                except Exception as e:
                    last_err = str(e)
                    continue
            else:
                log.warning(f"  ✗ Slide {i+1}/{num_slides}: could not generate unique slide ({last_err}) — skipping")

        log.info(f"Generated {len(slides)}/{num_slides} slides successfully.")
        return {"topic": topic, "slides": slides}

    def generate_lesson(self, query: str, context_chunks: list,
                        num_slides: int = 5, language: str = "English",
                        available_images: list = None) -> dict:
        """Sync version for CLI tests — generates slides one by one."""
        if not context_chunks:
            return {"topic": query, "slides": [], "_error": "No context retrieved."}

        lang_label   = "French" if language.lower() in ("fr", "french", "français") else "English"
        slides       = []
        topic        = query[:40]
        prior_titles: list[str] = []
        prior_fps: set[str] = set()
        prior_hints: list[str] = []

        for i in range(num_slides):
            slide_type = SLIDE_ARC[i % len(SLIDE_ARC)]

            # Per-slide retrieval
            if self.retriever is not None:
                sub_q = _subquery_for_slide(query, slide_type, i + 1, prior_titles)
                slide_chunks = self.retriever.search(sub_q, top_k=5)
                if not slide_chunks:
                    slide_chunks = context_chunks
            else:
                slide_chunks = context_chunks

            context_text = self._prepare_context(slide_chunks)

            last_err = None
            for attempt in range(3):
                prompt = _build_single_slide_prompt(
                    query, context_text, slide_type,
                    i + 1, num_slides, lang_label, prior_titles, prior_hints
                )
                if attempt > 0:
                    prompt += (
                        "\n\nCRITICAL: Your previous attempt was too similar to earlier slides. "
                        "You MUST change the title and ALL 3 bullets to be substantially different.\n"
                    )
                try:
                    raw = self.llm.generate(query, slide_chunks, prompt_override=prompt)
                    slide = _extract_slide_json(raw)
                    if not slide:
                        last_err = "extraction failed"
                        continue
                    fp = _slide_fingerprint(slide)
                    title_norm = _norm_title(slide.get("title", ""))
                    if fp in prior_fps or (title_norm and title_norm in {_norm_title(t) for t in prior_titles}):
                        last_err = "duplicate slide"
                        continue
                    if i == 0:
                        topic = _extract_topic(raw, query[:40])
                    slides.append(slide)
                    prior_fps.add(fp)
                    t = (slide.get("title") or "").strip()
                    if t:
                        prior_titles.append(t)
                    h = (slide.get("visual_hint") or "none").strip().lower()
                    prior_hints.append(h)
                    break
                except Exception as e:
                    last_err = str(e)
                    continue
            else:
                log.warning(f"Slide {i+1} skipped (could not generate unique slide: {last_err})")

        return {"topic": topic, "slides": slides}

