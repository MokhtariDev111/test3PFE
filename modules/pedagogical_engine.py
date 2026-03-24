"""
pedagogical_engine.py — v5: Quality gate + LLM judge + parallel generation
===========================================================================
Changes from v4:
  1. Quality gate: slides with quality_score < threshold (config) are rejected
     and the quality_feedback is injected into the retry prompt so the LLM
     knows exactly what was wrong.
  2. LLM-as-judge: after each accepted slide, a second LLM call (judge model)
     independently scores the slide. If the judge score is also below threshold,
     the slide is rejected and retried. Controlled by config llm.judge_enabled.
  3. Parallel generation: when config generation.parallel=true, all N slide
     coroutines are launched with asyncio.gather() simultaneously. Total wait
     time drops from (N × latency) to (~1 × latency). A sequential dedup pass
     runs after to catch cross-slide repetition.
  4. _repair_json kept as fallback but rarely needed now that format:json is
     used in llm.py.
  5. model_override passed to llm.generate_async so different models can be
     used for generation vs judging (configured in config.yaml).
"""

import asyncio
import json
import logging
import re
from pathlib import Path
import sys
import hashlib

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.config_loader import CONFIG
from modules.llm import LLMEngine

log = logging.getLogger("pedagogical_engine")
if not log.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

# ── Pedagogical arc ───────────────────────────────────────────────────────────
# Slide types in teaching order: define → explain → illustrate → compare → summarize
# Slide types follow the user's preferred curriculum order
SLIDE_ARC = [
    "intro", "definition", "concept", "example", "process", 
    "comparison", "case_study", "summary", "concept", "example",
    "process", "comparison", "summary"
]

# ── JSON schema shown in every prompt ─────────────────────────────────────────
_ONE_SLIDE_SCHEMA = """{
  "slide_type": "concept",
  "title": "Slide Title Here",
  "bullets": [
    {"text": "Specific fact, mechanism, or data point with concrete detail", "source_id": "Page X or Source Name"},
    {"text": "Second distinct, non-overlapping point advancing the lesson", "source_id": "Page Y or Source Name"},
    {"text": "Third point — application, implication, or real-world example", "source_id": "Page Z or Source Name"}
  ],
  "key_message": "One-sentence high-level synthesis NOT present in the bullets",
  "visual_hint": "flowchart",
  "image_id": "Exact Image ID from list or null",
  "speaker_notes": "One or two short sentences explaining the 'why' behind the facts.",
  "quality_score": 8,
  "quality_feedback": "Brief reason why this score was given, focusing on specificity and no repetition."
}"""

# ── Sub-query templates (per-slide targeted retrieval) ────────────────────────
_SLIDE_TYPE_SUBQUERY = {
    "definition": "definition and meaning of {query}",
    "concept":    "key concepts and principles of {query}",
    "example":    "real-world examples and applications of {query}",
    "comparison": "comparison and differences related to {query}",
    "summary":    "summary and main takeaways of {query}",
}


def _subquery_for_slide(query: str, slide_type: str, prior_titles: list[str]) -> str:
    template = _SLIDE_TYPE_SUBQUERY.get(slide_type, "{query}")
    base = template.format(query=query)
    if prior_titles:
        covered = ", ".join(prior_titles[-3:])
        base += f" (do not repeat: {covered})"
    return base


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_slide_prompt(
    query: str,
    context_text: str,
    slide_type: str,
    slide_number: int,
    total: int,
    language: str,
    prior_titles: list[str],
    prior_hints: list[str],
    quality_feedback: str = "",
    available_images: list[str] | None = None,
    image_contexts: dict[str, str] | None = None,
) -> str:
    prior_str  = "\n".join(f"- {t}" for t in prior_titles[-6:]) or "(none)"
    used_hints = ", ".join(prior_hints[-4:]) or "(none)"
    all_hints  = ["flowchart", "mindmap", "timeline", "comparison", "process", "hierarchy"]
    suggested  = next(
        (h for h in all_hints if not prior_hints or h not in prior_hints[-3:]),
        all_hints[slide_number % len(all_hints)],
    )

    # If a previous attempt had low quality, include the feedback
    quality_note = ""
    if quality_feedback:
        quality_note = (
            f"\n\nIMPORTANT — previous attempt was rejected for low quality:\n"
            f"  Reason: {quality_feedback}\n"
            f"  Fix this specifically. Make bullets more concrete and specific.\n"
        )

    # PDF images: tell LLM it can embed one if it matches slide content
    image_rule = ""
    if available_images:
        ctx = image_contexts or {}
        img_lines = []
        for x in available_images[:12]:
            desc = ctx.get(x, "")
            if desc:
                img_lines.append(f'  - "{x}" (context: {desc[:120]}...)')
            else:
                img_lines.append(f'  - "{x}"')
        img_block = "\n".join(img_lines)
        image_rule = f"""
10. image_id: PROMPTING PRIORITY — If an embedded PDF image matches this slide's content, 
    you MUST set image_id to its exact ID. Available:
{img_block}
    If image_id is used, set visual_hint to "none". Otherwise use null for image_id."""

    return f"""You are an expert AI teaching assistant. Output ONLY a valid JSON object for ONE slide.

Generate slide {slide_number} of {total} about: {query}
Slide type: {slide_type}
Language: {language}
{quality_note}
JSON schema:
{_ONE_SLIDE_SCHEMA}

Rules:
1. slide_type must be exactly: {slide_type}
2. **CONTENT CREATIVITY**: Use analogies and fresh examples. DO NOT copy text verbatim from the context. Re-explain in a teaching tone.
3. **CONTENT DEPTH**: Every bullet MUST contain a specific fact, date, statistic, or mechanism. No generic filler.
4. **NO REPETITION**: Check previous slides: {prior_str}. DO NOT repeat information or sentence structures.
5. **KEY MESSAGE**: A strategic takeaway or analogy that summarizes the slide's purpose.
6. **VISUAL HINT**: Mandatory. Suggested: "{suggested}". Choose from: flowchart, mindmap, timeline, comparison, process, hierarchy, none.
7. **MOTION HINTS**: For each bullet, focus on a "stepwise build". Mention in speaker_notes how the animator should handle the build for maximum impact.
8. All content in {language}.
9. Output starts with {{ and ends with }}. No markdown. No conversational filler.
{image_rule}

Context from documents:
{context_text}

Topic: {query}

JSON:"""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _repair_json(raw: str) -> str:
    """Fallback repair for when format:json didn't fully work."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    raw = raw.replace(": None", ": null").replace(": True", ": true").replace(": False", ": false")
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    brace_start = raw.find("{")
    if brace_start > 0:
        raw = raw[brace_start:]
    brace_end = raw.rfind("}")
    if brace_end != -1 and brace_end < len(raw) - 1:
        raw = raw[:brace_end + 1]
    return raw


def _extract_slide_json(raw: str) -> dict | None:
    if not raw or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
        if "title" in obj and "bullets" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: try after repair
    try:
        obj = json.loads(_repair_json(raw))
        if "title" in obj and "bullets" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Last resort: regex extract
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            if "title" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    log.error(f"Could not parse slide JSON from: {raw[:200]!r}")
    return None


# ── Deduplication helpers ─────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _slide_fingerprint(slide: dict) -> str:
    title = _norm(slide.get("title", ""))
    bullets = slide.get("bullets", [])
    if not isinstance(bullets, list):
        bullets = [bullets] if bullets else []
    btxt = []
    for b in bullets:
        if isinstance(b, dict):
            btxt.append(_norm(b.get("text", "")))
        else:
            btxt.append(_norm(str(b)))
    key = title + "|" + "|".join(btxt[:3])
    return hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()


# ── Context preparation ───────────────────────────────────────────────────────

def _prepare_context(context_chunks: list, max_chars: int = 2400) -> str:
    seen = set()
    unique = []
    for c in context_chunks:
        key = c.text.strip()[:120]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    text = "\n".join(f"[{c.source} p{c.page}] {c.text}" for c in unique)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[truncated]"
    return text


# ── Main engine ───────────────────────────────────────────────────────────────

class PedagogicalEngine:
    def __init__(self, retriever=None, model_override: str = None):
        self.llm            = LLMEngine()
        self.retriever      = retriever
        self.model_override = model_override  # per-request model, avoids global mutation
        self.quality_threshold = CONFIG.get("generation", {}).get("quality_threshold", 5)
        self.parallel          = CONFIG.get("generation", {}).get("parallel", False)

    # ── Single slide generation (used by both sequential and parallel paths) ──

    async def _generate_one_slide(
        self,
        query: str,
        context_chunks: list,
        slide_index: int,
        slide_type: str,
        num_slides: int,
        language: str,
        prior_titles: list[str],
        prior_hints: list[str],
        prior_fps: set,
        available_images: list[str] | None = None,
        image_contexts: dict[str, str] | None = None,
    ) -> dict | None:
        """
        Generates one slide with up to 3 attempts.
        Returns the slide dict, or None if all attempts failed.
        """
        if self.retriever is not None:
            sub_q  = _subquery_for_slide(query, slide_type, prior_titles)
            chunks = self.retriever.search(sub_q, top_k=5) or context_chunks
        else:
            chunks = context_chunks

        context_text = _prepare_context(chunks)
        last_quality_feedback = ""

        for attempt in range(3):
            prompt = _build_slide_prompt(
                query, context_text, slide_type,
                slide_index + 1, num_slides, language,
                prior_titles, prior_hints,
                quality_feedback=last_quality_feedback if attempt > 0 else "",
                available_images=available_images,
                image_contexts=image_contexts,
            )

            if attempt == 2:
                prompt += (
                    "\n\nFINAL ATTEMPT: Previous attempts were rejected. "
                    "You MUST write entirely new content — different title, "
                    "different bullets, more specific facts.\n"
                )

            try:
                raw   = await self.llm.generate_async(
                    query, chunks,
                    prompt_override=prompt,
                    model_override=self.model_override,
                )
                slide = _extract_slide_json(raw)

                if not slide:
                    last_quality_feedback = "JSON parsing failed"
                    continue

                # Self-reported quality gate
                self_score = int(slide.get("quality_score", 5))
                if self_score < self.quality_threshold:
                    last_quality_feedback = (
                        slide.get("quality_feedback", "")
                        or f"quality_score {self_score} below threshold {self.quality_threshold}"
                    )
                    log.info(f"  Slide {slide_index+1} self-score {self_score} too low: {last_quality_feedback[:60]}")
                    continue

                # Fingerprint dedup
                fp = _slide_fingerprint(slide)
                title_norm  = _norm(slide.get("title", ""))
                prior_norms = {_norm(t) for t in prior_titles}
                if fp in prior_fps or (title_norm and title_norm in prior_norms):
                    last_quality_feedback = "too similar to a previous slide"
                    log.info(f"  Slide {slide_index+1} rejected — duplicate")
                    continue

                # LLM-as-judge — only if enabled in config
                if self.llm.judge_enabled:
                    judgment   = await self.llm.judge_async(slide)
                    judge_score = judgment.get("score", 7)
                    if judge_score < self.quality_threshold:
                        last_quality_feedback = (
                            judgment.get("feedback", "")
                            or f"judge score {judge_score} below threshold"
                        )
                        log.info(f"  Slide {slide_index+1} rejected by judge ({judge_score}/10)")
                        continue
                    slide["judge_score"]    = judge_score
                    slide["judge_feedback"] = judgment.get("feedback", "")

                log.info(f"  ✔ Slide {slide_index+1}/{num_slides}: '{slide.get('title','?')}'")
                return slide

            except Exception as e:
                last_quality_feedback = f"error: {e}"
                log.warning(f"  Slide {slide_index+1} attempt {attempt+1} error: {e}")
                continue

        log.warning(f"  ✗ Slide {slide_index+1}: all attempts failed — skipping.")
        return None

    # ── Async generation entry point ──────────────────────────────────────────

    async def generate_lesson_async(
        self,
        query: str,
        context_chunks: list,
        num_slides: int = 5,
        language: str = "English",
        available_images: list = None,
        image_contexts: dict[str, str] | None = None,
    ) -> dict:
        if not context_chunks:
            return {"topic": query, "slides": [], "_error": "No context retrieved."}

        lang_label = "French" if language.lower() in ("fr", "french", "français") else "English"
        topic      = query[:40]

        if self.parallel:
            return await self._generate_parallel(
                query, context_chunks, num_slides, lang_label, topic,
                available_images=available_images, image_contexts=image_contexts,
            )
        else:
            return await self._generate_sequential(
                query, context_chunks, num_slides, lang_label, topic,
                available_images=available_images, image_contexts=image_contexts,
            )

    async def _generate_parallel(
        self, query, context_chunks, num_slides, lang_label, topic,
        available_images=None, image_contexts=None,
    ) -> dict:
        log.info(f"Parallel generation: {num_slides} slides for '{query}'")

        # Each coroutine gets its own hint list snapshot to avoid race conditions
        tasks = []
        for i in range(num_slides):
            slide_type = SLIDE_ARC[i % len(SLIDE_ARC)]
            tasks.append(
                self._generate_one_slide(
                    query, context_chunks,
                    slide_index=i,
                    slide_type=slide_type,
                    num_slides=num_slides,
                    language=lang_label,
                    prior_titles=[],   # dedup runs post-gather
                    prior_hints=[],
                    prior_fps=set(),
                    available_images=available_images,
                    image_contexts=image_contexts,
                )
            )

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        slides      = []
        seen_fps    = set()
        seen_titles = set()
        prior_hints = []
        for result in raw_results:
            if isinstance(result, Exception) or result is None:
                continue
            fp    = _slide_fingerprint(result)
            title = _norm(result.get("title", ""))
            if fp in seen_fps or (title and title in seen_titles):
                log.info(f"  Post-dedup removed: '{result.get('title','')}'")
                continue
            seen_fps.add(fp)
            if title:
                seen_titles.add(title)
            slides.append(result)
            prior_hints.append((result.get("visual_hint") or "none").lower())

        log.info(f"Parallel done: {len(slides)}/{num_slides} unique slides.")
        return {"topic": topic, "slides": slides}

    async def _generate_sequential(
        self, query, context_chunks, num_slides, lang_label, topic,
        available_images=None, image_contexts=None,
    ) -> dict:
        log.info(f"Sequential generation: {num_slides} slides for '{query}'")

        slides       = []
        prior_titles = []
        prior_fps    = set()
        prior_hints  = []

        for i in range(num_slides):
            slide_type = SLIDE_ARC[i % len(SLIDE_ARC)]
            result = await self._generate_one_slide(
                query, context_chunks,
                slide_index=i,
                slide_type=slide_type,
                num_slides=num_slides,
                language=lang_label,
                prior_titles=prior_titles,
                prior_hints=prior_hints,
                prior_fps=prior_fps,
                available_images=available_images,
                image_contexts=image_contexts,
            )
            if result:
                slides.append(result)
                prior_fps.add(_slide_fingerprint(result))
                t = (result.get("title") or "").strip()
                if t:
                    prior_titles.append(t)
                prior_hints.append((result.get("visual_hint") or "none").lower())

        log.info(f"Sequential done: {len(slides)}/{num_slides} slides.")
        return {"topic": topic, "slides": slides}

    # ── Sync fallback (CLI only) ──────────────────────────────────────────────

    def generate_lesson(self, query, context_chunks, num_slides=5, language="English", available_images=None):
        """Sync wrapper for CLI testing only."""
        import asyncio
        return asyncio.run(
            self.generate_lesson_async(
                query, context_chunks, num_slides, language,
                available_images, image_contexts=None,
            )
        )
