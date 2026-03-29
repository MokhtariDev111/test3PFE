"""
visual_strategy.py — Intelligent Visual Assignment
===================================================
Decides WHEN to include images/diagrams based on slide content.
Prevents unnecessary visuals that reduce clarity.
"""

import logging
from typing import Literal

log = logging.getLogger("visual_strategy")

VisualDecision = Literal["diagram", "image", "both", "none"]

# Slide types that should NEVER have visuals
NO_VISUAL_TYPES = {"title"}

# Slide types that RARELY need visuals
MINIMAL_VISUAL_TYPES = {"intro", "summary", "definition"}

# Keywords that STRONGLY suggest diagrams
DIAGRAM_KEYWORDS = {
    "process", "workflow", "steps", "flow", "pipeline",
    "architecture", "structure", "system",
    "hierarchy", "tree", "organization",
    "comparison", "vs", "versus", "compare", "differ",
    "timeline", "evolution", "history", "phases",
    "cycle", "loop", "iterative",
    "relationship", "connection", "interaction",
    "layer", "stack", "component"
}

# Keywords that suggest images would help
IMAGE_KEYWORDS = {
    "example", "case study", "illustration", "demonstration",
    "screenshot", "photo", "picture", "visual",
    "interface", "ui", "dashboard", "screen"
}

# Keywords that suggest NO visual is needed
NO_VISUAL_KEYWORDS = {
    "definition", "meaning", "what is", "overview",
    "introduction", "summary", "conclusion",
    "key points", "takeaways", "recap"
}


def decide_visual(slide: dict) -> VisualDecision:
    """
    Decide what visual (if any) should accompany a slide.
    
    Args:
        slide: Slide dictionary with title, bullets, slide_type, visual_hint
    
    Returns:
        "diagram" | "image" | "both" | "none"
    """
    slide_type = (slide.get("slide_type") or "content").lower().strip()
    title = (slide.get("title") or "").lower()
    visual_hint = (slide.get("visual_hint") or "none").lower().strip()
    
    # Build content string for keyword matching
    bullets = slide.get("bullets", [])
    bullet_text = " ".join(
        b.get("text", "") if isinstance(b, dict) else str(b)
        for b in bullets
    ).lower()
    content = f"{title} {bullet_text}"
    
    # Rule 1: Title slides never have visuals
    if slide_type in NO_VISUAL_TYPES:
        return "none"
    
    # Rule 2: Check for explicit "no visual" keywords
    if any(kw in content for kw in NO_VISUAL_KEYWORDS):
        # Even if there's a visual_hint, skip for these
        if slide_type in MINIMAL_VISUAL_TYPES:
            return "none"
    
    # Rule 3: Intro/summary slides only get visuals if VERY relevant
    if slide_type in MINIMAL_VISUAL_TYPES:
        diagram_score = sum(1 for kw in DIAGRAM_KEYWORDS if kw in content)
        if diagram_score >= 3:  # Strong signal
            return "diagram"
        return "none"
    
    # Rule 4: Check visual_hint from LLM
    if visual_hint not in ("none", ""):
        # LLM suggested a diagram type — validate it
        diagram_score = sum(1 for kw in DIAGRAM_KEYWORDS if kw in content)
        if diagram_score >= 2:
            return "diagram"
        # LLM suggested but content doesn't support it
        if diagram_score == 0:
            log.info(f"Ignoring visual_hint={visual_hint} for '{title[:30]}' (no diagram keywords)")
            return "none"
        return "diagram"
    
    # Rule 5: Check for image-worthy content
    has_image = slide.get("image_id") is not None
    if has_image:
        image_score = sum(1 for kw in IMAGE_KEYWORDS if kw in content)
        if image_score >= 1:
            return "image"
    
    # Rule 6: Auto-detect diagram need based on keywords
    diagram_score = sum(1 for kw in DIAGRAM_KEYWORDS if kw in content)
    if diagram_score >= 3:
        return "diagram"
    
    # Rule 7: Default — no visual for factual content
    return "none"


def validate_image_relevance(
    slide: dict, 
    image_context: str, 
    threshold: float = 0.25
) -> bool:
    """
    Check if an image is actually relevant to slide content.
    
    Args:
        slide: Slide dictionary
        image_context: Text description/context of the image
        threshold: Minimum similarity score (0-1)
    
    Returns:
        True if image should be included
    """
    if not image_context:
        return False
    
    title = (slide.get("title") or "").lower()
    bullets = slide.get("bullets", [])
    bullet_text = " ".join(
        b.get("text", "") if isinstance(b, dict) else str(b)
        for b in bullets
    ).lower()
    
    slide_content = f"{title} {bullet_text}"
    slide_words = set(slide_content.split())
    image_words = set(image_context.lower().split())
    
    # Remove stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "and", "or", "but", "if", "then", "so", "than", "that", "this",
        "it", "its", "they", "them", "their", "we", "our", "you", "your"
    }
    slide_words -= stop_words
    image_words -= stop_words
    
    if not slide_words or not image_words:
        return False
    
    # Calculate Jaccard similarity
    intersection = len(slide_words & image_words)
    union = len(slide_words | image_words)
    similarity = intersection / union if union > 0 else 0
    
    # Also check for key concept overlap
    # If slide title words appear in image context, boost relevance
    title_words = set(title.split()) - stop_words
    title_overlap = len(title_words & image_words)
    
    if title_overlap >= 2:
        similarity += 0.15
    
    log.debug(f"Image relevance for '{title[:30]}': {similarity:.2f} (threshold: {threshold})")
    
    return similarity >= threshold


def filter_slides_for_diagrams(slides: list) -> list[int]:
    """
    Return indices of slides that should have diagrams.
    Limits total diagrams to avoid visual overload.
    
    Args:
        slides: List of slide dictionaries
    
    Returns:
        List of slide indices that should have diagrams
    """
    diagram_candidates = []
    
    for i, slide in enumerate(slides):
        decision = decide_visual(slide)
        if decision in ("diagram", "both"):
            # Score the slide for diagram importance
            content = f"{slide.get('title', '')} {' '.join(str(b) for b in slide.get('bullets', []))}".lower()
            score = sum(1 for kw in DIAGRAM_KEYWORDS if kw in content)
            diagram_candidates.append((i, score))
    
    # Sort by score descending
    diagram_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Limit to max 3 diagrams per presentation (or 40% of slides)
    max_diagrams = min(3, max(1, len(slides) * 2 // 5))
    
    selected = [idx for idx, _ in diagram_candidates[:max_diagrams]]
    
    log.info(f"Selected {len(selected)} slides for diagrams out of {len(slides)} total")
    
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("── Visual Strategy Test ──\n")
    
    test_slides = [
        {
            "title": "Introduction to Machine Learning",
            "slide_type": "intro",
            "bullets": [{"text": "Overview of ML concepts"}],
            "visual_hint": "mindmap"
        },
        {
            "title": "Neural Network Architecture",
            "slide_type": "concept",
            "bullets": [
                {"text": "Input layer receives data"},
                {"text": "Hidden layers process information"},
                {"text": "Output layer produces results"}
            ],
            "visual_hint": "hierarchy"
        },
        {
            "title": "Training Process Steps",
            "slide_type": "process",
            "bullets": [
                {"text": "Step 1: Data preprocessing"},
                {"text": "Step 2: Model initialization"},
                {"text": "Step 3: Forward propagation"},
                {"text": "Step 4: Backpropagation"},
                {"text": "Step 5: Optimization"}
            ],
            "visual_hint": "flowchart"
        },
        {
            "title": "Key Definitions",
            "slide_type": "definition",
            "bullets": [{"text": "Supervised learning: Learning from labeled data"}],
            "visual_hint": "none"
        },
        {
            "title": "Summary",
            "slide_type": "summary",
            "bullets": [{"text": "Key takeaways from this presentation"}],
            "visual_hint": "mindmap"
        }
    ]
    
    for slide in test_slides:
        decision = decide_visual(slide)
        print(f"'{slide['title'][:40]}'")
        print(f"  Type: {slide['slide_type']}, Hint: {slide['visual_hint']}")
        print(f"  Decision: {decision}")
        print()
    
    print("Slides selected for diagrams:", filter_slides_for_diagrams(test_slides))
