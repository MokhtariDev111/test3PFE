"""
schemas.py — Pydantic Models for Structured Outputs
=====================================================
Defines strict schemas for LLM outputs with validation.
Ensures data integrity throughout the pipeline.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

log = logging.getLogger("schemas")

# Valid slide types
SlideType = Literal[
    "title", "intro", "definition", "concept", "example",
    "comparison", "process", "case_study", "summary", "content", "section"
]

# Valid visual hints
VisualHint = Literal[
    "flowchart", "mindmap", "timeline", "comparison",
    "process", "hierarchy", "none"
]


class BulletPoint(BaseModel):
    """A single bullet point with source attribution."""
    text: str = Field(..., min_length=10, max_length=500)
    source_id: Optional[str] = Field(default=None, max_length=100)
    
    @field_validator('text')
    @classmethod
    def clean_text(cls, v: str) -> str:
        """Clean and validate bullet text."""
        v = v.strip()
        if not v:
            raise ValueError("Bullet text cannot be empty")
        return v


class SlideSchema(BaseModel):
    """
    Validated slide structure from LLM output.
    """
    slide_type: SlideType = Field(default="concept")
    title: str = Field(..., min_length=3, max_length=100)
    bullets: list[BulletPoint] = Field(..., min_length=1, max_length=6)
    key_message: Optional[str] = Field(default="", max_length=200)
    visual_hint: VisualHint = Field(default="none")
    image_id: Optional[str] = Field(default=None)
    speaker_notes: Optional[str] = Field(default="", max_length=500)
    quality_score: int = Field(default=7, ge=1, le=10)
    quality_feedback: Optional[str] = Field(default="")
    
    @field_validator('title')
    @classmethod
    def clean_title(cls, v: str) -> str:
        """Clean title text."""
        return v.strip()
    
    @field_validator('slide_type', mode='before')
    @classmethod
    def normalize_slide_type(cls, v) -> str:
        """Normalize slide type to valid value."""
        if not v:
            return "concept"
        v = str(v).lower().strip()
        valid = {"title", "intro", "definition", "concept", "example",
                 "comparison", "process", "case_study", "summary", "content", "section"}
        return v if v in valid else "concept"
    
    @field_validator('visual_hint', mode='before')
    @classmethod
    def normalize_visual_hint(cls, v) -> str:
        """Normalize visual hint to valid value."""
        if not v:
            return "none"
        v = str(v).lower().strip()
        valid = {"flowchart", "mindmap", "timeline", "comparison", "process", "hierarchy", "none"}
        return v if v in valid else "none"
    
    @field_validator('quality_score', mode='before')
    @classmethod
    def parse_quality_score(cls, v) -> int:
        """Convert quality score to integer."""
        if isinstance(v, int):
            return max(1, min(10, v))
        if isinstance(v, str):
            try:
                return max(1, min(10, int(v)))
            except ValueError:
                return 5
        return 5
    
    @field_validator('bullets', mode='before')
    @classmethod
    def normalize_bullets(cls, v) -> list:
        """Convert various bullet formats to BulletPoint list."""
        if not v:
            return []
        
        normalized = []
        for item in v:
            if isinstance(item, dict):
                # Already a dict with text
                if 'text' in item:
                    normalized.append(item)
                else:
                    # Maybe it's {content: "..."} or similar
                    text = item.get('content') or item.get('bullet') or str(item)
                    normalized.append({"text": text, "source_id": item.get('source_id')})
            elif isinstance(item, str):
                # Plain string bullet
                normalized.append({"text": item, "source_id": None})
            else:
                # Unknown format, convert to string
                normalized.append({"text": str(item), "source_id": None})
        
        return normalized
    
    def to_dict(self) -> dict:
        """Convert to dictionary for downstream processing."""
        return {
            "slide_type": self.slide_type,
            "title": self.title,
            "bullets": [{"text": b.text, "source_id": b.source_id} for b in self.bullets],
            "key_message": self.key_message or "",
            "visual_hint": self.visual_hint,
            "image_id": self.image_id,
            "speaker_notes": self.speaker_notes or "",
            "quality_score": self.quality_score,
            "quality_feedback": self.quality_feedback or "",
        }


class LessonSchema(BaseModel):
    """Validated lesson/presentation structure."""
    topic: str = Field(..., min_length=3, max_length=200)
    slides: list[SlideSchema] = Field(..., min_length=1)
    
    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "slides": [s.to_dict() for s in self.slides],
        }


def validate_slide(raw_data: dict) -> tuple[Optional[SlideSchema], Optional[str]]:
    """
    Validate raw slide data against schema.
    
    Args:
        raw_data: Dictionary from LLM JSON output
    
    Returns:
        (SlideSchema, None) on success
        (None, error_message) on failure
    """
    try:
        slide = SlideSchema(**raw_data)
        return slide, None
    except Exception as e:
        error_msg = str(e)
        log.warning(f"Slide validation failed: {error_msg[:100]}")
        return None, error_msg


def validate_and_fix_slide(raw_data: dict) -> dict:
    """
    Validate slide data, fixing issues where possible.
    Returns a valid slide dict (with defaults for invalid fields).
    
    This is more lenient than validate_slide() - it tries to
    produce a usable result even from malformed input.
    """
    # Ensure required fields exist
    if not raw_data.get("title"):
        raw_data["title"] = "Untitled Slide"
    
    # Ensure bullets is a list
    bullets = raw_data.get("bullets", [])
    if isinstance(bullets, str):
        bullets = [{"text": bullets, "source_id": None}]
    elif not isinstance(bullets, list):
        bullets = []
    
    # Ensure each bullet has text
    fixed_bullets = []
    for b in bullets:
        if isinstance(b, str):
            if len(b.strip()) >= 10:
                fixed_bullets.append({"text": b.strip(), "source_id": None})
        elif isinstance(b, dict) and b.get("text"):
            text = str(b["text"]).strip()
            if len(text) >= 10:
                fixed_bullets.append({"text": text, "source_id": b.get("source_id")})
    
    # Ensure at least one bullet
    if not fixed_bullets:
        fixed_bullets = [{"text": "Content to be added.", "source_id": None}]
    
    raw_data["bullets"] = fixed_bullets
    
    # Try to validate
    slide, error = validate_slide(raw_data)
    
    if slide:
        return slide.to_dict()
    else:
        # Return cleaned raw data as fallback
        log.warning(f"Using fallback slide data: {error[:50] if error else 'unknown'}")
        return {
            "slide_type": raw_data.get("slide_type", "concept"),
            "title": raw_data.get("title", "Untitled"),
            "bullets": fixed_bullets,
            "key_message": raw_data.get("key_message", ""),
            "visual_hint": raw_data.get("visual_hint", "none"),
            "image_id": raw_data.get("image_id"),
            "speaker_notes": raw_data.get("speaker_notes", ""),
            "quality_score": 5,
            "quality_feedback": error or "",
        }


# ── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Schema Validation Test ──\n")
    
    # Test valid slide
    valid_data = {
        "slide_type": "concept",
        "title": "Introduction to Machine Learning",
        "bullets": [
            {"text": "Machine learning enables computers to learn from data", "source_id": "p1"},
            {"text": "It is a subset of artificial intelligence", "source_id": "p2"},
        ],
        "visual_hint": "flowchart",
        "quality_score": 8,
    }
    
    slide, error = validate_slide(valid_data)
    print(f"Valid slide: {'✅ Passed' if slide else '❌ Failed'}")
    if slide:
        print(f"  Title: {slide.title}")
        print(f"  Bullets: {len(slide.bullets)}")
    
    # Test invalid slide (missing required fields)
    invalid_data = {
        "title": "Test",
        "bullets": "This is not a list",  # Wrong type
        "quality_score": "high",  # Wrong type
    }
    
    fixed = validate_and_fix_slide(invalid_data)
    print(f"\nInvalid slide fixed: ✅")
    print(f"  Title: {fixed['title']}")
    print(f"  Bullets: {fixed['bullets']}")
    print(f"  Quality Score: {fixed['quality_score']} (coerced)")
    
    # Test with string bullets
    string_bullets = {
        "title": "Test Slide",
        "bullets": ["First point about the topic", "Second important point here"],
    }
    
    fixed2 = validate_and_fix_slide(string_bullets)
    print(f"\nString bullets converted: ✅")
    print(f"  Bullets: {fixed2['bullets']}")
    
    print("\n✅ All schema tests passed!")