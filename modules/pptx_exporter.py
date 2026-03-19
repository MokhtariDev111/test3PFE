"""
pptx_exporter.py  —  Step 10 (Upgraded): PowerPoint Export with Themes & Diagrams
==================================================================================
Receives: list[SlideData] + optional ThemeConfig + optional diagram image paths.
Action:   Creates a beautifully styled .pptx with:
          - 5 colour themes (Dark Navy, Crimson, Forest, Purple, White)
          - Split layout on slides that have a diagram image (text left, image right)
          - Speaker notes embedded in the PowerPoint notes pane
Returns:  The absolute path to the saved .pptx file.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import PP_PLACEHOLDER

try:
    from modules.slide_generator import SlideData
except ImportError:
    pass

from modules.config_loader import CONFIG

log = logging.getLogger("pptx_exporter")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


# ── Colour Themes ──────────────────────────────────────────────────────────────
@dataclass
class ThemeConfig:
    name: str
    bg:          RGBColor
    accent:      RGBColor
    text:        RGBColor
    subtext:     RGBColor
    bullet:      RGBColor

THEMES: dict[str, ThemeConfig] = {
    "Dark Navy": ThemeConfig(
        name="Dark Navy",
        bg=RGBColor(0x0D, 0x1B, 0x2A),
        accent=RGBColor(0x1F, 0x6F, 0xEB),
        text=RGBColor(0xFF, 0xFF, 0xFF),
        subtext=RGBColor(0xCC, 0xD6, 0xE0),
        bullet=RGBColor(0x7E, 0xC8, 0xFF),
    ),
    "Crimson Pro": ThemeConfig(
        name="Crimson Pro",
        bg=RGBColor(0x1A, 0x07, 0x07),
        accent=RGBColor(0xC0, 0x39, 0x2B),
        text=RGBColor(0xFF, 0xFF, 0xFF),
        subtext=RGBColor(0xF5, 0xC6, 0xC6),
        bullet=RGBColor(0xFF, 0x8A, 0x80),
    ),
    "Forest Green": ThemeConfig(
        name="Forest Green",
        bg=RGBColor(0x07, 0x18, 0x0D),
        accent=RGBColor(0x27, 0xAE, 0x60),
        text=RGBColor(0xFF, 0xFF, 0xFF),
        subtext=RGBColor(0xC3, 0xE6, 0xC3),
        bullet=RGBColor(0x82, 0xEF, 0xAA),
    ),
    "Midnight Purple": ThemeConfig(
        name="Midnight Purple",
        bg=RGBColor(0x0F, 0x08, 0x1A),
        accent=RGBColor(0x8E, 0x44, 0xAD),
        text=RGBColor(0xFF, 0xFF, 0xFF),
        subtext=RGBColor(0xD7, 0xC4, 0xE8),
        bullet=RGBColor(0xCC, 0x99, 0xFF),
    ),
    "Corporate White": ThemeConfig(
        name="Corporate White",
        bg=RGBColor(0xFA, 0xFA, 0xFC),
        accent=RGBColor(0x2C, 0x3E, 0x50),
        text=RGBColor(0x1A, 0x1A, 0x2E),
        subtext=RGBColor(0x55, 0x66, 0x77),
        bullet=RGBColor(0x2C, 0x3E, 0x50),
    ),
}
DEFAULT_THEME = THEMES["Dark Navy"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _set_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _add_textbox(slide, text: str, left, top, width, height,
                 font_size: int, bold: bool, color: RGBColor,
                 align=PP_ALIGN.LEFT, word_wrap: bool = True):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = word_wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = color
    return txBox


def _add_rect(slide, left, top, width, height, fill_color: RGBColor):
    """Add a solid color rectangle (using MSO_SHAPE_TYPE 1 = Rectangle)."""
    from pptx.util import Emu
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()   # no border
    return shape


def _add_transition(slide):
    from pptx.oxml import parse_xml
    # Inject smooth fade transition for more "life" and impressiveness
    transition_xml = '<p:transition xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" spd="med"><p:fade/></p:transition>'
    slide.element.insert(-1, parse_xml(transition_xml))

# ── Slide Builders ────────────────────────────────────────────────────────────
def build_title_slide(prs: Presentation, slide_data: SlideData, theme: ThemeConfig):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_transition(slide)
    _set_bg(slide, theme.bg)
    W = prs.slide_width
    H = prs.slide_height

    # Vertical left accent bar
    _add_rect(slide, Inches(0), Inches(0), Inches(0.15), H, theme.accent)
    # Decorative bottom strip
    _add_rect(slide, Inches(0), H - Inches(0.12), W, Inches(0.12), theme.accent)

    _add_textbox(slide, slide_data.title,
                 left=Inches(0.4), top=Inches(2.6), width=Inches(8.8), height=Inches(1.6),
                 font_size=42, bold=True, color=theme.text)

    _add_textbox(slide, f"{datetime.now().strftime('%B %d, %Y')} • Generated by EduGenius AI",
                 left=Inches(0.4), top=Inches(4.3), width=Inches(8), height=Inches(0.6),
                 font_size=16, bold=False, color=theme.subtext)


def build_content_slide(prs: Presentation, slide_data: SlideData,
                        theme: ThemeConfig, diagram_path: str = None, pdf_images: dict = None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_transition(slide)
    _set_bg(slide, theme.bg)
    W = prs.slide_width
    H = prs.slide_height

    # Header bar
    _add_rect(slide, Inches(0), Inches(0), W, Inches(1.15), theme.accent)
    _add_textbox(slide, slide_data.title,
                 left=Inches(0.3), top=Inches(0.18), width=Inches(9), height=Inches(0.8),
                 font_size=26, bold=True, color=theme.text)

    # Decide layout: split (text + image) or full-width text
    bullets_to_render = slide_data.bullets[:5]  # Limit to 5 to prevent slide overflow
    citations = set()
    pdf_images = pdf_images or {}
    
    has_visual = False
    visual_stream = None
    
    if diagram_path and Path(diagram_path).exists():
        has_visual = True
        visual_stream = diagram_path
    elif slide_data.image_id and slide_data.image_id in pdf_images:
        has_visual = True
        import base64
        from io import BytesIO
        img_data = base64.b64decode(pdf_images[slide_data.image_id])
        visual_stream = BytesIO(img_data)
        
    if has_visual:
        text_width = Inches(5.3)
        for i, bullet in enumerate(bullets_to_render):
            txt = bullet.get("text", "") if isinstance(bullet, dict) else str(bullet)
            src = bullet.get("source_id", "") if isinstance(bullet, dict) else ""
            if src and src.lower() not in ("", "general knowledge"):
                citations.add(src)
                
            _add_textbox(slide, f"▸  {txt}",
                         left=Inches(0.4), top=Inches(1.3 + i * 0.82), width=text_width, height=Inches(0.75),
                         font_size=16, bold=False, color=theme.bullet)
        # Image on the right (width constrained to maintain aspect ratio)
        pic = slide.shapes.add_picture(visual_stream,
                                       left=Inches(5.8), top=Inches(1.2),
                                       width=Inches(3.8))
        # Add subtle shadow for more elegance
        pic.shadow.inherit = True
    else:
        for i, bullet in enumerate(bullets_to_render):
            txt = bullet.get("text", "") if isinstance(bullet, dict) else str(bullet)
            src = bullet.get("source_id", "") if isinstance(bullet, dict) else ""
            if src and src.lower() not in ("", "general knowledge"):
                citations.add(src)
                
            _add_textbox(slide, f"▸  {txt}",
                         left=Inches(0.4), top=Inches(1.3 + i * 0.9), width=Inches(9.2), height=Inches(0.8),
                         font_size=18, bold=False, color=theme.bullet)

    # Render footnotes for citations
    if citations:
        cit_text = "Sources: " + ", ".join(list(citations)[:3])
        if len(citations) > 3: cit_text += "..."
        _add_textbox(slide, cit_text, left=Inches(0.4), top=Inches(5.15), width=Inches(9.2), height=Inches(0.35),
                     font_size=10, bold=False, color=theme.subtext)

    # Speaker notes
    notes_text = slide_data.speaker_notes
    if hasattr(slide_data, "quality_score") and slide_data.quality_score:
        notes_text = f"[Slide Quality: {slide_data.quality_score}/10] {slide_data.quality_feedback}\n\n{notes_text}"
        
    if notes_text and slide.has_notes_slide:
        slide.notes_slide.notes_text_frame.text = notes_text


# ── Main Export Function ──────────────────────────────────────────────────────
def export_pptx(slides: list[SlideData],
                topic: str = "Presentation",
                theme: ThemeConfig = None,
                diagrams: dict[int, str] = None,
                pdf_images: dict[str, str] = None,
                output_dir: str = None) -> str:
    """
    Builds and saves a .pptx from SlideData list.
    
    Args:
        slides:     List of SlideData objects from slide_generator.
        topic:      Descriptive name used for the output filename.
        theme:      ThemeConfig (defaults to Dark Navy).
        diagrams:   Dict mapping slide index → PNG path from diagram_generator.
        output_dir: Override the output folder.
    Returns:
        Absolute path to the saved .pptx file.
    """
    theme = theme or DEFAULT_THEME
    diagrams = diagrams or {}

    prs = Presentation()
    prs.slide_width  = Inches(10)
    prs.slide_height = Inches(5.625)  # 16:9

    for i, slide_data in enumerate(slides):
        if slide_data.slide_type == "title":
            build_title_slide(prs, slide_data, theme)
        else:
            build_content_slide(prs, slide_data, theme, diagrams.get(i), pdf_images)

    # Always use an absolute path so it works from any working directory
    if output_dir:
        out_dir = Path(output_dir).resolve()
    else:
        out_dir = ROOT_DIR / CONFIG["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sanitize the topic into a safe filename
    safe_topic = re.sub(r'[^a-zA-Z0-9_\-]', '_', topic[:35]).strip("_")
    out_path = out_dir / f"{safe_topic}_{theme.name.replace(' ','_')}_{ts}.pptx"

    prs.save(str(out_path))
    log.info(f"✔ Saved PPTX ({theme.name}): {out_path}")
    return str(out_path)
