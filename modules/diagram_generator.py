"""
diagram_generator.py  — v2: visual_hint-driven diagram selection
================================================================
Fix: reads slide.visual_hint from PedagogicalEngine output and
     generates the matching Mermaid diagram type instead of
     blindly alternating flow/mindmap by index.

Supported visual_hint values:
  flowchart  → Mermaid flowchart TD
  mindmap    → Mermaid mindmap
  timeline   → Mermaid timeline
  comparison → Mermaid quadrant or table-style flow
  process    → Mermaid flowchart with loop arrow
  hierarchy  → Mermaid flowchart TD (tree shape)
  none       → skip diagram for this slide
"""

import logging
import tempfile
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from modules.config_loader import CONFIG

log = logging.getLogger("diagram_generator")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%H:%M:%S")


# ── Mermaid generators ────────────────────────────────────────────────────────

def _safe(text: str, max_len: int = 120) -> str:
    """Escape quotes and truncate for Mermaid labels."""
    return text.replace('"', "'").replace('\n', ' ')[:max_len]


def _to_diagram_labels(bullets: list[str], max_len: int = 100) -> list[str]:
    """
    Convert bullet text into expressive, complete diagram labels.
    Removes aggressive truncation so sentences remain readable.
    """
    out = []
    seen = set()
    for b in bullets:
        s = (b if isinstance(b, str) else str(b)).strip()
        if not s:
            continue
        
        # Take first phrase if it's very long, but allow up to 15 words
        words = s.split()
        if len(words) > 15:
            s = " ".join(words[:15]) + "..."
            
        s = _safe(s, max_len)
        key = s.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _mermaid_flowchart(title: str, steps: list[str]) -> str:
    labels = _to_diagram_labels(steps) if steps else steps
    lines = ["flowchart TD"]
    ids   = [chr(65 + i) for i in range(min(len(labels), 7))]
    for nid, step in zip(ids, labels):
        lines.append(f'  {nid}["{_safe(step, 35)}"]')
    for i in range(len(ids) - 1):
        lines.append(f"  {ids[i]} --> {ids[i+1]}")
    return "\n".join(lines)


def _mermaid_mindmap(title: str, items: list[str], key_message: str = "") -> str:
    root = key_message.strip() or title
    lines = ["mindmap", f'  root(("{_safe(root, 30)}"))']
    labels = _to_diagram_labels(items)
    for item in labels[:6]:
        lines.append(f'    ("{_safe(item, 25)}")')
    return "\n".join(lines)


def _mermaid_timeline(title: str, items: list[str]) -> str:
    labels = _to_diagram_labels(items)
    lines = ["timeline", f"  title {_safe(title, 40)}"]
    for i, item in enumerate(labels[:5]):
        # Use short step labels to avoid repetition
        step_label = f"Step {i+1}" if len(labels) > 3 else item[:20]
        lines.append(f"  {step_label} : {_safe(item, 35)}")
    return "\n".join(lines)


def _mermaid_comparison(title: str, items: list[str]) -> str:
    """Two-branch flowchart — split by logical pairs (e.g. A vs B)."""
    if len(items) < 2:
        return _mermaid_flowchart(title, items)
    labels = _to_diagram_labels(items)
    mid   = (len(labels) + 1) // 2
    left  = labels[:mid]
    right = labels[mid:]
    lines = ["flowchart TD", f'  ROOT["{_safe(title, 30)}"]']
    lines.append(f'  L["{_safe(left[0], 30)}"]')
    lines.append(f'  R["{_safe(right[0], 30)}"]')
    lines.append("  ROOT --> L")
    lines.append("  ROOT --> R")
    for i, item in enumerate(left[1:], 1):
        lines.append(f'  L{i}["{_safe(item, 28)}"]')
        lines.append(f"  L --> L{i}")
    for i, item in enumerate(right[1:], 1):
        lines.append(f'  R{i}["{_safe(item, 28)}"]')
        lines.append(f"  R --> R{i}")
    return "\n".join(lines)


def _mermaid_process(title: str, steps: list[str]) -> str:
    """Cyclical process — flowchart with return arrow."""
    labels = _to_diagram_labels(steps) if steps else steps
    lines = ["flowchart TD"]
    ids   = [chr(65 + i) for i in range(min(len(labels), 6))]
    for nid, step in zip(ids, labels):
        lines.append(f'  {nid}["{_safe(step)}"]')
    for i in range(len(ids) - 1):
        lines.append(f"  {ids[i]} --> {ids[i+1]}")
    if len(ids) >= 2:
        lines.append(f"  {ids[-1]} -.->|cycle| {ids[0]}")
    return "\n".join(lines)


def _mermaid_hierarchy(title: str, items: list[str], key_message: str = "") -> str:
    """Tree hierarchy — root fans out to children."""
    root = key_message.strip() or title
    lines = ["flowchart TD", f'  ROOT["{_safe(root, 30)}"]']
    labels = _to_diagram_labels(items)
    for i, item in enumerate(labels[:6]):
        nid = f"N{i}"
        lines.append(f'  {nid}["{_safe(item)}"]')
        lines.append(f"  ROOT --> {nid}")
    return "\n".join(lines)


# ── PNG fallback (matplotlib) ─────────────────────────────────────────────────

def _flow_png(title: str, steps: list[str], color: str, out_path: str) -> str:
    n   = min(len(steps), 6)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1)
    ax.axis("off")
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")
    for i, step in enumerate(steps[:n]):
        y    = n - i
        rect = mpatches.FancyBboxPatch(
            (0.5, y - 0.35), 9, 0.7,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#FFFFFF", linewidth=0.8
        )
        ax.add_patch(rect)
        ax.text(5, y, step[:55], ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(5, n - i - 1.35), xytext=(5, n - i - 0.65),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    ax.set_title(title[:50], color="white", fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ── Main generator ────────────────────────────────────────────────────────────

def generate_all_diagrams(slides, theme_color: str = "#1F6FEB",
                           tmp_dir: str = None) -> dict[int, dict]:
    """
    For each content slide, generate a diagram using slide.visual_hint.
    Returns: { slide_index: {"mermaid": str, "png": str|None} }
    """
    tmp = Path(tmp_dir or tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    diagrams: dict[int, dict] = {}

    # Cycle through hint types for slides that have hint="none" or repeat,
    # so consecutive slides never share the same diagram type.
    _fallback_cycle = ["flowchart", "mindmap", "timeline", "comparison", "process", "hierarchy"]
    _used_hints: list[str] = []

    for i, slide in enumerate(slides):
        # Skip title slides and slides with no bullets
        if slide.slide_type == "title" or not slide.bullets:
            continue
        # Skip if slide already uses a PDF image
        if getattr(slide, 'image_id', None):
            continue

        hint = (getattr(slide, 'visual_hint', 'none') or 'none').lower().strip()

        # If hint is "none", skip — don't force a diagram on every slide
        if hint == "none":
            continue

        # If this hint was already used by the previous slide, rotate to next available
        if _used_hints and hint == _used_hints[-1]:
            for candidate in _fallback_cycle:
                if candidate != hint:
                    hint = candidate
                    break

        # Extract bullet texts — use full text for uniqueness
        steps = []
        for b in slide.bullets[:6]:
            txt = b.get("text", "") if isinstance(b, dict) else str(b)
            if txt:
                steps.append(txt)

        if len(steps) < 2:
            continue

        title       = slide.title
        key_message = getattr(slide, "key_message", "") or ""
        out_png  = str(tmp / f"slide_{i}.png")
        mermaid_src: str | None = None
        png_path: str | None = None

        try:
            if hint == "flowchart":
                mermaid_src = _mermaid_flowchart(title, steps)
                _flow_png(title, steps, theme_color, out_png)
                png_path = out_png

            elif hint == "mindmap":
                mermaid_src = _mermaid_mindmap(title, steps, key_message)

            elif hint == "timeline":
                mermaid_src = _mermaid_timeline(title, steps)

            elif hint == "comparison":
                mermaid_src = _mermaid_comparison(title, steps)
                _flow_png(title, steps, theme_color, out_png)
                png_path = out_png

            elif hint == "process":
                mermaid_src = _mermaid_process(title, steps)
                _flow_png(title, steps, theme_color, out_png)
                png_path = out_png

            elif hint == "hierarchy":
                mermaid_src = _mermaid_hierarchy(title, steps, key_message)

            else:
                mermaid_src = _mermaid_flowchart(title, steps)

        except Exception as e:
            log.warning(f"Diagram generation failed for slide {i} (hint={hint}): {e}")
            mermaid_src = None

        if mermaid_src:
            diagrams[i] = {"mermaid": mermaid_src, "png": png_path}
            _used_hints.append(hint)
            log.info(f"  ✔ Slide {i}: {hint} diagram generated.")

    return diagrams
