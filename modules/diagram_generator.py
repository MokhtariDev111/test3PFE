"""
diagram_generator.py — Mermaid Diagram Generator
===================================================
For each content slide, asks the LLM what kind of diagram would help,
then generates the corresponding Mermaid syntax (and optionally a PNG via
matplotlib as a fallback for the PPTX export).

The returned dict maps: slide_index → {"mermaid": str, "png": str|None}
so the React frontend can render the live Mermaid diagram, and the PPTX
exporter can embed the PNG if needed.
"""
import json
import logging
import re
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
import numpy as np

from modules.config_loader import CONFIG
from modules.llm import LLMEngine

log = logging.getLogger("diagram_generator")
if not log.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", datefmt="%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Mermaid generators
# ─────────────────────────────────────────────────────────────────────────────

def _mermaid_bar(title: str, labels: list[str], values: list[float]) -> str:
    """Generate Mermaid xychart-beta bar chart syntax."""
    x_labels = " ".join(f'"{l}"' for l in labels[:8])
    y_values  = ", ".join(str(round(v, 1)) for v in values[:8])
    return (
        "xychart-beta\n"
        f'  title "{title}"\n'
        f"  x-axis [{x_labels}]\n"
        f"  bar [{y_values}]"
    )


def _mermaid_flow(title: str, steps: list[str]) -> str:
    """Generate Mermaid flowchart TD syntax."""
    lines = [f"flowchart TD", f'  title["{title}"]']
    node_ids = [chr(65 + i) for i in range(len(steps))]  # A, B, C, …
    for nid, step in zip(node_ids, steps):
        safe = step.replace('"', "'")
        lines.append(f'  {nid}["{safe}"]')
    # Add arrows between consecutive nodes
    for i in range(len(node_ids) - 1):
        lines.append(f"  {node_ids[i]} --> {node_ids[i+1]}")
    return "\n".join(lines)


def _mermaid_mindmap(title: str, items: list[str]) -> str:
    """Generate Mermaid mindmap syntax."""
    lines = ["mindmap", f"  root((\"{title}\"))"]
    for item in items[:6]:
        safe = item.replace('"', "'")[:50]
        lines.append(f'    ("{safe}")')
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib PNG fallback (kept for PPTX embed)
# ─────────────────────────────────────────────────────────────────────────────

def _to_float(x) -> float | None:
    try:
        if x is None: return None
        if isinstance(x, (int, float, np.number)): return float(x)
        if isinstance(x, str):
            s = x.strip().replace(",", "")
            s = re.sub(r"%$", "", s)
            return float(s) if s else None
        return None
    except Exception:
        return None


def _normalize_bar_data(labels, values):
    if not isinstance(labels, list): labels = [str(labels)]
    if not isinstance(values, list): values = [values]
    fvals = [f for v in values if (f := _to_float(v)) is not None]
    slabels = [str(l) for l in labels]
    n = min(len(slabels), len(fvals))
    return slabels[:n], fvals[:n]


def _style_fig(fig, ax, bg: str = "#0D1B2A", text: str = "#FFFFFF"):
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=text, labelsize=9)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3A50")


def _bar_chart_png(title: str, labels: list[str], values: list[float], color: str, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(labels, values, color=color, edgecolor="#0D1B2A", linewidth=0.8)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    _style_fig(fig, ax)
    max_val = max(values) if values else 0.0
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max_val,
                f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _flow_diagram_png(title: str, steps: list[str], color: str, out_path: str) -> str:
    n = len(steps)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, n + 1)
    ax.axis("off")
    fig.patch.set_facecolor("#0D1B2A")
    ax.set_facecolor("#0D1B2A")
    for i, step in enumerate(steps):
        y = n - i
        rect = mpatches.FancyBboxPatch((0.5, y - 0.35), 9, 0.7, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor="#FFFFFF", linewidth=0.8)
        ax.add_patch(rect)
        ax.text(5, y, step, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(5, n - i - 1.35), xytext=(5, n - i - 0.65),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Static Diagram Mapping
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_diagrams(slides, theme_color: str = "#1F6FEB",
                           tmp_dir: str = None) -> dict[int, dict]:
    """
    For each content slide, generate a diagram using its text bullets.
    This entirely skips the LLM sequential loop, running instantly.
    Returns a mapping:
        { slide_index: {"mermaid": "<mermaid syntax>", "png": "<path or None>"} }
    """
    tmp = Path(tmp_dir or tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    diagrams: dict[int, dict] = {}

    for i, slide in enumerate(slides):
        if slide.slide_type != "content" or not slide.bullets:
            continue
            
        # Avoid diagram overlap if the slide already features a raw PDF image
        if slide.image_id:
            continue

        # Extract text elements directly from the slide bullets
        steps = []
        for b in slide.bullets[:5]:
            txt = b.get("text", "") if isinstance(b, dict) else str(b)
            if txt:
                steps.append(txt)
                
        # Need at least 2 points to make a meaningful comparative diagram
        if len(steps) < 2:
            continue
            
        out_png = str(tmp / f"slide_{i}.png")
        diag_title = slide.title

        # Alternate dynamically between Flowcharts and Mindmaps for visual distinction
        diag_type = "flow" if i % 2 == 0 else "mindmap"

        mermaid_src: str | None = None
        png_path:    str | None = None

        if diag_type == "flow":
            mermaid_src = _mermaid_flow(diag_title, steps)
            try:
                _flow_diagram_png(diag_title, steps, theme_color, out_png)
                png_path = out_png
            except Exception as e:
                log.warning(f"PNG fallback failed for slide {i}: {e}")
        else:  # mindmap
            mermaid_src = _mermaid_mindmap(diag_title, steps)
            # matplotlib doesn't natively draw mindmaps cleanly, use the frontend UI mermaid engine
            
        diagrams[i] = {"mermaid": mermaid_src, "png": png_path}
        log.info(f"  ✔ Generated static {diag_type} diagram for slide {i}.")

    return diagrams
