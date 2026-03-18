"""
diagram_generator.py — Step 11 Support: AI-Assisted Diagram Generation
======================================================================
For each content slide, asks the LLM what kind of diagram would help,
then generates a styled Matplotlib chart/diagram and saves it as a PNG.
Returns a dict mapping slide index -> image path.
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


def _style_fig(fig, ax, bg: str = "#0D1B2A", text: str = "#FFFFFF"):
    """Apply consistent dark‐theme styling to a Matplotlib figure."""
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors=text, labelsize=9)
    ax.xaxis.label.set_color(text)
    ax.yaxis.label.set_color(text)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A3A50")


def _bar_chart(title: str, labels: list[str], values: list[float], color: str, out_path: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(labels, values, color=color, edgecolor="#0D1B2A", linewidth=0.8)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    _style_fig(fig, ax)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                f"{val:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _flow_diagram(title: str, steps: list[str], color: str, out_path: str) -> str:
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


def _decide_diagram(slide_title: str, bullets: list[str], llm: LLMEngine) -> dict | None:
    """
    Ask the LLM whether a diagram is useful for this slide and what kind.
    Returns a dict or None if no diagram is needed.
    """
    bullets_text = "\n".join(f"- {b}" for b in bullets)
    prompt = (
        "You are a data visualization expert. Given a slide's content, "
        "decide if a simple diagram would help (yes or no), and if yes, what type.\n\n"
        f"Slide Title: {slide_title}\nBullets:\n{bullets_text}\n\n"
        "Respond ONLY with JSON, no other text:\n"
        '{"diagram": true/false, "type": "bar"|"flow"|"none", '
        '"title": "short chart title", '
        '"labels": ["A","B","C"], "values": [10, 20, 30]}\n'
        "For flow: use 'steps' key instead of labels/values: [\"Step 1\", \"Step 2\", ...]"
    )
    raw = llm.generate("", [], prompt_override=prompt)
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        info = json.loads(match.group())
        if not info.get("diagram"):
            return None
        return info
    except json.JSONDecodeError:
        return None


def generate_all_diagrams(slides, theme_color: str = "#1F6FEB",
                           tmp_dir: str = None) -> dict[int, str]:
    """
    For each content slide, optionally generate a diagram PNG.
    Returns a mapping { slide_index: png_path }.
    """
    llm = LLMEngine()
    tmp = Path(tmp_dir or tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    diagrams: dict[int, str] = {}

    for i, slide in enumerate(slides):
        if slide.slide_type != "content" or not slide.bullets:
            continue

        log.info(f"Deciding diagram for slide {i}: '{slide.title}'...")
        info = _decide_diagram(slide.title, slide.bullets, llm)
        if not info:
            continue

        out_path = str(tmp / f"slide_{i}.png")

        if info.get("type") == "bar":
            labels = info.get("labels", ["A", "B", "C"])
            values = info.get("values", [10, 20, 30])
            _bar_chart(info.get("title", slide.title), labels, values, theme_color, out_path)
            diagrams[i] = out_path
            log.info(f"  ✔ Bar chart generated for slide {i}.")

        elif info.get("type") == "flow":
            steps = info.get("steps", slide.bullets[:4])
            _flow_diagram(info.get("title", slide.title), steps, theme_color, out_path)
            diagrams[i] = out_path
            log.info(f"  ✔ Flow diagram generated for slide {i}.")

    return diagrams
