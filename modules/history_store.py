"""
history_store.py — Persistent presentation history
====================================================
Saves and loads metadata for every generated presentation.
Stored as a JSON array in outputs/presentation_history.json.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.config_loader import CONFIG

log = logging.getLogger("history_store")

HISTORY_FILE = ROOT_DIR / CONFIG["paths"]["outputs"] / "presentation_history.json"


def _load() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save(records: list[dict]):
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def record_presentation(
    pptx_path: str,
    prompt: str,
    topic: str,
    num_slides: int,
    theme_name: str,
    model: str,
):
    """Append a presentation record to the history file."""
    p = Path(pptx_path)
    entry = {
        "id":         datetime.now().strftime("%Y%m%d_%H%M%S"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "prompt":     prompt,
        "topic":      topic,
        "num_slides": num_slides,
        "theme":      theme_name,
        "model":      model,
        "filename":   p.name,
        "filepath":   str(p),
        "size_kb":    round(p.stat().st_size / 1024, 1) if p.exists() else 0,
    }
    records = _load()
    records.insert(0, entry)   # newest first
    _save(records)
    log.info(f"History updated: {entry['filename']}")
    return entry


def load_history() -> list[dict]:
    """Return all past presentation records, newest first."""
    return _load()


def clear_history():
    """Delete all presentation history and the underlying JSON file."""
    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()
    log.info("History cleared.")
