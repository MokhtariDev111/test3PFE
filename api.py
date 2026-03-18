"""
api.py — FastAPI Backend for EduGenius AI Teaching Assistant
=============================================================
This file exposes the full RAG → PPTX pipeline as a REST API so the
React/Vite frontend (deployed on Netlify) can call it.

Endpoints:
  POST /generate     — Run the pipeline, return JSON slides + pptx_id
  GET  /download/{id} — Download the generated PPTX file
  GET  /history      — Return the presentation history JSON
  POST /transcribe   — Receive an audio blob and return transcribed text

Run:
  uvicorn api:app --reload --host 0.0.0.0 --port 8000

CORS is open for dev; tighten the allowed_origins list for production.
"""

import sys
import logging
import shutil
import tempfile
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional

from modules.config_loader import CONFIG
from modules.ingestion import ingest_directory
from modules.ocr import run_ocr
from modules.text_processing import process_pages
from modules.embeddings import build_vector_db
from modules.retrieval import Retriever
from modules.pedagogical_engine import PedagogicalEngine
from modules.slide_generator import build_slides
from modules.pptx_exporter import export_pptx, THEMES
from modules.diagram_generator import generate_all_diagrams
from modules.history_store import record_presentation, load_history, clear_history

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EduGenius AI Teaching Assistant",
    version="1.0.0",
    description="RAG-powered presentation generation API",
)

# CORS — allow all origins in development.
# For production on Netlify, change "*" to your Netlify URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory store for generated PPTX paths (maps pptx_id → absolute path)
# ---------------------------------------------------------------------------
_pptx_store: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Helper: convert our Slide objects → JSON the React frontend understands
# ---------------------------------------------------------------------------
def _slides_to_json(slides, diagrams: dict) -> list[dict]:
    result = []
    for i, s in enumerate(slides):
        diag = None
        if i in diagrams and Path(diagrams[i]).exists():
            # Embed diagram path as a relative URL — frontend can request it
            # or we send the mermaid source if available
            diag = None  # diagrams are image files; for now skip binary
        result.append({
            "id": i + 1,
            "title": s.title,
            "bullets": s.bullets,
            "speakerNotes": s.speaker_notes or "",
            "diagram": diag,
            "slideType": s.slide_type,
        })
    return result


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    theme: str = Form("Dark Navy"),
    num_slides: int = Form(5),
    model: str = Form("mistral"),
    top_k: int = Form(4),
    files: list[UploadFile] = File(default=[]),
):
    """
    Run the full pipeline.
    Returns JSON:
      {
        "slides": [...],
        "pptx_id": "<uuid>",
        "topic": "...",
        "filename": "...",
        "num_slides": N
      }
    """
    tmp_dir = Path(tempfile.mkdtemp())
    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir()

    try:
        # 1. Save uploaded files to temp dir
        if files:
            for f in files:
                dest = raw_dir / f.filename
                with dest.open("wb") as out:
                    shutil.copyfileobj(f.file, out)
        else:
            # Fall back to the project's default data directory
            src = ROOT_DIR / CONFIG["paths"]["data_raw"]
            if src.exists():
                for p in src.iterdir():
                    shutil.copy(p, raw_dir / p.name)

        # 2. Override model in config
        CONFIG["llm"]["model"] = model

        # 3. Run pipeline
        selected_theme = THEMES.get(theme, list(THEMES.values())[0])

        log.info("Ingesting documents …")
        pages = ingest_directory(raw_dir)
        pages = run_ocr(pages)
        chunks = process_pages(pages)
        build_vector_db(chunks)

        retriever = Retriever()
        results = retriever.search(prompt, top_k=top_k)

        engine = PedagogicalEngine()
        lesson = engine.generate_lesson(prompt, results, num_slides=num_slides)
        topic = lesson.get("topic", prompt[:40])

        slides = build_slides(lesson)

        # Diagrams (optional — may fail gracefully)
        diag_map: dict = {}
        try:
            accent_hex = "#{:02X}{:02X}{:02X}".format(*selected_theme.accent)
            diag_tmp = tmp_dir / "diagrams"
            diag_map = generate_all_diagrams(slides, theme_color=accent_hex, tmp_dir=str(diag_tmp))
        except Exception as e:
            log.warning("Diagram generation failed (non-fatal): %s", e)

        pptx_path = export_pptx(slides, theme=selected_theme, diagrams=diag_map)
        log.info("PPTX saved to %s", pptx_path)

        # 4. Store path for download
        pptx_id = str(uuid.uuid4())
        _pptx_store[pptx_id] = str(pptx_path)

        # 5. Record in history
        record_presentation(pptx_path, prompt, topic, len(slides), theme, model)

        return {
            "slides": _slides_to_json(slides, diag_map),
            "pptx_id": pptx_id,
            "topic": topic,
            "filename": Path(pptx_path).name,
            "num_slides": len(slides),
        }

    except Exception as e:
        log.exception("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Don't delete tmp_dir here — we still need the PPTX for download
        pass


# ---------------------------------------------------------------------------
# GET /download/{pptx_id}
# ---------------------------------------------------------------------------
@app.get("/download/{pptx_id}")
async def download(pptx_id: str, background_tasks: BackgroundTasks):
    path = _pptx_store.get(pptx_id)
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found or expired")
    # Optionally remove from store after download
    background_tasks.add_task(_pptx_store.pop, pptx_id, None)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=Path(path).name,
    )


# ---------------------------------------------------------------------------
# GET /history
# ---------------------------------------------------------------------------
@app.get("/history")
async def get_history():
    records = load_history()
    return {"history": records}


# ---------------------------------------------------------------------------
# DELETE /history
# ---------------------------------------------------------------------------
@app.delete("/history")
async def delete_history():
    clear_history()
    return {"message": "History cleared"}


# ---------------------------------------------------------------------------
# POST /transcribe  (voice input — requires SpeechRecognition)
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """
    Accepts a WAV or WebM audio blob from the browser.
    Returns {"text": "transcribed text"}.
    """
    try:
        import io
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        contents = await audio.read()
        audio_file = io.BytesIO(contents)

        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return {"text": text}
    except Exception as e:
        log.error("Transcription error: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


# ---------------------------------------------------------------------------
# GET /themes  — expose available themes to the frontend
# ---------------------------------------------------------------------------
@app.get("/themes")
async def get_themes():
    return {"themes": list(THEMES.keys())}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "project": CONFIG["project"]["name"]}
