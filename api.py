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
import os
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

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_PROMPT_LENGTH = 2000


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

# CORS — reads ALLOWED_ORIGINS env var.
# Include common local dev origins so the browser doesn't block fetch() calls.
# Example in production: ALLOWED_ORIGINS=https://your-app.netlify.app
_default_allowed_origins = ",".join(
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ]
)
_allowed_origins = os.getenv("ALLOWED_ORIGINS", _default_allowed_origins).split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
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
        # diagrams[i] is now {"mermaid": str, "png": str|None}
        diag_info = diagrams.get(i, {})
        mermaid_src = diag_info.get("mermaid") if isinstance(diag_info, dict) else None
        result.append({
            "id": i + 1,
            "title": s.title,
            "bullets": s.bullets,
            "speakerNotes": s.speaker_notes or "",
            "diagram": mermaid_src,  # Mermaid syntax string or None
            "slideType": s.slide_type,
            "qualityScore": getattr(s, "quality_score", 0),
            "qualityFeedback": getattr(s, "quality_feedback", ""),
        })
    return result


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------
@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    theme: str = Form("Dark Navy"),
    num_slides: int = Form(5),
    model: str = Form("mistral"),
    top_k: int = Form(4),
    language: str = Form("English"),
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
    # 0. Input validation
    if not prompt or len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(status_code=422, detail=f"Prompt must be 1-{MAX_PROMPT_LENGTH} characters")
    if not (1 <= num_slides <= 15):
        raise HTTPException(status_code=422, detail="num_slides must be between 1 and 15")
    if theme not in THEMES:
        theme = "Dark Navy"
    language = language if language in ("English", "French") else "English"

    tmp_dir = Path(tempfile.mkdtemp())
    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir()

    try:
        # 1. Save uploaded files to temp dir (with size and filename safety check)
        if files:
            for f in files:
                content = await f.read()
                if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{f.filename}' exceeds {MAX_FILE_SIZE_MB} MB limit"
                    )
                # Sanitize filename to prevent path traversal
                safe_name = Path(f.filename).name.replace("..", "").replace("/", "").replace("\\", "")
                dest = raw_dir / safe_name
                dest.write_bytes(content)
        else:
            # Fall back to the project's default data directory
            src = ROOT_DIR / CONFIG["paths"]["data_raw"]
            if src.exists():
                for p in src.iterdir():
                    shutil.copy(p, raw_dir / p.name)

        # 2. Use model locally without mutating the global CONFIG singleton
        import copy
        local_config_llm = copy.copy(CONFIG["llm"])
        local_config_llm["model"] = model

        # 3. Run pipeline
        selected_theme = THEMES.get(theme, list(THEMES.values())[0])

        log.info("Ingesting documents …")
        pages = ingest_directory(raw_dir)
        
        # Extract raw embedded PDF images to supply to LLM context and PPTX exporter
        pdf_images = {p.source: p.image for p in pages if p.type == "pdf_image" and isinstance(p.image, str)}
        available_image_ids = list(pdf_images.keys())
        
        pages = run_ocr(pages)
        chunks = process_pages(pages)
        
        # Isolate FAISS index for this specific request to avoid global state race conditions
        isolated_index = tmp_dir / "faiss"
        build_vector_db(chunks, index_path=isolated_index)

        retriever = Retriever(index_path=isolated_index)
        results = retriever.search(prompt, top_k=top_k)

        engine = PedagogicalEngine()
        lesson = engine.generate_lesson(prompt, results, num_slides=num_slides, language=language, available_images=available_image_ids)
        topic = lesson.get("topic", prompt[:40])

        slides = build_slides(lesson)

        # Diagrams — new format: {slide_idx: {"mermaid": str, "png": str|None}}
        diag_map: dict = {}
        try:
            accent_hex = "#{:02X}{:02X}{:02X}".format(*selected_theme.accent)
            diag_tmp = tmp_dir / "diagrams"
            diag_map = generate_all_diagrams(slides, theme_color=accent_hex, tmp_dir=str(diag_tmp))
        except Exception as e:
            log.warning("Diagram generation failed (non-fatal): %s", e)

        # Build PNG-only map for PPTX, Mermaid map for frontend
        png_map = {k: v["png"] for k, v in diag_map.items() if v.get("png")}
        pptx_path = export_pptx(slides, theme=selected_theme, diagrams=png_map, pdf_images=pdf_images)
        log.info("PPTX saved to %s", pptx_path)

        # 4. Store path for download
        pptx_id = str(uuid.uuid4())
        _pptx_store[pptx_id] = str(pptx_path)

        # 5. Record in history
        json_slides = _slides_to_json(slides, diag_map)
        record_presentation(pptx_path, prompt, topic, len(slides), theme, model, slides=json_slides)

        return {
            "slides": json_slides,
            "pptx_id": pptx_id,
            "topic": topic,
            "filename": Path(pptx_path).name,
            "num_slides": len(slides),
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Schedule temp dir cleanup (excluding the outputs dir where PPTX lives)
        # The PPTX is saved to outputs/ (not tmp_dir), so it's safe to clean tmp
        background_tasks.add_task(shutil.rmtree, tmp_dir, True)


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
