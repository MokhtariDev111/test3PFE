"""
api.py — FastAPI Backend for EduGenius AI Teaching Assistant
FIX A: All pipeline steps now use async/await — Ollama call no longer
       blocks the event loop.
FIX D: EasyOCR and embedding model pre-warmed at startup.
"""

import sys, os, logging, shutil, tempfile, uuid, copy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

MAX_FILE_SIZE_MB  = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_PROMPT_LENGTH = 2000

from modules.config_loader     import CONFIG
from modules.ingestion         import ingest_directory
from modules.ocr               import run_ocr, warm_ocr_engine   # FIX D
from modules.text_processing   import process_pages
from modules.embeddings        import build_vector_db, _GLOBAL_EMBED_MODEL
from modules.retrieval         import Retriever
from modules.pedagogical_engine import PedagogicalEngine
from modules.slide_generator   import build_slides
from modules.pptx_exporter     import export_pptx, THEMES
from modules.diagram_generator import generate_all_diagrams
from modules.history_store     import record_presentation, load_history, clear_history

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EduGenius AI Teaching Assistant",
    version="1.0.0",
    description="RAG-powered presentation generation API",
)

_default_origins = ",".join([
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8080", "http://127.0.0.1:8080",
    "http://localhost:8081", "http://127.0.0.1:8081",
])
_allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for generated PPTX assets.
# We keep a temp directory per generation and delete it after download.
_pptx_store: dict[str, dict] = {}


# ── FIX D: Pre-warm heavy models at startup so first request is fast ─────────
@app.on_event("startup")
async def startup_event():
    log.info("Server starting — pre-warming models...")
    # Pre-warm EasyOCR
    warm_ocr_engine()
    # Pre-warm embedding model
    from modules.embeddings import VectorDB
    VectorDB()._load_model()
    log.info("  ✔ All models pre-warmed. Server ready.")


# ── Helper ────────────────────────────────────────────────────────────────────
def _slides_to_json(slides, diagrams: dict) -> list[dict]:
    result = []
    for i, s in enumerate(slides):
        diag_info  = diagrams.get(i, {})
        mermaid_src = diag_info.get("mermaid") if isinstance(diag_info, dict) else None
        result.append({
            "id":            i + 1,
            "title":         s.title,
            "bullets":       s.bullets,
            "speakerNotes":  s.speaker_notes or "",
            "diagram":       mermaid_src,
            "slideType":     s.slide_type,
            "qualityScore":  getattr(s, "quality_score", 0),
            "qualityFeedback": getattr(s, "quality_feedback", ""),
        })
    return result


# ── POST /generate ────────────────────────────────────────────────────────────
@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    prompt:     str  = Form(...),
    theme:      str  = Form("Dark Navy"),
    num_slides: int  = Form(5),
    model:      str  = Form("mistral"),
    top_k:      int  = Form(4),
    language:   str  = Form("English"),
    files: list[UploadFile] = File(default=[]),
):
    if not prompt or len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(422, f"Prompt must be 1-{MAX_PROMPT_LENGTH} chars")
    if not (1 <= num_slides <= 15):
        raise HTTPException(422, "num_slides must be 1–15")
    if theme not in THEMES:
        theme = "Dark Navy"
    language = language if language in ("English", "French") else "English"

    tmp_dir = Path(tempfile.mkdtemp(prefix="edugenius_"))
    raw_dir = tmp_dir / "raw"
    raw_dir.mkdir()
    keep_tmp = False

    try:
        # 1. Save uploaded files
        if files:
            for f in files:
                content = await f.read()
                if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(413, f"'{f.filename}' exceeds {MAX_FILE_SIZE_MB} MB")
                safe_name = Path(f.filename).name.replace("..", "").replace("/", "").replace("\\", "")
                (raw_dir / safe_name).write_bytes(content)
        else:
            src = ROOT_DIR / CONFIG["paths"]["data_raw"]
            if src.exists():
                for p in src.iterdir():
                    shutil.copy(p, raw_dir / p.name)

        selected_theme = THEMES.get(theme, list(THEMES.values())[0])

        # 2–5. Ingest → OCR → Chunk → Embed (sync steps, all fast)
        log.info("Ingesting documents...")
        pages = ingest_directory(raw_dir)

        # Only run OCR on standalone image files — skip pdf_image pages (already have text or are diagrams)
        image_pages = [p for p in pages if p.type == "image" and p.image is not None]
        if image_pages:
            pages = run_ocr(pages)
        else:
            log.info("No standalone images — skipping OCR.")

        # Collect PDF images for slide embedding (separate from text pipeline)
        pdf_images          = {p.source: p.image for p in pages if p.type == "pdf_image" and isinstance(p.image, str)}
        available_image_ids = list(pdf_images.keys())

        # Only chunk text-bearing pages
        text_pages = [p for p in pages if p.type in ("pdf", "txt", "image") and p.text]
        chunks = process_pages(text_pages)

        isolated_index = tmp_dir / "faiss"
        build_vector_db(chunks, index_path=isolated_index)

        retriever = Retriever(index_path=isolated_index)
        results   = retriever.search(prompt, top_k=top_k)

        # 6. LLM — FIX A: fully async, never blocks event loop
        engine = PedagogicalEngine(retriever=retriever)
        lesson = await engine.generate_lesson_async(
            prompt, results,
            num_slides=num_slides,
            language=language,
            available_images=available_image_ids
        )
        topic  = lesson.get("topic", prompt[:40])
        slides = build_slides(lesson)

        # 7. Diagrams
        diag_map: dict = {}
        try:
            accent_hex = "#{:02X}{:02X}{:02X}".format(*selected_theme.accent)
            diag_map   = generate_all_diagrams(slides, theme_color=accent_hex,
                                               tmp_dir=str(tmp_dir / "diagrams"))
        except Exception as e:
            log.warning("Diagram generation failed (non-fatal): %s", e)

        # 8. Export PPTX
        png_map   = {k: v["png"] for k, v in diag_map.items() if v.get("png")}
        pptx_out_dir = tmp_dir / "pptx"
        pptx_path = export_pptx(
            slides,
            theme=selected_theme,
            diagrams=png_map,
            pdf_images=pdf_images,
            output_dir=str(pptx_out_dir),
        )
        log.info("PPTX saved: %s", pptx_path)

        pptx_id = str(uuid.uuid4())
        _pptx_store[pptx_id] = {"path": str(pptx_path), "tmp_dir": str(tmp_dir)}
        keep_tmp = True

        json_slides = _slides_to_json(slides, diag_map)
        record_presentation(pptx_path, prompt, topic, len(slides), theme, model, slides=json_slides)

        return {
            "slides":     json_slides,
            "pptx_id":    pptx_id,
            "topic":      topic,
            "filename":   Path(pptx_path).name,
            "num_slides": len(slides),
        }

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Pipeline error: %s", e)
        raise HTTPException(500, str(e))
    finally:
        if not keep_tmp:
            background_tasks.add_task(shutil.rmtree, tmp_dir, True)


# ── GET /download/{pptx_id} ───────────────────────────────────────────────────
@app.get("/download/{pptx_id}")
async def download(pptx_id: str, background_tasks: BackgroundTasks):
    info = _pptx_store.get(pptx_id)
    path = info.get("path") if isinstance(info, dict) else None
    tmp_dir = info.get("tmp_dir") if isinstance(info, dict) else None
    if not path or not Path(path).exists():
        raise HTTPException(404, "File not found or expired")
    background_tasks.add_task(_pptx_store.pop, pptx_id, None)
    if tmp_dir:
        background_tasks.add_task(shutil.rmtree, tmp_dir, True)
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=Path(path).name,
    )


# ── GET /history ──────────────────────────────────────────────────────────────
@app.get("/history")
async def get_history():
    return {"history": load_history()}


# ── DELETE /history ───────────────────────────────────────────────────────────
@app.delete("/history")
async def delete_history():
    clear_history()
    return {"message": "History cleared"}


# ── POST /transcribe ──────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        import io, speech_recognition as sr
        recognizer = sr.Recognizer()
        contents   = await audio.read()
        with sr.AudioFile(io.BytesIO(contents)) as source:
            audio_data = recognizer.record(source)
        return {"text": recognizer.recognize_google(audio_data)}
    except Exception as e:
        log.error("Transcription error: %s", e)
        raise HTTPException(500, f"Transcription failed: {e}")


# ── GET /themes ───────────────────────────────────────────────────────────────
@app.get("/themes")
async def get_themes():
    return {"themes": list(THEMES.keys())}


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "project": CONFIG["project"]["name"]}
