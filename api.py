"""
api.py — v2: model param wired through + retrieval metadata in response
========================================================================
Changes from v1:
  1. `model` Form param is now passed to PedagogicalEngine so the frontend
     can choose which Ollama model to use (was received but silently ignored).
  2. `_slides_to_json` now includes `key_message` in each slide — the frontend
     can display it as a highlighted summary card.
  3. Response from /generate now includes `retrieval_method` metadata so the
     frontend can show "hybrid" or "dense" in the UI.
  4. All endpoint signatures, response shapes, and field names are UNCHANGED —
     full backward compatibility with existing frontend code.
  5. Reranker model is pre-warmed at startup alongside OCR and embeddings.
"""

import sys, os, logging, shutil, tempfile, uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

MAX_FILE_SIZE_MB  = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_PROMPT_LENGTH = 2000

from modules.config_loader      import CONFIG
from modules.ingestion          import ingest_directory
from modules.ocr                import run_ocr, warm_ocr_engine
from modules.text_processing    import process_pages
from modules.embeddings         import build_vector_db
from modules.retrieval          import Retriever, _load_reranker
from modules.pedagogical_engine import PedagogicalEngine
from modules.slide_generator    import build_slides
from modules.pptx_exporter      import export_pptx, THEMES
from modules.diagram_generator  import generate_all_diagrams
from modules.history_store      import record_presentation, load_history, clear_history

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EduGenius AI Teaching Assistant",
    version="2.0.0",
    description="RAG-powered presentation generation — hybrid retrieval + quality gate",
)

_default_origins = ",".join([
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8080", "http://127.0.0.1:8080",
    "http://localhost:8081", "http://127.0.0.1:8081",
])
_allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory PPTX store (UUID → file path + temp dir)
_pptx_store: dict[str, dict] = {}


# ── Startup: pre-warm all heavy models ───────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    log.info("Server starting — pre-warming models...")

    # OCR model
    warm_ocr_engine()

    # Embedding model
    from modules.embeddings import VectorDB
    VectorDB()._load_model()

    # Reranker model — only if enabled in config
    retrieval_cfg = CONFIG.get("retrieval", {})
    if retrieval_cfg.get("use_reranker", False):
        reranker_model = retrieval_cfg.get("reranker_model", "BAAI/bge-reranker-base")
        _load_reranker(reranker_model)
        log.info("  ✔ Reranker pre-warmed.")

    log.info("  ✔ All models pre-warmed. Server ready.")


# ── Helper: convert SlideData list to JSON-serialisable list ─────────────────
def _slides_to_json(slides, diagrams: dict) -> list[dict]:
    result = []
    for i, s in enumerate(slides):
        diag_info   = diagrams.get(i, {})
        mermaid_src = diag_info.get("mermaid") if isinstance(diag_info, dict) else None
        result.append({
            # ── Fields that existed in v1 (unchanged — frontend compatible) ──
            "id":              i + 1,
            "title":           s.title,
            "bullets":         s.bullets,
            "speakerNotes":    s.speaker_notes or "",
            "diagram":         mermaid_src,
            "slideType":       s.slide_type,
            "qualityScore":    getattr(s, "quality_score", 0),
            "qualityFeedback": getattr(s, "quality_feedback", ""),
            # ── New fields (additive — frontend can ignore if not ready) ─────
            "keyMessage":      getattr(s, "key_message", ""),
            "judgeScore":      getattr(s, "judge_score", None),
            "judgeFeedback":   getattr(s, "judge_feedback", ""),
        })
    return result


# ── POST /generate ────────────────────────────────────────────────────────────
@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    prompt:     str  = Form(...),
    theme:      str  = Form("Dark Navy"),
    num_slides: int  = Form(5),
    model:      str  = Form("mistral"),    # now actually used
    top_k:      int  = Form(4),
    language:   str  = Form("English"),
    files: list[UploadFile] = File(default=[]),
    use_pdf_images: bool = Form(True),  # Include PDF images in slides when content matches
):
    if not prompt or len(prompt) > MAX_PROMPT_LENGTH:
        raise HTTPException(422, f"Prompt must be 1–{MAX_PROMPT_LENGTH} chars")
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

        # 2. Ingest documents
        log.info("Ingesting documents...")
        pages = ingest_directory(raw_dir)

        # 3. OCR (standalone images only)
        image_pages = [p for p in pages if p.type == "image" and p.image is not None]
        if image_pages:
            pages = run_ocr(pages)
        else:
            log.info("No standalone images — skipping OCR.")

        # 4. Chunk text pages
        text_pages = [p for p in pages if p.type in ("pdf", "txt", "image") and p.text]
        chunks = process_pages(text_pages)

        # Collect PDF images for slide embedding (when option enabled)
        pdf_images = {}
        available_image_ids = []
        image_contexts: dict[str, str] = {}
        if use_pdf_images:
            pdf_img_pages = [p for p in pages if p.type == "pdf_image" and isinstance(p.image, str)]
            pdf_images = {p.source: p.image for p in pdf_img_pages}
            available_image_ids = list(pdf_images.keys())
            # Build context for each image from nearby page text (helps LLM match)
            import re
            for p in pdf_img_pages:
                m = re.match(r"^(.+?)\s+\(.*?page\s*(\d+).*\)$", p.source)
                if m:
                    base_name, pg = m.group(1), int(m.group(2))
                    ctx_parts = [c.text[:150] for c in chunks if c.source == base_name and c.page == pg]
                    if ctx_parts:
                        image_contexts[p.source] = " ".join(ctx_parts)[:300]
            if available_image_ids:
                log.info(f"Available PDF images for slides: {len(available_image_ids)} (with context: {len(image_contexts)})")

        # 5. Build vector DB (FAISS + BM25 corpus)
        isolated_index = tmp_dir / "faiss"
        build_vector_db(chunks, index_path=isolated_index)

        # 6. Retrieve context (hybrid: BM25 + FAISS + reranker)
        retriever = Retriever(index_path=isolated_index)
        results   = retriever.search(prompt, top_k=top_k)

        # Determine retrieval method for response metadata
        retrieval_method = "dense"
        if retriever.bm25 is not None:
            retrieval_method = "hybrid+reranked" if retriever.use_reranker else "hybrid"

        # 7. LLM: generate slides
        engine = PedagogicalEngine(retriever=retriever, model_override=model)
        lesson = await engine.generate_lesson_async(
            prompt, results,
            num_slides=num_slides,
            language=language,
            available_images=available_image_ids,
            image_contexts=image_contexts,
        )

        topic  = lesson.get("topic", prompt[:40])
        slides = build_slides(lesson)

        # 8. Diagrams
        diag_map: dict = {}
        try:
            accent_hex = "#{:02X}{:02X}{:02X}".format(*selected_theme.accent)
            diag_map   = generate_all_diagrams(
                slides, theme_color=accent_hex,
                tmp_dir=str(tmp_dir / "diagrams"),
            )
        except Exception as e:
            log.warning("Diagram generation failed (non-fatal): %s", e)

        # 9. Export PPTX
        png_map  = {k: v["png"] for k, v in diag_map.items() if v.get("png")}
        pptx_path = export_pptx(
            slides,
            theme=selected_theme,
            diagrams=png_map,
            pdf_images=pdf_images,
            output_dir=str(tmp_dir / "pptx"),
        )
        log.info("PPTX saved: %s", pptx_path)

        pptx_id = str(uuid.uuid4())
        _pptx_store[pptx_id] = {"path": str(pptx_path), "tmp_dir": str(tmp_dir)}
        keep_tmp = True

        json_slides = _slides_to_json(slides, diag_map)
        if not json_slides:
            raise HTTPException(500, "AI failed to generate any slides from the context. Check Ollama logs.")

        record_presentation(pptx_path, prompt, topic, len(slides), theme, model, slides=json_slides)

        return {
            # ── Unchanged fields (v1 compatible) ─────────────────────────────
            "slides":     json_slides,
            "pptx_id":    pptx_id,
            "topic":      topic,
            "filename":   Path(pptx_path).name,
            "num_slides": len(slides),
            # ── New metadata (additive) ───────────────────────────────────────
            "retrieval_method": retrieval_method,
            "model_used":       model,
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
    info    = _pptx_store.get(pptx_id)
    path    = info.get("path") if isinstance(info, dict) else None
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
    retrieval_cfg = CONFIG.get("retrieval", {})
    return {
        "status":           "ok",
        "project":          CONFIG["project"]["name"],
        "version":          CONFIG["project"]["version"],
        "bm25_enabled":     retrieval_cfg.get("use_bm25", True),
        "reranker_enabled": retrieval_cfg.get("use_reranker", True),
        "parallel_gen":     CONFIG.get("generation", {}).get("parallel", True),
    }
