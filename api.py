"""
api.py — Unified AI Presentation Engine (HTML-Native)
======================================================
Architectural Unification:
1. RAG-based intelligence (Mistral/Ollama + FAISS + Hybrid Search).
2. Interactive HTML Rendering (Ported from friend's project).
3. PowerPoint dependencies COMPLETELY REMOVED.
4. Professional, fast, and scalable.
"""

import sys
import os
import logging
import shutil
import tempfile
import uuid
import json
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

# ── Project Modules ──────────────────────────────────────────────────────────
from modules.config_loader      import CONFIG
from modules.ingestion          import ingest_directory
from modules.ocr                import run_ocr, warm_ocr_engine
from modules.text_processing    import process_pages
from modules.embeddings         import build_vector_db
from modules.retrieval          import Retriever, _load_reranker
from modules.pedagogical_engine import PedagogicalEngine
from modules.slide_generator    import build_slides
from modules.diagram_generator  import generate_all_diagrams
from modules.history_store      import record_presentation, load_history, clear_history
from modules.html_renderer      import render as render_html

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EduGenius AI Interactive Presentation System",
    version="3.0.0",
    description="Modernized RAG-powered presentation generation with HTML-native rendering.",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
_default_origins = ",".join([
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:8080", "http://127.0.0.1:8080",
    "http://localhost:8081", "http://127.0.0.1:8081",
    "http://localhost:8085", "http://127.0.0.1:8085",
    "http://localhost:8086", "http://127.0.0.1:8086",
    "http://localhost:8087", "http://127.0.0.1:8087",
    "http://localhost:8088", "http://127.0.0.1:8088",
    "http://localhost:8089", "http://127.0.0.1:8089",
    "http://localhost:8090", "http://127.0.0.1:8090",
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

# ── Caches & Stores ──────────────────────────────────────────────────────────
_index_cache: dict[str, dict] = {}
_INDEX_CACHE_MAX = 8
_session_store: dict[str, dict] = {} # UUID → {html_path, topic, slides, ...}

def _files_hash(raw_dir: Path) -> str:
    h = hashlib.sha1()
    for p in sorted(raw_dir.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()

# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    log.info("Server Unification: Pre-warming RAG & OCR engines...")
    warm_ocr_engine()
    from modules.embeddings import VectorDB
    VectorDB()._load_model()
    retrieval_cfg = CONFIG.get("retrieval", {})
    if retrieval_cfg.get("use_reranker", False):
        reranker_model = retrieval_cfg.get("reranker_model", "BAAI/bge-reranker-base")
        _load_reranker(reranker_model)
    log.info("✔ System Ready.")

# ── Helper: Slide conversion ─────────────────────────────────────────────────
def _slides_to_json(slides, diagrams: dict) -> list[dict]:
    result = []
    for i, s in enumerate(slides):
        diag_info   = diagrams.get(i, {})
        mermaid_src = diag_info.get("mermaid") if isinstance(diag_info, dict) else None
        result.append({
            "id":              i + 1,
            "title":           s.title,
            "bullets":         s.bullets,
            "speakerNotes":    s.speaker_notes or "",
            "diagram":         mermaid_src,
            "slideType":       s.slide_type,
            "keyMessage":      getattr(s, "key_message", ""),
            "image_id":        getattr(s, "image_id", None),
        })
    return result

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "engine": "V3-Unified", "output": "HTML-Native"}

@app.get("/themes")
async def get_themes():
    """Return the list of available presentation theme names."""
    from modules.html_renderer import _PPTX_TO_HTML_THEME
    return list(_PPTX_TO_HTML_THEME.keys())

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Audio transcription stub.
    Returns empty text — wire up Whisper or another ASR engine here when ready.
    """
    return {"text": ""}

@app.post("/generate-stream")
async def generate_stream(
    prompt:     str  = Form(...),
    theme:      str  = Form("Dark Navy"),
    num_slides: int  = Form(5),
    model:      str  = Form("mistral"),
    top_k:      int  = Form(4),
    language:   str  = Form("English"),
    files: list[UploadFile] = File(default=[]),
    use_pdf_images: bool = Form(True),
):
    async def _stream():
        session_id = str(uuid.uuid4())
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"edugenius_{session_id[:8]}_"))
        raw_dir = tmp_dir / "raw"
        raw_dir.mkdir()

        def _emit(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            print(f"\n========================================================", flush=True)
            print(f"🚀 NEW POST REQUEST: Generating {num_slides} slides", flush=True)
            print(f"📝 Topic: '{prompt[:50]}...'", flush=True)
            print(f"========================================================\n", flush=True)
            
            yield _emit("status", {"step": "ingesting", "message": "Analyzing documents…"})
            # 1. Save Files
            if files:
                for f in files:
                    content = await f.read()
                    safe_name = Path(f.filename).name.replace("..", "")
                    (raw_dir / safe_name).write_bytes(content)
            else:
                src = ROOT_DIR / CONFIG["paths"]["data_raw"]
                if src.exists():
                    for p in src.iterdir(): shutil.copy(p, raw_dir / p.name)

            # 2. Cache & Ingest
            file_hash = _files_hash(raw_dir)
            cached    = _index_cache.get(file_hash)

            if cached:
                chunks = cached["chunks"]
                isolated_index = Path(cached["index_path"])
                pdf_images = cached.get("pdf_images", {})
                available_image_ids = list(pdf_images.keys())
                image_contexts = cached.get("image_contexts", {})
                yield _emit("status", {"step": "indexing", "message": "Using cached knowledge base…"})
            else:
                def _process_documents_sync():
                    pages = ingest_directory(raw_dir)
                    image_pages = [p for p in pages if p.type == "image" and p.image is not None]
                    if image_pages: pages = run_ocr(pages)
                    
                    text_pages = [p for p in pages if p.type in ("pdf", "txt", "image") and p.text]
                    chunks = process_pages(text_pages)
                    
                    pdf_images = {}
                    image_contexts = {}
                    if use_pdf_images:
                        import re as _re
                        pdf_img_pages = [p for p in pages if p.type == "pdf_image" and isinstance(p.image, str)]
                        pdf_images = {p.source: p.image for p in pdf_img_pages}
                        for p in pdf_img_pages:
                            m = _re.match(r"^(.+?)\s+\(.*?page\s*(\d+).*\)$", p.source)
                            if m:
                                base_name, pg = m.group(1), int(m.group(2))
                                ctx_p = [c.text[:150] for c in chunks if c.source == base_name and c.page == pg]
                                if ctx_p: image_contexts[p.source] = " ".join(ctx_p)[:300]

                    isolated_index = tmp_dir / "faiss"
                    build_vector_db(chunks, index_path=isolated_index)
                    _index_cache[file_hash] = {
                        "index_path": str(isolated_index), "chunks": chunks, 
                        "pdf_images": pdf_images, "image_contexts": image_contexts
                    }
                
                await asyncio.to_thread(_process_documents_sync)

                # 2.5 Re-assign variables from the populated cache
                _cached = _index_cache[file_hash]
                chunks = _cached["chunks"]
                isolated_index = Path(_cached["index_path"])
                pdf_images = _cached.get("pdf_images", {})
                available_image_ids = list(pdf_images.keys())
                image_contexts = _cached.get("image_contexts", {})
                yield _emit("status", {"step": "indexing", "message": "Knowledge base indexed successfully…"})



            # 3. Retrieve
            yield _emit("status", {"step": "retrieving", "message": "Retrieving context…"})
            retriever = Retriever(index_path=isolated_index)
            results   = retriever.search(prompt, top_k=top_k)

            # 4. Generate & Stream
            yield _emit("status", {"step": "generating", "message": "AI is crafting your slides…"})
            from modules.pedagogical_engine import (
                SLIDE_ARC, _subquery_for_slide, _build_slide_prompt,
                _extract_slide_json, _prepare_context, _slide_fingerprint, _norm,
            )
            from modules.llm import LLMEngine
            llm = LLMEngine()
            lang_label = "French" if language.lower() in ("fr", "french") else "English"
            
            prior_titles, prior_hints, prior_fps = [], [], set()
            slides_obj = []

            slides_generated = 0
            attempts = 0
            max_attempts = num_slides * 3

            while slides_generated < num_slides and attempts < max_attempts:
                attempts += 1
                i = slides_generated
                print(f"⏳ Working on slide {i+1} of {num_slides} (attempt {attempts})...", flush=True)

                stype = SLIDE_ARC[i % len(SLIDE_ARC)]
                sub_q = _subquery_for_slide(prompt, stype, prior_titles)
                slide_chunks = retriever.search(sub_q, top_k=5) or results
                ctx_text = _prepare_context(slide_chunks)
                
                slide = None
                for attempt in range(2):
                    slide_prompt = _build_slide_prompt(
                        prompt, ctx_text, stype, i+1, num_slides, lang_label,
                        prior_titles, prior_hints, available_images=available_image_ids,
                        image_contexts=image_contexts
                    )
                    raw = await llm.generate_async(prompt, slide_chunks, prompt_override=slide_prompt, model_override=model)
                    parsed = _extract_slide_json(raw)
                    if not parsed: continue
                    
                    fp = _slide_fingerprint(parsed)
                    if fp in prior_fps: continue
                    
                    slide = parsed
                    break
                
                
                if not slide:
                    print(f"❌ Slide {i+1} attempt failed, retrying...", flush=True)
                    continue

                slides_generated += 1
                print(f"✅ Slide {i+1} generated successfully: '{slide.get('title', '')}'", flush=True)
                
                # Convert dict to SlideData-like object for build_slides compatibility
                from modules.slide_generator import SlideData
                s_obj = SlideData(
                    title=slide.get("title",""), bullets=slide.get("bullets",[]),
                    speaker_notes=slide.get("speaker_notes",""), slide_type=slide.get("slide_type","content")
                )
                for k,v in slide.items(): setattr(s_obj, k, v)
                slides_obj.append(s_obj)
                prior_titles.append(s_obj.title)
                prior_hints.append(slide.get("visual_hint",""))
                prior_fps.add(_slide_fingerprint(slide))
                
                yield _emit("slide", {
                    "index": i, "title": s_obj.title, "bullets": s_obj.bullets,
                    "slideType": s_obj.slide_type, "image_id": getattr(s_obj, "image_id", None)
                })

            # 5. Diagrams & HTML Render
            yield _emit("status", {"step": "finalizing", "message": "Rendering interactive presentation…"})
            diag_map = {}
            try:
                diag_map = await asyncio.to_thread(
                    generate_all_diagrams,
                    slides_obj,
                    "#1F6FEB",
                    str(tmp_dir / "diagrams")
                )
            except Exception as e: 
                log.warning(f"Diagram generation error: {e}")

            html_slides = []
            for s in slides_obj:
                html_bullets = []
                for b in s.bullets:
                    if isinstance(b, dict):
                        html_bullets.append(b.get("text", "") or b.get("content", ""))
                    else:
                        html_bullets.append(str(b))
                        
                html_slides.append({
                    "title": s.title, "bullets": html_bullets, "speaker_notes": s.speaker_notes,
                    "slide_type": s.slide_type, "image_id": getattr(s, "image_id", None),
                    "chart_data": getattr(s, "chart_data", None)
                })
            
            # Prepare images for renderer
            render_images = {}
            for i, s in enumerate(slides_obj):
                iid = getattr(s, "image_id", None)
                if iid and iid in pdf_images:
                    # Injects base64 as a data URL
                    render_images[i] = f"data:image/jpeg;base64,{pdf_images[iid]}"

            html_path = render_html(
                topic=prompt[:50], slides=html_slides, session_id=session_id,
                output_dir=str(tmp_dir / "html"), images=render_images, theme_name=theme
            )
            
            _session_store[session_id] = {"html_path": html_path, "tmp_dir": str(tmp_dir)}
            record_presentation(html_path, prompt, prompt[:50], len(slides_obj), theme, model, session_id=session_id)

            yield _emit("done", {
                "session_id": session_id, "topic": prompt[:50],
                "num_slides": len(slides_obj), "html_url": f"/view/{session_id}"
            })

        except Exception as e:
            log.exception("Stream pipeline error: %s", e)
            yield _emit("error", {"detail": str(e)})

    return StreamingResponse(_stream(), media_type="text/event-stream")

@app.get("/view/{session_id}")
async def view_presentation(session_id: str):
    sess = _session_store.get(session_id)
    if not sess or not sess.get("html_path"):
        # Try checking history file if session expired
        hist = load_history()
        match = next((h for h in hist if h.get("id") == session_id), None)
        if match and match.get("html_path") and os.path.exists(match["html_path"]):
            return FileResponse(match["html_path"])
        raise HTTPException(404, "Presentation not found or expired.")
    return FileResponse(sess["html_path"])

@app.get("/history")
async def get_history():
    data = load_history()
    # Ensure each item has an html_url
    for item in data:
        if "id" in item and "html_url" not in item:
            item["html_url"] = f"/view/{item['id']}"
    return data

@app.delete("/history")
async def delete_history():
    clear_history()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
