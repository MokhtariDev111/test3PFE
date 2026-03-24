"""
api.py — Optimized AI Presentation Engine
==========================================
OPTIMIZATIONS:
1. Batch generation: ALL slides in ONE LLM call (5x faster)
2. Groq support: 10-20x faster than local Ollama
3. Simplified image IDs for reliable LLM matching
4. Fallback image assignment
5. Proper diagram injection
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
import re
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from modules.config_loader import CONFIG
from modules.ingestion import ingest_directory
from modules.ocr import run_ocr, warm_ocr_engine
from modules.text_processing import process_pages
from modules.embeddings import build_vector_db
from modules.retrieval import Retriever, _load_reranker
from modules.slide_generator import SlideData
from modules.diagram_generator import generate_all_diagrams
from modules.history_store import record_presentation, load_history, clear_history
from modules.html_renderer import render as render_html
from modules.context_manager import prepare_context_for_slides
from modules.llm_cache import clear_cache, cache_stats
from modules.evaluation import RAGEvaluator
from modules.health import full_health_check, quick_status

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="EduGenius AI - Optimized",
    version="4.0.0",
    description="Fast RAG-powered presentations with Groq + batch generation",
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Caches ───────────────────────────────────────────────────────────────────
_index_cache: dict[str, dict] = {}
_session_store: dict[str, dict] = {}


def _files_hash(raw_dir: Path) -> str:
    h = hashlib.sha1()
    for p in sorted(raw_dir.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _build_image_registry(pages: list, chunks: list) -> tuple[dict, list, dict]:
    """Create simple image IDs (IMG_001) with context descriptions."""
    pdf_images = {}
    image_contexts = {}
    
    pdf_img_pages = [p for p in pages if p.type == "pdf_image" and isinstance(p.image, str)]
    
    for idx, page in enumerate(pdf_img_pages):
        simple_id = f"IMG_{idx + 1:03d}"
        pdf_images[simple_id] = page.image
        
        match = re.match(r"^(.+?)\s*\(.*?page\s*(\d+).*\)$", page.source, re.IGNORECASE)
        if match:
            base_name, pg_num = match.group(1).strip(), int(match.group(2))
            context_chunks = [c.text[:200] for c in chunks if c.source == base_name and c.page == pg_num]
            image_contexts[simple_id] = " ".join(context_chunks)[:400] if context_chunks else f"Image from {base_name}, page {pg_num}"
        else:
            image_contexts[simple_id] = f"Image: {page.source[:100]}"
    
    return pdf_images, list(pdf_images.keys()), image_contexts


def _assign_fallback_images(slides: list, pdf_images: dict, image_contexts: dict) -> dict:
    """Auto-assign images to slides without image_id based on content matching."""
    if not pdf_images:
        return {}
    
    used_images = set()
    render_images = {}
    
    for i, slide in enumerate(slides):
        iid = slide.get("image_id") if isinstance(slide, dict) else getattr(slide, "image_id", None)
        
        if iid and iid in pdf_images:
            render_images[i] = f"data:image/jpeg;base64,{pdf_images[iid]}"
            used_images.add(iid)
            continue
        
        # Skip title slides
        slide_type = slide.get("slide_type") if isinstance(slide, dict) else getattr(slide, "slide_type", "")
        if slide_type == "title" or i == 0:
            continue
        
        # Build slide text
        title = slide.get("title", "") if isinstance(slide, dict) else getattr(slide, "title", "")
        bullets = slide.get("bullets", []) if isinstance(slide, dict) else getattr(slide, "bullets", [])
        bullet_texts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in bullets]
        slide_text = f"{title} {' '.join(bullet_texts)}".lower()
        
        # Find best matching unused image
        best_match, best_score = None, 0
        for img_id, context in image_contexts.items():
            if img_id in used_images:
                continue
            overlap = len(set(slide_text.split()) & set(context.lower().split()))
            if overlap > best_score and overlap >= 3:
                best_score, best_match = overlap, img_id
        
        if best_match:
            render_images[i] = f"data:image/jpeg;base64,{pdf_images[best_match]}"
            used_images.add(best_match)
    
    return render_images


# ── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    log.info("Pre-warming engines...")
    warm_ocr_engine()
    from modules.embeddings import VectorDB
    VectorDB()._load_model()
    if CONFIG.get("retrieval", {}).get("use_reranker", False):
        _load_reranker(CONFIG["retrieval"]["reranker_model"])
    log.info("✔ System Ready")


@app.get("/")
async def root():
    from modules.llm import LLMEngine
    engine = LLMEngine()
    return {"status": "ok", "version": "4.0-optimized", "backend": engine.backend}


@app.get("/themes")
async def get_themes():
    from modules.html_renderer import _PPTX_TO_HTML_THEME
    return list(_PPTX_TO_HTML_THEME.keys())


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    return {"text": ""}


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED GENERATION ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/generate-stream")
async def generate_stream(
    prompt: str = Form(...),
    theme: str = Form("Dark Navy"),
    num_slides: int = Form(5),
    model: str = Form(""),  # Empty = use default from config/env
    top_k: int = Form(4),
    language: str = Form("English"),
    files: list[UploadFile] = File(default=[]),
    use_pdf_images: bool = Form(True),
    use_batch_mode: bool = Form(True),  # NEW: Enable batch generation
):
    async def _stream():
        session_id = str(uuid.uuid4())
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"edu_{session_id[:8]}_"))
        raw_dir = tmp_dir / "raw"
        raw_dir.mkdir()

        def _emit(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            import time
            start_time = time.time()
            
            print(f"\n{'='*60}", flush=True)
            print(f"🚀 OPTIMIZED GENERATION: {num_slides} slides", flush=True)
            print(f"📝 Topic: '{prompt[:50]}...'", flush=True)
            print(f"⚡ Batch mode: {use_batch_mode}", flush=True)
            print(f"{'='*60}\n", flush=True)

            yield _emit("status", {"step": "ingesting", "message": "Analyzing documents…"})

            # 1. Save Files
            if files:
                for f in files:
                    content = await f.read()
                    (raw_dir / Path(f.filename).name.replace("..", "")).write_bytes(content)
            else:
                src = ROOT_DIR / CONFIG["paths"]["data_raw"]
                if src.exists():
                    for p in src.iterdir():
                        shutil.copy(p, raw_dir / p.name)

            # 2. Index
            file_hash = _files_hash(raw_dir)
            cached = _index_cache.get(file_hash)

            if cached:
                chunks = cached["chunks"]
                isolated_index = Path(cached["index_path"])
                pdf_images = cached.get("pdf_images", {})
                available_image_ids = cached.get("available_image_ids", [])
                image_contexts = cached.get("image_contexts", {})
                yield _emit("status", {"step": "indexing", "message": "Using cached index…"})
            else:
                yield _emit("status", {"step": "indexing", "message": "Building knowledge base…"})
                
                def _process_sync():
                    pages = ingest_directory(raw_dir)
                    pages = run_ocr(pages)
                    return process_pages(pages), pages
                
                chunks, pages = await asyncio.to_thread(_process_sync)
                
                pdf_images, available_image_ids, image_contexts = {}, [], {}
                if use_pdf_images:
                    pdf_images, available_image_ids, image_contexts = _build_image_registry(pages, chunks)
                
                isolated_index = tmp_dir / "faiss"
                build_vector_db(chunks, index_path=isolated_index)
                
                _index_cache[file_hash] = {
                    "index_path": str(isolated_index),
                    "chunks": chunks,
                    "pdf_images": pdf_images,
                    "available_image_ids": available_image_ids,
                    "image_contexts": image_contexts,
                }
                yield _emit("status", {"step": "indexing", "message": "Index built successfully…"})

            # 3. Retrieve
            yield _emit("status", {"step": "retrieving", "message": "Retrieving context…"})
            retriever = Retriever(index_path=isolated_index)
            # Use expanded search for better retrieval coverage
            results = retriever.search_expanded(prompt, top_k=top_k)
            
            # Build context text
            # Smart context preparation with sentence-boundary truncation
            context_text = prepare_context_for_slides(
                chunks=results,
                num_slides=num_slides,
            )

            # 4. Generate Slides
            yield _emit("status", {"step": "generating", "message": "AI is generating slides…"})
            
            from modules.llm import LLMEngine
            llm = LLMEngine()
            lang_label = "French" if language.lower() in ("fr", "french") else "English"

            slides_raw = []
            
            if use_batch_mode:
                # ══════════════════════════════════════════════════════════════
                # BATCH MODE: Single LLM call for all slides (5x faster)
                # ══════════════════════════════════════════════════════════════
                print("⚡ Using BATCH generation mode", flush=True)
                
                slides_raw = await llm.generate_all_slides_batch(
                    query=prompt,
                    context_text=context_text,
                    num_slides=num_slides,
                    language=lang_label,
                    available_images=available_image_ids,
                    image_contexts=image_contexts,
                )
                
                # Stream each slide for UI feedback
                for i, slide in enumerate(slides_raw):
                    yield _emit("slide", {
                        "index": i,
                        "title": slide.get("title", ""),
                        "bullets": slide.get("bullets", []),
                        "slideType": slide.get("slide_type", "content"),
                        "visualHint": slide.get("visual_hint", "none"),
                        "image_id": slide.get("image_id"),
                    })
                    print(f"✅ Slide {i+1}: '{slide.get('title', '')}'", flush=True)
            
            else:
                # ══════════════════════════════════════════════════════════════
                # SEQUENTIAL MODE: One LLM call per slide (original behavior)
                # ══════════════════════════════════════════════════════════════
                print("🐢 Using SEQUENTIAL generation mode", flush=True)
                
                from modules.pedagogical_engine import (
                    SLIDE_ARC, _subquery_for_slide, _build_slide_prompt,
                    _extract_slide_json, _prepare_context, _slide_fingerprint,
                )
                
                prior_titles, prior_hints, prior_fps = [], [], set()
                
                for i in range(num_slides):
                    stype = SLIDE_ARC[i % len(SLIDE_ARC)]
                    sub_q = _subquery_for_slide(prompt, stype, prior_titles)
                    slide_chunks = retriever.search(sub_q, top_k=5) or results
                    ctx_text = _prepare_context(slide_chunks)
                    
                    slide = None
                    for attempt in range(2):
                        slide_prompt = _build_slide_prompt(
                            prompt, ctx_text, stype, i+1, num_slides, lang_label,
                            prior_titles, prior_hints,
                            available_images=available_image_ids,
                            image_contexts=image_contexts
                        )
                        raw = await llm.generate_async(prompt, slide_chunks, prompt_override=slide_prompt)
                        parsed = _extract_slide_json(raw)
                        if parsed and _slide_fingerprint(parsed) not in prior_fps:
                            slide = parsed
                            break
                    
                    if slide:
                        slides_raw.append(slide)
                        prior_titles.append(slide.get("title", ""))
                        prior_hints.append(slide.get("visual_hint", "none"))
                        prior_fps.add(_slide_fingerprint(slide))
                        
                        yield _emit("slide", {
                            "index": i,
                            "title": slide.get("title", ""),
                            "bullets": slide.get("bullets", []),
                            "slideType": slide.get("slide_type", "content"),
                        })
                        print(f"✅ Slide {i+1}: '{slide.get('title', '')}'", flush=True)

            # 5. Convert to SlideData objects
            slides_obj = []
            for slide in slides_raw:
                s_obj = SlideData(
                    title=slide.get("title", ""),
                    bullets=slide.get("bullets", []),
                    speaker_notes=slide.get("speaker_notes", ""),
                    slide_type=slide.get("slide_type", "content"),
                    visual_hint=slide.get("visual_hint", "none"),
                    image_id=slide.get("image_id"),
                )
                slides_obj.append(s_obj)

            # 6. Generate Diagrams
            yield _emit("status", {"step": "diagrams", "message": "Generating diagrams…"})
            diag_map = {}
            try:
                diag_map = await asyncio.to_thread(
                    generate_all_diagrams, slides_obj, "#1F6FEB", str(tmp_dir / "diagrams")
                )
                print(f"📊 Generated {len(diag_map)} diagrams", flush=True)
            except Exception as e:
                log.warning(f"Diagram error: {e}")

            # 7. Build HTML slides with diagrams
            html_slides = []
            for idx, s in enumerate(slides_obj):
                html_bullets = [
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in s.bullets
                ]
                diag_info = diag_map.get(idx, {})
                
                html_slides.append({
                    "title": s.title,
                    "bullets": html_bullets,
                    "speaker_notes": s.speaker_notes,
                    "slide_type": s.slide_type,
                    "image_id": s.image_id,
                    "diagram": diag_info.get("mermaid") if isinstance(diag_info, dict) else None,
                    "visual_hint": s.visual_hint,
                })

            # 8. Assign images
            render_images = _assign_fallback_images(html_slides, pdf_images, image_contexts)
            print(f"🖼️ Assigned {len(render_images)} images", flush=True)

            # 9. Render HTML
            yield _emit("status", {"step": "rendering", "message": "Rendering presentation…"})
            html_path = render_html(
                topic=prompt[:50],
                slides=html_slides,
                session_id=session_id,
                output_dir=str(tmp_dir / "html"),
                images=render_images,
                theme_name=theme,
            )

            _session_store[session_id] = {"html_path": html_path, "tmp_dir": str(tmp_dir)}
            record_presentation(html_path, prompt, prompt[:50], len(slides_obj), theme, model or llm.backend, session_id=session_id)

            elapsed = time.time() - start_time
            print(f"\n✨ DONE in {elapsed:.1f}s ({len(slides_obj)} slides, {len(diag_map)} diagrams, {len(render_images)} images)\n", flush=True)

            yield _emit("done", {
                "session_id": session_id,
                "topic": prompt[:50],
                "num_slides": len(slides_obj),
                "num_diagrams": len(diag_map),
                "num_images": len(render_images),
                "elapsed_seconds": round(elapsed, 1),
                "html_url": f"/view/{session_id}",
            })

        except Exception as e:
            log.exception("Pipeline error: %s", e)
            yield _emit("error", {"detail": str(e)})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/view/{session_id}")
async def view_presentation(session_id: str):
    sess = _session_store.get(session_id)
    if not sess or not sess.get("html_path"):
        hist = load_history()
        match = next((h for h in hist if h.get("id") == session_id), None)
        if match and match.get("html_path") and os.path.exists(match["html_path"]):
            return FileResponse(match["html_path"])
        raise HTTPException(404, "Presentation not found")
    return FileResponse(sess["html_path"])


@app.get("/history")
async def get_history():
    data = load_history()
    for item in data:
        if "id" in item and "html_url" not in item:
            item["html_url"] = f"/view/{item['id']}"
    return data


@app.delete("/history")
async def delete_history():
    clear_history()
    return {"status": "cleared"}

@app.get("/cache/stats")
async def get_cache_stats():
    """Get LLM cache statistics."""
    return cache_stats()

@app.post("/evaluate")
async def evaluate_query(
    query: str = Form(...),
    top_k: int = Form(5),
):
    """Evaluate RAG pipeline quality for a query."""
    from modules.retrieval import Retriever
    
    retriever = Retriever()
    evaluator = RAGEvaluator()
    
    chunks = retriever.search_expanded(query, top_k=top_k)
    
    if not chunks:
        return {"error": "No chunks retrieved", "query": query}
    
    metrics = evaluator.evaluate_retrieval(query, chunks)
    
    return {
        "query": query,
        "metrics": metrics.to_dict(),
        "summary": metrics.summary(),
    }

@app.delete("/cache")
async def clear_llm_cache():
    """Clear all cached LLM responses."""
    count = clear_cache()
    return {"status": "cleared", "entries_removed": count}

@app.get("/health")
async def health_check():
    """Comprehensive system health check."""
    return full_health_check()


@app.get("/health/quick")
async def quick_health():
    """Lightweight status for frequent polling."""
    return quick_status()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
