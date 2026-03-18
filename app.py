"""
app.py — Step 11: Streamlit Web Interface (3-Tab Edition)
=========================================================
Tab 1 — 🎯 Generate  : Run the full RAG → PPTX pipeline
Tab 2 — 🖼️ Preview   : Visual slide carousel with navigation
Tab 3 — 📚 History   : Dashboard of all past presentations
Run: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import tempfile
import shutil

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

ROOT_DIR = Path(__file__).resolve().parent

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0f1e35 100%);
    color: #e8f0fe;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #0a1525 100%);
    border-right: 1px solid #1f3a5a;
}
section[data-testid="stSidebar"] * { color: #d0dff0 !important; }

.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2d4a 50%, #1f3a5a 100%);
    border: 1px solid #1f6feb; border-radius: 16px;
    padding: 1.8rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content:''; position: absolute; top:0; left:0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, #1f6feb, #7ec8ff);
}
.hero-title {
    font-size: 2.1rem; font-weight: 700;
    background: linear-gradient(90deg, #7ec8ff, #1f6feb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #7a9cc4; font-size: 0.95rem; }

div.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #0a4db5) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 1.5rem !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
    width: 100% !important; transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(31,111,235,0.35) !important;
}
div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(31,111,235,0.5) !important;
}
div[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #27ae60, #1e8449) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    width: 100% !important; padding: 0.7rem 1rem !important;
}
.stTextArea textarea, .stTextInput input {
    background: #0d1b2a !important; color: #e8f0fe !important;
    border: 1px solid #1f3a5a !important; border-radius: 8px !important;
}
.section-header {
    font-size: 1rem; font-weight: 700; color: #7ec8ff;
    border-bottom: 1px solid #1f3a5a;
    padding-bottom: 0.4rem; margin: 1rem 0 0.6rem;
}

/* History card */
.hist-card {
    background: linear-gradient(135deg, #0f1e35, #162840);
    border: 1px solid #1f3a5a; border-radius: 12px;
    padding: 1rem 1.3rem; margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.hist-card:hover { border-color: #1f6feb; }
.hist-meta { color: #5a7a99; font-size: 0.82rem; margin-top: 0.4rem; }
.hist-topic { font-size: 1rem; font-weight: 700; color: #e8f0fe; }
.hist-prompt { color: #7a9cc4; font-size: 0.87rem; font-style: italic; margin-top: 0.2rem; }

/* Slide nav */
.slide-nav-bar {
    display: flex; align-items: center; justify-content: center;
    gap: 1rem; padding: 0.6rem 0; margin-bottom: 0.8rem;
}
.slide-counter {
    background: #1f3a5a; color: #7ec8ff;
    border-radius: 20px; padding: 3px 14px;
    font-size: 0.82rem; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
if "slides"       not in st.session_state: st.session_state.slides       = []
if "pptx_bytes"   not in st.session_state: st.session_state.pptx_bytes   = None
if "pptx_fname"   not in st.session_state: st.session_state.pptx_fname   = None
if "diagrams"     not in st.session_state: st.session_state.diagrams      = {}
if "theme_cfg"    not in st.session_state: st.session_state.theme_cfg     = None
if "slide_idx"    not in st.session_state: st.session_state.slide_idx     = 0
if "last_prompt"  not in st.session_state: st.session_state.last_prompt   = ""
if "last_topic"   not in st.session_state: st.session_state.last_topic    = ""


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 AI Teaching Assistant")
    st.markdown("---")
    st.markdown('<div class="section-header">📁 Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDFs or TXT", type=["pdf","txt"],
                                accept_multiple_files=True)

    st.markdown('<div class="section-header">🎨 Style</div>', unsafe_allow_html=True)
    theme_name  = st.selectbox("Color Theme", list(THEMES.keys()), index=0)
    num_slides  = st.slider("Content Slides", 2, 8, 3)

    st.markdown('<div class="section-header">🤖 AI Settings</div>', unsafe_allow_html=True)
    model_name  = st.selectbox("Ollama Model", ["mistral","llama3.2","qwen2.5"], index=0)
    top_k       = st.slider("Context Chunks (top-k)", 2, 8, 4)
    diagrams_on = st.toggle("Generate Diagrams", value=True)

CONFIG["llm"]["model"] = model_name


# ── Helper: render a single slide as themed HTML ─────────────────────────────
def _slide_html(s, idx: int, theme, diagrams: dict, show_full: bool = True) -> str:
    bg     = "#{:02X}{:02X}{:02X}".format(*theme.bg)
    accent = "#{:02X}{:02X}{:02X}".format(*theme.accent)
    text   = "#{:02X}{:02X}{:02X}".format(*theme.text)
    bullet = "#{:02X}{:02X}{:02X}".format(*theme.bullet)
    sub    = "#{:02X}{:02X}{:02X}".format(*theme.subtext)

    if s.slide_type == "title":
        return f"""
        <div style="background:{bg};border:2px solid {accent};border-radius:12px;
                    padding:0;overflow:hidden;position:relative;min-height:220px;
                    display:flex;flex-direction:column;justify-content:center;">
          <div style="position:absolute;left:0;top:0;width:8px;height:100%;background:{accent};"></div>
          <div style="position:absolute;bottom:0;left:0;width:100%;height:6px;background:{accent};"></div>
          <div style="padding:2rem 2.5rem;">
            <div style="font-size:0.65rem;color:{accent};font-weight:700;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.5rem;">TITLE SLIDE • {idx+1}</div>
            <div style="font-size:1.9rem;font-weight:700;color:{text};margin-bottom:0.4rem;">{s.title}</div>
            <div style="font-size:0.85rem;color:{sub};">AI Teaching Assistant • Powered by RAG</div>
          </div>
        </div>"""
    else:
        has_diag = idx in diagrams
        diag_badge = f'<span style="background:{accent}33;color:{accent};border-radius:4px;padding:1px 8px;font-size:0.62rem;font-weight:700;margin-left:8px;">📊 DIAGRAM</span>' if has_diag else ""
        bullets_html = "".join(
            f'<div style="color:{bullet};font-size:0.9rem;margin:0.35rem 0 0.35rem 0.5rem;line-height:1.4;">▸&nbsp; {b}</div>'
            for b in s.bullets)
        notes = f'<div style="color:{sub};font-size:0.75rem;border-top:1px solid {accent}33;margin-top:0.8rem;padding-top:0.5rem;font-style:italic;">📝 {s.speaker_notes[:160]}...</div>' if show_full and s.speaker_notes else ""
        return f"""
        <div style="background:{bg};border:1px solid {accent}66;border-radius:12px;overflow:hidden;">
          <div style="background:{accent};padding:0.55rem 1rem;display:flex;align-items:center;">
            <span style="color:{text};font-weight:700;font-size:0.95rem;">Slide {idx+1}: {s.title}</span>
            {diag_badge}
          </div>
          <div style="padding:0.9rem 1.2rem;">{bullets_html}{notes}</div>
        </div>"""


# ── 3 Main Tabs ───────────────────────────────────────────────────────────────
tab_gen, tab_prev, tab_hist = st.tabs(["🎯 Generate", "🖼️ Preview", "📚 History"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ════════════════════════════════════════════════════════════════════════════════
with tab_gen:
    st.markdown("""
    <div class="hero-banner">
      <div class="hero-title">🎓 AI Teaching Assistant</div>
      <div class="hero-sub">Automatic PowerPoint generation from documents using Retrieval-Augmented Generation</div>
    </div>
    """, unsafe_allow_html=True)

    query = st.text_area("💬 Topic or question",
        placeholder='"Explain the Transformer attention mechanism"',
        height=90, key="query_input")

    col_btn, col_pad = st.columns([1, 3])
    with col_btn:
        go = st.button("⚡ Generate Slides")

    if go:
        if not query.strip():
            st.error("⚠️ Please enter a topic first.")
            st.stop()

        tmp_dir = Path(tempfile.mkdtemp())
        raw_dir = tmp_dir / "raw"
        raw_dir.mkdir()

        if uploaded:
            for f in uploaded:
                (raw_dir / f.name).write_bytes(f.getvalue())
        else:
            src = ROOT_DIR / CONFIG["paths"]["data_raw"]
            if src.exists():
                for p in src.iterdir():
                    shutil.copy(p, raw_dir / p.name)

        theme = THEMES[theme_name]

        with st.status("🚀 Running AI pipeline...", expanded=True) as status_box:
            st.write("**Step 2** — Ingesting documents...")
            pages = ingest_directory(raw_dir)
            st.write(f"  ✅ {len(pages)} pages from {len(list(raw_dir.iterdir()))} file(s).")

            st.write("**Step 3** — OCR...")
            pages = run_ocr(pages)
            st.write(f"  ✅ {len([p for p in pages if p.type=='image'])} image(s) processed.")

            st.write("**Step 4** — Chunking text...")
            chunks = process_pages(pages)
            st.write(f"  ✅ {len(chunks)} chunks.")

            st.write("**Step 5** — Building FAISS index...")
            build_vector_db(chunks)
            st.write("  ✅ Index saved.")

            st.write(f"**Step 6** — Retrieving top {top_k} chunks...")
            retriever = Retriever()
            results = retriever.search(query, top_k=top_k)
            st.write(f"  ✅ {len(results)} chunks retrieved.")

            st.write(f"**Step 8** — Generating {num_slides}-slide plan ({model_name})...")
            engine = PedagogicalEngine()
            lesson = engine.generate_lesson(query, results, num_slides=num_slides)
            topic = lesson.get("topic", query)
            st.write(f"  ✅ Topic: *{topic}*")

            st.write("**Step 9** — Building slide structure...")
            slides = build_slides(lesson)
            st.write(f"  ✅ {len(slides)} slide(s).")

            diag_map: dict = {}
            if diagrams_on and slides:
                st.write("**Diagrams** — Generating charts...")
                accent_hex = "#{:02X}{:02X}{:02X}".format(*theme.accent)
                diag_tmp = tmp_dir / "diagrams"
                diag_map = generate_all_diagrams(slides, theme_color=accent_hex, tmp_dir=str(diag_tmp))
                st.write(f"  ✅ {len(diag_map)} diagram(s).")

            st.write(f"**Step 10** — Exporting PPTX ({theme_name})...")
            pptx_path = export_pptx(slides, theme=theme, diagrams=diag_map)
            st.write(f"  ✅ `{Path(pptx_path).name}`")

            status_box.update(label="✅ Done!", state="complete", expanded=False)

        # Read bytes before cleanup
        with open(pptx_path, "rb") as fh:
            pptx_bytes = fh.read()
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # Save to history
        record_presentation(pptx_path, query, topic, len(slides), theme_name, model_name)

        # Store in session state for Preview tab
        st.session_state.slides     = slides
        st.session_state.pptx_bytes = pptx_bytes
        st.session_state.pptx_fname = Path(pptx_path).name
        st.session_state.diagrams   = diag_map
        st.session_state.theme_cfg  = theme
        st.session_state.slide_idx  = 0
        st.session_state.last_prompt = query
        st.session_state.last_topic  = topic

        # Quick summary in generate tab
        st.markdown("---")
        st.success(f"✅ **{len(slides)} slide(s)** generated! "
                   f"Switch to the **🖼️ Preview** tab to review before downloading.")

        col_dl, _ = st.columns([1, 2])
        with col_dl:
            st.download_button(
                label=f"📥 Download {Path(pptx_path).name}",
                data=pptx_bytes,
                file_name=Path(pptx_path).name,
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_prev:
    slides   = st.session_state.slides
    theme    = st.session_state.theme_cfg
    diagrams = st.session_state.diagrams

    if not slides or theme is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 0;color:#3a5a7a;">
          <div style="font-size:3.5rem;">🖼️</div>
          <div style="font-size:1.1rem;font-weight:600;color:#5a7a9a;margin-top:1rem;">No presentation yet</div>
          <div style="font-size:0.9rem;margin-top:0.5rem;">Go to <b>🎯 Generate</b> first, then come back here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        n = len(slides)
        idx = st.session_state.slide_idx

        # Topic header
        st.markdown(f"### 🖼️ {st.session_state.last_topic or 'Presentation Preview'}")
        st.caption(f"Prompt: *\"{st.session_state.last_prompt[:120]}\"*")
        st.markdown("---")

        # ── Navigation bar ──
        nav_l, nav_c, nav_r = st.columns([1, 2, 1])
        with nav_l:
            if st.button("◀ Previous", disabled=(idx == 0), key="prev_slide"):
                st.session_state.slide_idx = max(0, idx - 1)
                st.rerun()
        with nav_c:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<span style="background:#1f3a5a;color:#7ec8ff;border-radius:20px;'
                f'padding:4px 18px;font-weight:600;font-size:0.85rem;">'
                f'Slide {idx+1} / {n}</span></div>',
                unsafe_allow_html=True)
        with nav_r:
            if st.button("Next ▶", disabled=(idx == n - 1), key="next_slide"):
                st.session_state.slide_idx = min(n - 1, idx + 1)
                st.rerun()

        # ── Current slide ──
        s = slides[idx]
        st.markdown(_slide_html(s, idx, theme, diagrams, show_full=True),
                    unsafe_allow_html=True)

        # ── Diagram image if available ──
        if idx in diagrams and Path(diagrams[idx]).exists():
            st.image(diagrams[idx], caption=f"📊 Diagram for slide {idx+1}", use_container_width=True)

        # ── Thumbnail strip ──
        st.markdown("---")
        st.markdown("**All slides:**")
        cols = st.columns(min(n, 5))
        for j, (col, sl) in enumerate(zip(cols * ((n // 5) + 1), slides)):
            with col:
                selected = "border: 2px solid #1f6feb;" if j == idx else "border: 1px solid #1f3a5a;"
                bg = "#{:02X}{:02X}{:02X}".format(*theme.bg)
                ac = "#{:02X}{:02X}{:02X}".format(*theme.accent)
                tx = "#{:02X}{:02X}{:02X}".format(*theme.text)
                thumb_label = sl.title[:30] + ("…" if len(sl.title) > 30 else "")
                st.markdown(f"""
                <div style="background:{bg};{selected}border-radius:8px;padding:0.5rem;
                            cursor:pointer;font-size:0.7rem;text-align:center;min-height:60px;
                            display:flex;flex-direction:column;justify-content:center;">
                  <div style="background:{ac};border-radius:4px;padding:1px 4px;
                              color:{tx};font-weight:700;font-size:0.6rem;margin-bottom:3px;">
                    #{j+1}</div>
                  <div style="color:{tx};font-size:0.68rem;">{thumb_label}</div>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Go", key=f"thumb_{j}"):
                    st.session_state.slide_idx = j
                    st.rerun()

        # ── Download from Preview tab too ──
        st.markdown("---")
        if st.session_state.pptx_bytes:
            st.download_button(
                label=f"📥 Download {st.session_state.pptx_fname}",
                data=st.session_state.pptx_bytes,
                file_name=st.session_state.pptx_fname,
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key="dl_from_preview",
            )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ════════════════════════════════════════════════════════════════════════════════
with tab_hist:
    records = load_history()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 📚 Presentation History")
    with col2:
        if records:
            if st.button("🗑️ Clear History", type="secondary", use_container_width=True):
                clear_history()
                st.rerun()

    if not records:
        st.info("No presentations generated yet. Use the **🎯 Generate** tab to create your first one!")
    else:
        # ── Stats bar ──
        total_slides = sum(r.get("num_slides", 0) for r in records)
        total_size   = sum(r.get("size_kb", 0) for r in records)
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Total Presentations", len(records))
        c2.metric("🗂️ Total Slides Generated", total_slides)
        c3.metric("💾 Total Size", f"{total_size:.0f} KB")
        st.markdown("---")

        # ── Search / filter ──
        search = st.text_input("🔍 Filter by keyword", placeholder="e.g. 'transformer'")
        filtered = [r for r in records if not search or search.lower() in r.get("prompt","").lower() or search.lower() in r.get("topic","").lower()]
        st.caption(f"Showing {len(filtered)} of {len(records)} presentations")

        for r in filtered:
            created = r.get("created_at","")[:16].replace("T", " at ")
            theme_badge = r.get("theme","")
            model_badge = r.get("model","")
            p_path = Path(r.get("filepath",""))
            exists_icon = "✅" if p_path.exists() else "❌ (deleted)"

            st.markdown(f"""
            <div class="hist-card">
              <div class="hist-topic">🎓 {r.get('topic', '—')}</div>
              <div class="hist-prompt">"{r.get('prompt','')[:140]}"</div>
              <div class="hist-meta">
                🕐 {created} &nbsp;|&nbsp;
                📑 {r.get('num_slides',0)} slides &nbsp;|&nbsp;
                💾 {r.get('size_kb',0)} KB &nbsp;|&nbsp;
                🎨 {theme_badge} &nbsp;|&nbsp;
                🤖 {model_badge} &nbsp;|&nbsp;
                📁 {r.get('filename','?')} {exists_icon}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if p_path.exists():
                with open(p_path, "rb") as fh:
                    st.download_button(
                        label=f"📥 Re-download {r.get('filename','file')}",
                        data=fh.read(),
                        file_name=r.get("filename","presentation.pptx"),
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        key=f"dl_hist_{r.get('id','')}",
                    )
