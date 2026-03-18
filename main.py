"""
main.py  —  Real-Time AI Teaching Assistant
=============================================
Central entry point for the project.

Run:
    python main.py

This file will grow step-by-step.  For now it:
  1. Loads configuration
  2. Prints a project banner
  3. Checks that all required directories exist
"""

import sys
from pathlib import Path

# ── make sure modules/ is on the path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from pathlib import Path
from modules.config_loader import CONFIG
from modules.ingestion import ingest_directory
from modules.ocr import run_ocr
from modules.text_processing import process_pages
from modules.embeddings import build_vector_db
from modules.retrieval import Retriever
from modules.llm import LLMEngine
from modules.pedagogical_engine import PedagogicalEngine
from modules.slide_generator import build_slides
from modules.pptx_exporter import export_pptx

def check_directories() -> None:
    """Ensure all defined data directories exist; create them if not."""
    paths = CONFIG.get("paths", {})
    for key, rel_path in paths.items():
        abs_path = Path(rel_path)
        abs_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✔  [{key}]  {abs_path.resolve()}")

def test_pipeline():
    """Runs all completed steps sequentially to verify they work together."""
    print("\n[Step 2] Data Ingestion...")
    raw_dir = Path(CONFIG["paths"]["data_raw"])
    pages = ingest_directory(raw_dir)
    
    print("\n[Step 3] OCR Integration...")
    pages = run_ocr(pages)
    
    # Quick sanity check on OCR text
    imgs = [p for p in pages if p.type == "image"]
    if imgs:
        print(f"  > OCR found text in {len(imgs)} images. Example: {imgs[0].text[:50]}...")

    print("\n[Step 4] Text Processing (Cleaning & Chunking)...")
    chunks = process_pages(pages)
    
    print("\n[Step 5] Building Vector DB...")
    build_vector_db(chunks)
    
    print("\n[Step 6] Retrieval / RAG Core...")
    retriever = Retriever()
    query = "What is the Transformer architecture?"
    print(f"  > Searching for: '{query}'")
    results = retriever.search(query, top_k=2)
    
    for i, res in enumerate(results, 1):
        print(f"  [Result {i} | {res.source} p.{res.page}]")
        print(f"  {res.text[:100]}...\n")
        
    print(f"\n[Step 7] LLM raw answer for context check ({CONFIG['llm']['model']})...")
    llm = LLMEngine()
    answer = llm.generate(query, results)
    import textwrap
    print(textwrap.fill(answer, width=80))

    print(f"\n[Step 8] Pedagogical Engine — Generating lesson plan...")
    engine = PedagogicalEngine()
    lesson = engine.generate_lesson(query, results)

    print(f"\n[Step 9] Building slide structure...")
    slides = build_slides(lesson)

    print(f"\n── Slide Structure ({len(slides)} slide(s)) ──────────────────")
    for i, s in enumerate(slides, 1):
        print(f"\n  [{s.slide_type.upper()}] Slide {i}: {s.title}")
        for b in s.bullets:
            print(f"    • {b}")
        if s.speaker_notes:
            print(f"  Notes: {s.speaker_notes[:80]}...")

    print(f"\n[Step 10] Exporting to PowerPoint...")
    pptx_path = export_pptx(slides)
    
    print(f"\n✅ Pipeline complete. .pptx file saved to:\n   {pptx_path}")


def main() -> None:
    print("=" * 60)
    print(f"  {CONFIG['project']['name']}  v{CONFIG['project']['version']}")
    print(f"  {CONFIG['project']['description']}")
    print("=" * 60)

    print("\nVerifying directories...")
    check_directories()
    
    print("\nRunning pipeline test...")
    test_pipeline()


if __name__ == "__main__":
    main()
