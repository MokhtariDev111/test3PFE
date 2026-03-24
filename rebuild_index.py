"""Rebuild the FAISS index with new semantic chunking."""
from pathlib import Path
from modules.config_loader import CONFIG
from modules.ingestion import ingest_directory
from modules.ocr import run_ocr
from modules.text_processing import process_pages
from modules.embeddings import build_vector_db

print("=" * 50)
print("REBUILDING VECTOR INDEX")
print("=" * 50)

# Step 1: Ingest
raw_dir = Path(CONFIG["paths"]["data_raw"])
print(f"\n[1/4] Ingesting from {raw_dir}...")
pages = ingest_directory(raw_dir)
print(f"      → {len(pages)} pages loaded")

if not pages:
    print("\n❌ No documents found! Add PDFs to data/raw/ first.")
    exit(1)

# Step 2: OCR (if needed)
print("\n[2/4] Running OCR on images...")
pages = run_ocr(pages)

# Step 3: Chunk
print("\n[3/4] Semantic chunking...")
chunks = process_pages(pages)
print(f"      → {len(chunks)} chunks created")

# Step 4: Embed & Index
print("\n[4/4] Building FAISS index...")
build_vector_db(chunks)

print("\n" + "=" * 50)
print("✅ INDEX REBUILT SUCCESSFULLY!")
print("=" * 50)
print(f"\nYou can now run: python -m modules.evaluation")