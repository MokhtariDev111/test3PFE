import sys
from pathlib import Path
import json

ROOT = Path('C:\\stage PFE\\PFE Project\\test3')
sys.path.insert(0, str(ROOT))

from modules.config_loader import CONFIG
from modules.ingestion import ingest_directory
from modules.ocr import run_ocr
from modules.text_processing import process_pages
from modules.pedagogical_engine import PedagogicalEngine

print('[1] Ingesting...')
pages = ingest_directory(ROOT / CONFIG['paths']['data_raw'])
print(f'Pages: {len(pages)}')

pdf_images = {p.source: p.image for p in pages if p.type == 'pdf_image' and isinstance(p.image, str)}
available_image_ids = list(pdf_images.keys())
print(f'Found {len(available_image_ids)} PDF images.')

pages = run_ocr(pages)
chunks = process_pages(pages)
print(f'Chunks: {len(chunks)}')

if not chunks:
    print('ERROR: No chunks produced! FAISS will fail.')
else:
    eng = PedagogicalEngine()
    print('[2] Generating lesson...')
    lesson = eng.generate_lesson('test', chunks[:2], num_slides=2, available_images=available_image_ids[:2])
    print('Slides:', len(lesson.get('slides', [])))
    if not lesson.get('slides'):
        print('RAW LLM OUTPUT START: \n', lesson.get('_raw', 'None').encode('utf-8'), '\nRAW LLM OUTPUT END')
