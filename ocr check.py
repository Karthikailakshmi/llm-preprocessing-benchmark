"""Usage:  python ocr_check.py YOUR_FILE.pdf
Finds pages with no extractable text, tries OCR on each, prints the real
reason it fails, and saves the page as a PNG so you can look at it."""
import io, sys
from shared.document_loader import ocr_available, _ocr
import pymupdf
from PIL import Image
from pypdf import PdfReader

path = sys.argv[1]
print("OCR available:", ocr_available())
reader = PdfReader(path)
doc = pymupdf.open(path)
for i, page in enumerate(reader.pages):
    a = (page.extract_text() or "").strip()
    b = (doc[i].get_text("text") or "").strip()
    if len(a) >= 20 or len(b) >= 20:
        continue
    print(f"\n=== page {i+1}: pypdf={len(a)} chars, pymupdf={len(b)} chars")
    try:
        pix = doc[i].get_pixmap(dpi=200)
        out = f"ocr_page_{i+1}.png"
        pix.save(out)
        print("rendered OK, saved", out)
        text = _ocr(Image.open(io.BytesIO(pix.tobytes("png"))))
        print("OCR text length:", len(text.strip()))
        print("OCR text preview:", repr(text.strip()[:200]))
    except Exception as e:
        print("ERROR:", type(e).__name__, e)