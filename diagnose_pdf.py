"""Usage:  python diagnose_pdf.py YOUR_FILE.pdf
Shows what each extractor reads, so we can see why questions fail."""
import sys
from shared.document_loader import _text_score
from pypdf import PdfReader
import pymupdf

path = sys.argv[1]
r = PdfReader(path); m = pymupdf.open(path)
print(f"{len(r.pages)} pages")
bad = 0
for i in range(len(r.pages)):
    a = r.pages[i].extract_text() or ""
    b = m[i].get_text("text", sort=True) or ""
    if max(_text_score(a), _text_score(b)) < 10:
        bad += 1
    if i < 3 or i in (len(r.pages)//2,):
        print(f"\n--- page {i+1} | pypdf score {_text_score(a)} | pymupdf score {_text_score(b)}")
        print("pypdf  :", repr(a[:250]))
        print("pymupdf:", repr(b[:250]))
print(f"\nPages with little/unreadable text: {bad}")