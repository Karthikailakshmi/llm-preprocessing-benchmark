"""Offline checks (no API keys needed): preprocessing, loaders, retrieval, dataset, formatting."""
import io
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chatbot_2_preprocessed.preprocessing import (  # noqa: E402
    PreprocessConfig, PreprocessingPipeline, chunk_text, clean_text, count_tokens,
    deduplicate_text, filter_input, load_dataset, normalize_text, pad_batch, truncate_text)
from shared.document_loader import load_document  # noqa: E402
from shared.metrics import format_duration  # noqa: E402
from shared.prompts import DOC_MODE, GENERAL_MODE, BaselinePipeline  # noqa: E402
from shared.retrieval import VectorIndex, build_chunks  # noqa: E402

n = 0


def check(cond, label):
    global n
    n += 1
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        raise SystemExit(1)


# 1-2 cleaning / normalization
check(clean_text("  hi   there \x00\x07  \n\n\n\nbye ") == "hi there\n\nbye", "clean_text collapses noise")
code = "```py\nx  =   1\n```"
check(clean_text("a   b\n" + code) == "a b\n" + code, "clean_text keeps code blocks")
check(normalize_text("a\u00a0b") == "a b", "normalize nbsp")
check("\u200c" in clean_text("ab\u200ccd"), "ZWNJ preserved (Persian/Indic)")
check(normalize_text("x²") == "x²", "normalize keeps superscripts")
# 3 tokens
t = count_tokens("hello world")
check(t.count > 0 and t.approximate and t.method.startswith("approx"), "count_tokens labelled approximate")
# 4 filtering
check(not filter_input("   ").passed, "filter blocks empty")
check(filter_input("?").passed, "filter allows very short input")
check(filter_input("a" * 100).warnings != [], "filter warns on repetition")
# 5 dedup
dup = "Please summarise the attached report for me."
check(deduplicate_text(dup + " " + dup) == dup, "dedup removes pasted-twice message")
lst = "- same item here please ok\n- same item here please ok"
check(deduplicate_text(lst) == lst, "dedup keeps list items")
check(deduplicate_text("yes yes") == "yes yes", "dedup keeps short repeats")
# 7 truncation / padding
long = "word " * 5000
tr = truncate_text(long, 500)
check(tr.truncated and tr.tokens_after < tr.tokens_before and "shortened" in tr.text, "truncate marks shortening")
check(pad_batch([[1], [1, 2, 3]]) == [[1, 0, 0], [1, 2, 3]], "pad_batch")
# 8 chunking
doc = " ".join(f"Sentence number {i} is here." for i in range(200))
ch = chunk_text(doc, 300, 60)
check(len(ch) > 1 and all(len(c) <= 420 for c in ch), "chunk_text sizes")
check(ch[1].split(".")[0] in ch[0], "chunk_text overlaps")
# formatting
check(format_duration(250) == "250 ms" and format_duration(2500) == "2.50 s"
      and format_duration(90000) == "1.50 min" and format_duration(None) == "n/a", "format_duration")

# loaders
tmp = Path(tempfile.mkdtemp())
txt = (ROOT / "evaluation/sample_docs/northwind_handbook.txt").read_bytes()
r = load_document("northwind_handbook.txt", txt)
check(r.ok and r.segments, "load txt")
check(not load_document("x.exe", b"abc").ok, "unsupported type reported")
check(not load_document("e.txt", b"").ok, "empty file reported")
check(load_document("a.csv", b"a,b\n1,2\n").ok, "load csv")
check(load_document("a.json", b'{"k": "v"}').ok, "load json")
check(load_document("a.html", b"<html><script>x</script><p>Hello</p></html>").segments[0].text == "Hello", "load html")
import docx, fitz, openpyxl
from pptx import Presentation
d = docx.Document(); d.add_paragraph("Docx paragraph text"); b = io.BytesIO(); d.save(b)
check(load_document("a.docx", b.getvalue()).ok, "load docx")
wb = openpyxl.Workbook(); wb.active.append(["h1", "h2"]); wb.active.append([1, 2]); b = io.BytesIO(); wb.save(b)
check(load_document("a.xlsx", b.getvalue()).segments[0].unit == "sheet", "load xlsx")
p = Presentation(); s = p.slides.add_slide(p.slide_layouts[5]); s.shapes.title.text = "Slide title text"
b = io.BytesIO(); p.save(b)
check(load_document("a.pptx", b.getvalue()).segments[0].unit == "slide", "load pptx")
pdf = fitz.open(); pg = pdf.new_page(); pg.insert_text((72, 72), "PDF page text for testing here"); data = pdf.tobytes()
rp = load_document("a.pdf", data)
check(rp.ok and rp.segments[0].page == 1, "load pdf with page number")
blank = fitz.open(); blank.new_page()
rb = load_document("blank.pdf", blank.tobytes())
check(not rb.ok and "OCR" in (rb.error + " ".join(rb.warnings)) or not rb.ok, "blank pdf reports no text")

# retrieval, same index for both chunkers
for pipe in (BaselinePipeline(), PreprocessingPipeline()):
    chunks = build_chunks(r.segments, 300, 50, pipe.split)
    idx = VectorIndex(); idx.build(chunks)
    top = idx.search("How many days per week can employees work remotely?", 2)
    check(any("remotely" in c.text for c, _ in top), f"retrieval finds remote policy ({pipe.name}, {idx.backend})")

# pipelines
pre = PreprocessingPipeline()
prep = pre.process_query("  wat   is  teh \u00a0 capital?? \x00 ", 6000)
check(prep.processed == "wat is teh capital??" and prep.original.startswith("  wat"), "pipeline cleans, keeps original")
check(pre.process_query("   ", 6000).blocked, "pipeline blocks empty")
check(BaselinePipeline().process_query("  x  ", 6000).processed == "  x  ", "baseline leaves query untouched")
built = pre.build_request(prep, [], [], GENERAL_MODE, 6000)
check(built.messages[-1]["content"].startswith("## User question"), "preprocessed prompt structure")
off = PreprocessingPipeline(PreprocessConfig(prompt_formatting=False)).build_request(prep, [], [], GENERAL_MODE, 6000)
check(off.messages[-1]["content"] == prep.processed, "prompt_formatting off falls back to baseline format")
check({"text_cleaning", "normalization"} <= set(prep.step_ms), "per-step timings recorded")

# dataset
ds = load_dataset(ROOT / "evaluation/test_questions.csv")
check(len(ds) >= 20 and {r["category"] for r in ds} >= {"general", "coding", "multilingual", "document"}, "dataset loads")
print(f"\nAll {n} checks passed.")
