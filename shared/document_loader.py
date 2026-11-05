"""
Document ingestion shared by BOTH chatbots: validate, extract text, clean.

Supported: PDF, DOCX, TXT, MD, CSV, XLSX, PPTX, HTML, JSON, PNG, JPG/JPEG.
PDF text extraction uses pypdf.
Scanned PDF pages and images need OCR (pytesseract + Tesseract).
PyMuPDF is used optionally to render scanned PDF pages for OCR.
If OCR is unavailable, the result includes a warning.
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

from shared.config import MAX_UPLOAD_MB, OCR_LANG


SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".pptx",
    ".html", ".htm", ".json", ".png", ".jpg", ".jpeg",
]

MAX_TABLE_ROWS = 5000


@dataclass
class Segment:
    text: str
    source: str
    page: Optional[Union[int, str]] = None
    unit: str = ""  # "page", "slide", "sheet" or ""

    @property
    def ref(self) -> str:
        return (
            f"{self.source}, {self.unit} {self.page}"
            if self.page is not None and self.unit
            else self.source
        )


@dataclass
class LoadResult:
    filename: str
    segments: list[Segment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff]")


def clean_document_text(text: str) -> str:
    """Common document cleaning applied identically in both chatbots."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = re.sub(r"[ \t\u00a0]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---- OCR ---------------------------------------------------------------------

def _find_tesseract() -> None:
    """Use the standard Windows install folders if tesseract is not on PATH."""
    import os
    import shutil
    import pytesseract
    if shutil.which("tesseract"):
        return
    for path in (
        os.getenv("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if path and os.path.isfile(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


@lru_cache(maxsize=1)
def ocr_available() -> bool:
    """Check whether pytesseract and the Tesseract program are available."""
    try:
        import pytesseract
        _find_tesseract()
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001
        return False


def _ocr(image) -> str:
    import pytesseract
    return pytesseract.image_to_string(image, lang=OCR_LANG)


# ---- Format loaders: (filename, bytes) -> (segments, warnings) ---------------

def _decode(data: bytes, warnings: list[str]) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue

    warnings.append(
        "Text was not valid UTF-8; decoded as Latin-1 "
        "(some characters may be wrong)."
    )
    return data.decode("latin-1", errors="replace")


def _load_text(name, data):
    warnings: list[str] = []
    return [Segment(_decode(data, warnings), name)], warnings



_WORD = re.compile(r"[^\W\d_]{2,20}")


def _text_score(text: str) -> int:
    """Rough readability score: count of normal-length words, minus junk."""
    if not text:
        return 0
    words = len(_WORD.findall(text))
    junk = text.count("(cid:") * 5 + text.count("\ufffd") * 3
    glued = len(re.findall(r"[^\W\d_]{30,}", text)) * 5  # words with no spaces
    return words - junk - glued


def _load_pdf(name, data):
    """
    Extract regular PDF text using pypdf.
    For pages with little or no text, optionally attempt OCR using
    PyMuPDF for rendering and Tesseract for text recognition.
    """
    from pypdf import PdfReader

    segments = []
    warnings = []
    pages_needing_ocr = 0
    ocr_failures = 0

    reader = PdfReader(io.BytesIO(data))

    # Only import PyMuPDF if OCR is needed.
    pdf_doc = None
    rendering_checked = False
    rendering_available = False

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        # Second opinion from PyMuPDF; keep whichever text is more readable.
        # (pypdf can return garbled / space-less text for edited PDFs.)
        try:
            import pymupdf
            if pdf_doc is None:
                pdf_doc = pymupdf.open(stream=data, filetype="pdf")
            alt = pdf_doc[i - 1].get_text("text", sort=True) or ""
            if _text_score(alt) > _text_score(text):
                text = alt
        except Exception:  # noqa: BLE001
            pass

        if len(text.strip()) < 20:
            pages_needing_ocr += 1

            if ocr_available():
                try:
                    if not rendering_checked:
                        rendering_checked = True
                        try:
                            import pymupdf
                            rendering_available = True
                        except Exception:  # noqa: BLE001
                            rendering_available = False

                    if rendering_available:
                        import pymupdf
                        from PIL import Image

                        if pdf_doc is None:
                            pdf_doc = pymupdf.open(
                                stream=data,
                                filetype="pdf",
                            )

                        pix = pdf_doc[i - 1].get_pixmap(dpi=200)
                        image = Image.open(
                            io.BytesIO(pix.tobytes("png"))
                        )
                        ocr_text = _ocr(image)

                        if ocr_text.strip():
                            text = ocr_text
                        # else: OCR ran fine and the page has no text
                        # (blank or picture-only). Nothing to warn about.
                    else:
                        ocr_failures += 1

                except Exception:  # noqa: BLE001
                    ocr_failures += 1

        segments.append(Segment(text, name, i, "page"))

    if pdf_doc is not None:
        pdf_doc.close()

    if pages_needing_ocr and not ocr_available():
        warnings.append(
            f"{pages_needing_ocr} page(s) had little or no extractable "
            "text. OCR is unavailable. Install pytesseract and the "
            "Tesseract program to read scanned pages."
        )
    elif ocr_failures:
        warnings.append(
            f"OCR could not process {ocr_failures} page(s). "
            "Check PyMuPDF, Tesseract, and OCR language configuration."
        )

    return segments, warnings


def _load_docx(name, data):
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))

    return [Segment("\n".join(parts), name)], []


def _frame_text(df, warnings, label):
    if len(df) > MAX_TABLE_ROWS:
        warnings.append(
            f"{label}: only the first {MAX_TABLE_ROWS} "
            f"of {len(df)} rows were read."
        )
        df = df.head(MAX_TABLE_ROWS)

    return df.to_csv(index=False)


def _load_csv(name, data):
    import pandas as pd

    warnings: list[str] = []

    try:
        df = pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
        warnings.append("CSV decoded as Latin-1.")

    return [Segment(_frame_text(df, warnings, name), name)], warnings


def _load_xlsx(name, data):
    import pandas as pd

    warnings: list[str] = []
    sheets = pd.read_excel(
        io.BytesIO(data),
        sheet_name=None,
        engine="openpyxl",
    )

    segments = [
        Segment(
            _frame_text(df, warnings, f"sheet '{sheet_name}'"),
            name,
            sheet_name,
            "sheet",
        )
        for sheet_name, df in sheets.items()
    ]

    return segments, warnings


def _load_pptx(name, data):
    from pptx import Presentation

    segments = []

    for i, slide in enumerate(
        Presentation(io.BytesIO(data)).slides,
        start=1,
    ):
        parts = []

        for shape in slide.shapes:
            if (
                getattr(shape, "has_text_frame", False)
                and shape.text_frame.text.strip()
            ):
                parts.append(shape.text_frame.text)

            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    parts.append(
                        " | ".join(cell.text.strip() for cell in row.cells)
                    )

        segments.append(Segment("\n".join(parts), name, i, "slide"))

    return segments, []


def _load_html(name, data):
    from bs4 import BeautifulSoup

    warnings: list[str] = []
    soup = BeautifulSoup(_decode(data, warnings), "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    return [Segment(soup.get_text("\n"), name)], warnings


def _load_json(name, data):
    warnings: list[str] = []
    raw = _decode(data, warnings)

    try:
        raw = json.dumps(
            json.loads(raw),
            indent=2,
            ensure_ascii=False,
        )
    except json.JSONDecodeError:
        warnings.append(
            "File is not valid JSON; it was read as plain text."
        )

    return [Segment(raw, name)], warnings


def _load_image(name, data):
    if not ocr_available():
        raise RuntimeError(
            "OCR is unavailable. Install Tesseract and "
            "'pip install pytesseract' to read images."
        )

    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        text = _ocr(image)

    return [Segment(text, name)], []


# ---- Loader registry ---------------------------------------------------------

LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_text,
    ".md": _load_text,
    ".csv": _load_csv,
    ".xlsx": _load_xlsx,
    ".pptx": _load_pptx,
    ".html": _load_html,
    ".htm": _load_html,
    ".json": _load_json,
    ".png": _load_image,
    ".jpg": _load_image,
    ".jpeg": _load_image,
}


# ---- Public API --------------------------------------------------------------

def load_document(filename: str, data: bytes) -> LoadResult:
    """Validate, extract and clean an uploaded document."""
    ext = Path(filename).suffix.lower()

    if ext not in LOADERS:
        return LoadResult(
            filename,
            ok=False,
            error=(
                f"Unsupported file type '{ext or 'none'}'. "
                f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
            ),
        )

    if not data:
        return LoadResult(
            filename,
            ok=False,
            error="The file is empty.",
        )

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        return LoadResult(
            filename,
            ok=False,
            error=f"The file is larger than {MAX_UPLOAD_MB} MB.",
        )

    try:
        segments, warnings = LOADERS[ext](filename, data)

    except ImportError as exc:
        return LoadResult(
            filename,
            ok=False,
            error=f"Missing dependency for {ext}: {exc.name}.",
        )

    except Exception as exc:  # noqa: BLE001
        return LoadResult(
            filename,
            ok=False,
            error=(
                f"Could not extract text "
                f"({type(exc).__name__}): {exc}"
            ),
        )

    cleaned = []

    for segment in segments:
        segment.text = clean_document_text(segment.text)
        if segment.text:
            cleaned.append(segment)

    if not cleaned:
        return LoadResult(
            filename,
            warnings=warnings,
            ok=False,
            error=(
                "No text could be extracted. The file may be scanned, "
                "empty, or require OCR."
            ),
        )

    return LoadResult(
        filename=filename,
        segments=cleaned,
        warnings=warnings,
        ok=True,
    )