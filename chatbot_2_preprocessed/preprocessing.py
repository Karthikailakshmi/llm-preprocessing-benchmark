"""Custom LLM preprocessing workflow: nine independent, reusable techniques.

Three different stages use these functions (do not mix them up):
  * OFFLINE dataset preparation : format_dataset_record / load_dataset / validate
  * DOCUMENT INGESTION          : chunk_text (the 'chunking' technique)
  * REAL-TIME QUERY PIPELINE    : clean, normalize, tokenize, filter, deduplicate,
                                  truncate, format_prompt (see PreprocessingPipeline)

Every runtime technique is toggled through PreprocessConfig. The original query
is always preserved; nothing is changed silently (the UI shows original vs
processed text, and warnings are surfaced).
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from shared.metrics import timed
from shared.prompts import (DOC_MODE, BuiltRequest, QueryPrep, build_basic_messages, system_for)
from shared.retrieval import Chunk, fixed_window_split

_FENCE = re.compile(r"(```.*?```)", re.S)


def _outside_code(text: str, fn) -> str:
    """Apply fn only to text outside ``` fenced code blocks (code stays untouched)."""
    parts = _FENCE.split(text)
    return "".join(p if i % 2 else fn(p) for i, p in enumerate(parts))


# 1. TEXT CLEANING ---------------------------------------------------------------
# Zero-width joiner/non-joiner (U+200C/U+200D) are meaningful in Indic and Persian
# scripts, so only U+200B, U+2060 and the BOM are removed.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u2060\ufeff]")


def clean_text(text: str) -> str:
    """Remove control characters and whitespace noise; keep punctuation, code
    blocks, indentation and paragraph breaks."""
    def _clean(seg: str) -> str:
        seg = _CONTROL.sub("", seg.replace("\r\n", "\n").replace("\r", "\n"))
        lines = []
        for line in seg.split("\n"):
            m = re.match(r"^([ \t]*)(.*?)[ \t]*$", line)
            lines.append(m.group(1) + re.sub(r"[ \t]{2,}", " ", m.group(2)))
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return _outside_code(text, _clean).strip()


# 2. TEXT NORMALIZATION ------------------------------------------------------------
_UNI_SPACES = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")


def normalize_text(text: str) -> str:
    """NFC Unicode form + unusual spaces -> ' '. NFC (not NFKC) so symbols such as
    superscripts and math notation keep their meaning. No lowercasing."""
    text = unicodedata.normalize("NFC", text)
    text = _UNI_SPACES.sub(" ", text)
    return text.replace("\u2028", "\n").replace("\u2029", "\n\n")


# 3. TOKENIZATION --------------------------------------------------------------------
@dataclass
class TokenCount:
    count: int
    method: str
    approximate: bool = True


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 (not installed, or vocabulary download blocked)
        return None


def count_tokens(text: str, model_id: Optional[str] = None) -> TokenCount:
    """Estimate tokens. The selected provider's own tokenizer is not available
    locally, so this is ALWAYS approximate and labelled; provider-reported usage
    is the authoritative number for evaluation."""
    enc = _encoder()
    if enc is not None:
        return TokenCount(len(enc.encode(text, disallowed_special=())), "approx:cl100k_base")
    return TokenCount(math.ceil(len(text) / 4), "approx:chars/4")


# 4. DATA FILTERING -----------------------------------------------------------------
@dataclass
class FilterResult:
    passed: bool
    reason: str = ""
    warnings: list = field(default_factory=list)


def filter_input(text: str, max_chars: int = 20000) -> FilterResult:
    """Conservative checks. Only empty or clearly malformed input is blocked; short
    queries pass. Everything else is a visible warning, never a silent drop."""
    if not text or not text.strip():
        return FilterResult(False, "The input is empty.")
    n, warnings = len(text), []
    if text.count("\ufffd") / n > 0.2:
        return FilterResult(False, "The input looks malformed (many undecodable characters).")
    if not any(ch.isalnum() for ch in text):
        warnings.append("The input contains no letters or digits.")
    if n >= 20 and max(Counter(text).values()) / n > 0.6:
        warnings.append("The input is dominated by one repeated character.")
    words = text.split()
    if len(words) >= 10 and len(set(words)) / len(words) < 0.2:
        warnings.append("The input is highly repetitive.")
    if n > max_chars:
        warnings.append(f"The input is very long ({n} characters).")
    return FilterResult(True, "", warnings)


# 5. DEDUPLICATION ------------------------------------------------------------------
_LIST_ITEM = re.compile(r"^\s*([-*+\u2022]|\d+[.)])\s+")


def deduplicate_text(text: str, min_len: int = 20) -> str:
    """Remove accidental repeats only: a message pasted twice, identical consecutive
    paragraphs or lines of >= min_len characters. List items, code blocks and
    short repeats are preserved."""
    s = text.strip()
    if "```" not in s and not any(_LIST_ITEM.match(l) for l in s.split("\n")):
        n = len(s)
        for sep in (" ", "\n", "\n\n"):
            if (n - len(sep)) % 2 == 0:
                h = (n - len(sep)) // 2
                if h >= min_len and s[h:h + len(sep)] == sep and s[h + len(sep):] == s[:h]:
                    return s[:h]

    def _dedupe(seg: str) -> str:
        paras, prev, kept = re.split(r"\n\s*\n", seg), None, []
        for p in paras:
            key = " ".join(p.split()).casefold()
            if key and key == prev and len(key) >= min_len:
                continue
            kept.append(p)
            prev = key
        out, prev = [], None
        for line in "\n\n".join(kept).split("\n"):
            key = line.strip().casefold()
            if key and key == prev and len(key) >= min_len and not _LIST_ITEM.match(line):
                continue
            out.append(line)
            prev = key
        return "\n".join(out)
    return _outside_code(text, _dedupe)


def deduplicate_chunks(hits: Sequence[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    """Drop retrieved chunks whose text is identical to an earlier (higher-ranked) one."""
    seen, out = set(), []
    for chunk, score in hits:
        key = " ".join(chunk.text.split()).casefold()
        if key not in seen:
            seen.add(key)
            out.append((chunk, score))
    return out


# 6. PROMPT FORMATTING --------------------------------------------------------------
def format_prompt(question: str, mode: str, hits: Optional[Sequence] = None,
                  history: Optional[list] = None, system_prompt: Optional[str] = None) -> list[dict]:
    """Consistent structure: system instructions, optional labelled document
    context, then the user question. Identical for every provider."""
    parts = []
    if mode == DOC_MODE and hits:
        ctx = "\n\n".join(f"[{i}] source: {c.ref}\n{c.text}" for i, (c, _) in enumerate(hits, 1))
        parts.append(f"## Retrieved document context\n{ctx}")
    parts.append(f"## User question\n{question}")
    return [{"role": "system", "content": system_prompt or system_for(mode)},
            *(history or []), {"role": "user", "content": "\n\n".join(parts)}]


# 7. TRUNCATION AND PADDING ---------------------------------------------------------
@dataclass
class TruncationResult:
    text: str
    truncated: bool
    tokens_before: int
    tokens_after: int
    method: str


def truncate_text(text: str, max_tokens: int, model_id: Optional[str] = None,
                  keep_tail_fraction: float = 0.3) -> TruncationResult:
    """If over budget, keep the start AND the end (instructions often sit at the
    end) and insert a visible marker. Token count is approximate."""
    tok = count_tokens(text, model_id)
    if tok.count <= max_tokens:
        return TruncationResult(text, False, tok.count, tok.count, tok.method)
    target = int(len(text) * max_tokens / tok.count * 0.95)
    tail = int(target * keep_tail_fraction)
    head = target - tail
    marker = "\n[...input shortened to fit the model's limit...]\n"
    new = text[:head].rstrip() + marker + (text[len(text) - tail:].lstrip() if tail > 0 else "")
    return TruncationResult(new, True, tok.count, count_tokens(new, model_id).count, tok.method)


def truncate_chunks(hits: Sequence[tuple[Chunk, float]], token_budget: int,
                    model_id: Optional[str] = None) -> tuple[list[tuple[Chunk, float]], int]:
    """Keep the most relevant chunks (list is ranked) that fit the budget."""
    kept, used = [], 0
    for i, (chunk, score) in enumerate(hits):
        n = count_tokens(chunk.text, model_id).count
        if used + n > token_budget:
            if not kept:  # always keep the best chunk, shortened if needed
                short = truncate_text(chunk.text, max(token_budget, 1), model_id).text
                kept.append((dataclasses.replace(chunk, text=short), score))
            return kept, len(hits) - len(kept)
        kept.append((chunk, score))
        used += n
    return kept, 0


def pad_batch(sequences: list[list], pad_value=0, length: Optional[int] = None) -> list[list]:
    """Right-pad sequences to equal length. NOT used for ordinary API requests;
    only for batching/embedding code paths that require equal lengths."""
    target = length if length is not None else max((len(s) for s in sequences), default=0)
    return [list(s[:target]) + [pad_value] * (target - len(s)) for s in sequences]


# 8. CHUNKING (document ingestion) ----------------------------------------------------
_SENTENCE = re.compile(r"(?<=[.!?\u0964\u0965\u3002\uff01\uff1f])\s+|\n{2,}")


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Sentence/paragraph-aware chunks of ~size characters with ~overlap characters
    carried into the next chunk. Applies to documents, never to short questions."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    units: list[str] = []
    for u in _SENTENCE.split(text):
        u = u.strip()
        if not u:
            continue
        units.extend([u] if len(u) <= size else [u[i:i + size] for i in range(0, len(u), size)])
    chunks, cur, cur_len = [], [], 0
    for u in units:
        if cur and cur_len + len(u) + 1 > size:
            chunks.append(" ".join(cur))
            carry, c_len = [], 0
            for prev in reversed(cur):
                if c_len + len(prev) + 1 > overlap:
                    break
                carry.insert(0, prev)
                c_len += len(prev) + 1
            cur, cur_len = carry, sum(len(x) + 1 for x in carry)
        cur.append(u)
        cur_len += len(u) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


# 9. DATASET FORMATTING (offline) ------------------------------------------------------
DATASET_FIELDS = ["id", "category", "question", "expected_answer", "document_ref",
                  "language", "repeat_group", "notes"]
_REQUIRED = ("id", "category", "question", "expected_answer")


def format_dataset_record(id: str, category: str, question: str, expected_answer: str,
                          document_ref: str = "", language: str = "en",
                          repeat_group: str = "", notes: str = "") -> dict:
    return {"id": str(id).strip(), "category": category.strip(), "question": question,
            "expected_answer": expected_answer, "document_ref": document_ref.strip(),
            "language": language.strip(), "repeat_group": repeat_group.strip(), "notes": notes}


def validate_dataset_records(records: list[dict]) -> list[str]:
    errors, seen = [], set()
    for i, r in enumerate(records, start=1):
        for f in _REQUIRED:
            if not str(r.get(f, "")).strip():
                errors.append(f"Row {i}: missing '{f}'.")
        rid = str(r.get("id", "")).strip()
        if rid in seen:
            errors.append(f"Row {i}: duplicate id '{rid}'.")
        seen.add(rid)
    return errors


def load_dataset(path: str | Path) -> list[dict]:
    """Load CSV or JSONL evaluation data into the common schema (raises ValueError)."""
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif p.suffix.lower() == ".csv":
        with p.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    else:
        raise ValueError("Dataset must be .csv or .jsonl")
    records = [format_dataset_record(**{k: v for k, v in r.items() if k in DATASET_FIELDS}) for r in rows]
    errors = validate_dataset_records(records)
    if errors:
        raise ValueError("Invalid dataset:\n" + "\n".join(errors[:20]))
    return records


def save_dataset(records: list[dict], path: str | Path) -> None:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    else:
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=DATASET_FIELDS)
            w.writeheader()
            w.writerows(records)


# CONFIG + REAL-TIME PIPELINE -----------------------------------------------------------
@dataclass
class PreprocessConfig:
    text_cleaning: bool = True
    normalization: bool = True
    tokenization: bool = True      # token counting (approximate)
    data_filtering: bool = True
    deduplication: bool = True
    prompt_formatting: bool = True
    truncation: bool = True
    chunking: bool = True          # applies at document ingestion
    padding: bool = False          # not applicable to ordinary API requests


class PreprocessingPipeline:
    name = "preprocessed"

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

    def techniques(self) -> dict:
        return dataclasses.asdict(self.config)

    def ingestion_signature(self) -> str:
        return "sentence_aware" if self.config.chunking else "fixed_window"

    def split(self, text: str, size: int, overlap: int) -> list[str]:
        return chunk_text(text, size, overlap) if self.config.chunking else \
            fixed_window_split(text, size, overlap)

    def process_query(self, question: str, max_input_tokens: int) -> QueryPrep:
        cfg, ms, warnings, details = self.config, {}, [], {}
        if not question or not question.strip():
            return QueryPrep(question, "", ms, blocked=True, reason="The input is empty.")
        text = question
        if cfg.normalization:
            with timed(ms, "normalization"):
                text = normalize_text(text)
        if cfg.text_cleaning:
            with timed(ms, "text_cleaning"):
                text = clean_text(text)
        if cfg.data_filtering:
            with timed(ms, "data_filtering"):
                res = filter_input(text)
            warnings += res.warnings
            if not res.passed:
                return QueryPrep(question, text, ms, warnings, True, res.reason)
        if cfg.deduplication:
            with timed(ms, "deduplication"):
                text = deduplicate_text(text)
        if cfg.tokenization:
            with timed(ms, "tokenization"):
                tok = count_tokens(text)
            details.update(query_tokens_estimate=tok.count, token_method=tok.method)
        if cfg.truncation:
            with timed(ms, "truncation"):
                tr = truncate_text(text, max_input_tokens // 2)
            if tr.truncated:
                text = tr.text
                warnings.append("The question was shortened to fit the model's input limit.")
                details["query_truncated"] = True
        if not text.strip():
            text = question
            warnings.append("Preprocessing produced empty text, so the original query was used.")
        return QueryPrep(question, text, ms, warnings, False, "", details)

    def build_request(self, prep: QueryPrep, hits, history, mode, max_input_tokens) -> BuiltRequest:
        cfg, ms, warnings = self.config, {}, []
        hits = list(hits) if mode == DOC_MODE else []
        if hits and cfg.deduplication:
            with timed(ms, "deduplication"):
                hits = deduplicate_chunks(hits)
        if hits and cfg.truncation:
            with timed(ms, "truncation"):
                budget = max(max_input_tokens - count_tokens(prep.processed).count - 600, 500)
                hits, dropped = truncate_chunks(hits, budget)
            if dropped:
                warnings.append(f"{dropped} lower-ranked context chunk(s) were dropped to fit the limit.")
        with timed(ms, "prompt_formatting"):
            messages = (format_prompt(prep.processed, mode, hits, history) if cfg.prompt_formatting
                        else build_basic_messages(prep.processed, hits, history, mode))
        return BuiltRequest(messages, hits, ms, warnings)
