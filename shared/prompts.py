"""Common system prompts, the baseline prompt builder, and pipeline data types.

Both chatbots use the SAME system prompts. The baseline sends the original
query with only this required formatting; the preprocessed bot replaces the
formatting step with preprocessing.format_prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from shared.retrieval import Chunk, fixed_window_split

GENERAL_MODE = "General Chat"
DOC_MODE = "Ask Uploaded Documents"

SYSTEM_GENERAL = (
    "You are a helpful, accurate assistant. Answer clearly and concisely. "
    "If you are unsure, say so instead of guessing. "
    "Reply in the same language as the user's question."
)
SYSTEM_DOCS = (
    SYSTEM_GENERAL + " Answer ONLY from the provided document context and cite sources like [1]. "
    'If the context does not contain the answer, reply exactly: '
    '"The answer is not in the uploaded documents."'
)

Hit = tuple[Chunk, float]


@dataclass
class QueryPrep:
    original: str
    processed: str
    step_ms: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    blocked: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class BuiltRequest:
    messages: list
    used_hits: list
    step_ms: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def system_for(mode: str) -> str:
    return SYSTEM_DOCS if mode == DOC_MODE else SYSTEM_GENERAL


def build_basic_messages(question: str, hits: list[Hit], history: list[dict], mode: str) -> list[dict]:
    messages = [{"role": "system", "content": system_for(mode)}, *history]
    if mode == DOC_MODE:
        ctx = "\n\n".join(f"[{i}] ({c.ref})\n{c.text}" for i, (c, _) in enumerate(hits, 1))
        user = f"Context:\n{ctx}\n\nQuestion: {question}"
    else:
        user = question
    messages.append({"role": "user", "content": user})
    return messages


class BaselinePipeline:
    """No preprocessing: original query + common system prompt + required formatting."""

    name = "baseline"

    def techniques(self) -> dict:
        return {}

    def ingestion_signature(self) -> str:
        return "fixed_window"

    def split(self, text: str, size: int, overlap: int) -> list[str]:
        return fixed_window_split(text, size, overlap)

    def process_query(self, question: str, max_input_tokens: int) -> QueryPrep:
        return QueryPrep(question, question)

    def build_request(self, prep, hits, history, mode, max_input_tokens) -> BuiltRequest:
        return BuiltRequest(build_basic_messages(prep.processed, hits, history, mode), hits)
