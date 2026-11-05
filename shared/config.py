"""Central configuration shared by both chatbots.

Each provider entry represents one selectable model.
Groq is listed three times to support three models using the same API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


DEFAULT_MAX_INPUT_TOKENS = 6000


@dataclass(frozen=True)
class ProviderConfig:
    key: str
    label: str
    model_id: str
    base_url: str
    api_key_env: str
    rpm: int
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    token_param: str = "max_tokens"
    notes: str = ""

    @property
    def api_key(self) -> str:
        return _env(self.api_key_env)

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def configured(self) -> bool:
        return self.has_key and bool(self.model_id)


def _provider(
    key: str,
    label: str,
    model_default: str,
    base_url: str,
    key_env: str,
    rpm: int,
    token_param: str = "max_tokens",
    notes: str = "",
    model_env: str | None = None,
) -> ProviderConfig:
    env_prefix = key.upper()

    return ProviderConfig(
        key=key,
        label=label,
        model_id=_env(
            model_env or f"{env_prefix}_MODEL",
            model_default,
        ),
        base_url=_env(
            f"{env_prefix}_BASE_URL",
            base_url,
        ),
        api_key_env=key_env,
        rpm=_int(f"{env_prefix}_RPM", rpm),
        max_input_tokens=_int(
            f"{env_prefix}_MAX_INPUT_TOKENS",
            DEFAULT_MAX_INPUT_TOKENS,
        ),
        token_param=token_param,
        notes=notes,
    )


# Five selectable model entries.
# The three Groq entries share GROQ_API_KEY and the Groq endpoint.
PROVIDERS: dict[str, ProviderConfig] = {
    p.key: p
    for p in [
        _provider(
            "gemini",
            "Google Gemini",
            "gemini-3.5-flash-lite",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "GEMINI_API_KEY",
            5,
        ),
        _provider(
            "nvidia",
            "NVIDIA Nemotron",
            "nvidia/nemotron-3.5-bolt-e4b",
            "https://integrate.api.nvidia.com/v1",
            "NVIDIA_API_KEY",
            40,
        ),
        _provider(
            "groq",
            "Groq - GPT-OSS 20B",
            "openai/gpt-oss-20b",
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY",
            30,
            token_param="max_completion_tokens",
            model_env="GROQ_MODEL",
        ),
        _provider(
            "groq_qwen",
            "Groq - Qwen 3.8 27B",
            "qwen/qwen3.8-27b",
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY",
            30,
            model_env="GROQ_MODEL_2",
        ),
        _provider(
            "groq_120b",
            "Groq - GPT-OSS 120B",
            "openai/gpt-oss-120b",
            "https://api.groq.com/openai/v1",
            "GROQ_API_KEY",
            30,
            token_param="max_completion_tokens",
            model_env="GROQ_MODEL_3",
        ),
    ]
}


# Shared generation settings.
GENERATION = {
    "temperature": _float("GEN_TEMPERATURE", 0.2),
    "max_tokens": _int("GEN_MAX_TOKENS", 1500),
    "timeout_s": _float("GEN_TIMEOUT_S", 60.0),
}


# Shared retrieval settings.
RAG = {
    "chunk_size": _int("CHUNK_SIZE", 800),
    "chunk_overlap": _int("CHUNK_OVERLAP", 120),
    "top_k": _int("TOP_K", 4),
}


EMBEDDING_MODEL = _env(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

OCR_LANG = _env("OCR_LANG", "eng")
MAX_UPLOAD_MB = _int("MAX_UPLOAD_MB", 25)

RESULTS_DIR = ROOT / "evaluation" / "results"