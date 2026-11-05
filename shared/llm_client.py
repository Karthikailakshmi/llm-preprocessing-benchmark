"""Modular LLM client. Every provider is reached through an OpenAI-compatible
endpoint, selected by an explicit provider key (no random router).

- No automatic retries (max_retries=0); the UI offers a manual Retry button.
- Client-side pacing keeps free-tier rate limits from causing avoidable 429s.
  Time spent waiting for pacing is reported separately (wait_ms) and is NOT
  counted as API duration.
- Errors become structured results; API keys are scrubbed from messages.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import openai
from openai import OpenAI

from shared.config import GENERATION, PROVIDERS, ProviderConfig


@dataclass
class LLMResult:
    ok: bool
    text: str
    provider: str
    model: str
    api_ms: Optional[float] = None
    wait_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error_type: str = ""
    error_message: str = ""


_clients: dict[str, OpenAI] = {}
_last_call: dict[str, float] = {}
_lock = threading.Lock()


def _client(cfg: ProviderConfig) -> OpenAI:
    if cfg.key not in _clients:
        _clients[cfg.key] = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=GENERATION["timeout_s"],
            max_retries=0,
        )
    return _clients[cfg.key]


def _pace(cfg: ProviderConfig) -> float:
    """Sleep just long enough to respect the provider's request rate."""
    interval = 60.0 / max(cfg.rpm, 1) * 1.05
    with _lock:
        now = time.monotonic()
        last = _last_call.get(cfg.key)
        wait = max(0.0, last + interval - now) if last is not None else 0.0
        _last_call[cfg.key] = now + wait
    if wait > 0:
        time.sleep(wait)
    return wait * 1000.0


def _scrub(message: str, secret: str) -> str:
    if secret:
        message = message.replace(secret, "***")
    return message[:500]


def _classify(exc: Exception) -> str:
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return "auth"
    if isinstance(exc, openai.RateLimitError):
        return "rate_limit"
    if isinstance(exc, openai.APITimeoutError):
        return "timeout"
    if isinstance(exc, openai.APIConnectionError):
        return "network"
    if isinstance(exc, openai.NotFoundError):
        return "model_not_found"
    if isinstance(exc, openai.BadRequestError):
        return "bad_request"
    if isinstance(exc, openai.APIStatusError):
        return "payment_required" if exc.status_code == 402 else f"http_{exc.status_code}"
    return "unknown"


def _fail(cfg: Optional[ProviderConfig], provider: str, kind: str, msg: str, **kw) -> LLMResult:
    return LLMResult(False, "", provider, cfg.model_id if cfg else "", error_type=kind,
                     error_message=msg, **kw)


def chat(provider_key: str, messages: list[dict], temperature: Optional[float] = None,
         max_tokens: Optional[int] = None) -> LLMResult:
    cfg = PROVIDERS.get(provider_key)
    if cfg is None:
        return _fail(None, provider_key, "config", f"Unknown provider '{provider_key}'.")
    if not cfg.has_key:
        return _fail(cfg, provider_key, "auth", f"{cfg.api_key_env} is not set in .env.")
    if not cfg.model_id:
        return _fail(cfg, provider_key, "config",
                     f"No model ID set. Add {provider_key.upper()}_MODEL=<model id> to .env.")

    wait_ms = _pace(cfg)
    start = time.perf_counter()
    try:
        resp = _client(cfg).chat.completions.create(
            model=cfg.model_id,
            messages=messages,
            temperature=GENERATION["temperature"] if temperature is None else temperature,
            **{cfg.token_param: max_tokens or GENERATION["max_tokens"]},
        )
        api_ms = (time.perf_counter() - start) * 1000.0
        text = ""
        if getattr(resp, "choices", None):
            text = (resp.choices[0].message.content or "").strip()
        usage = getattr(resp, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", None) if usage else None
        out_tok = getattr(usage, "completion_tokens", None) if usage else None
        if not text:
            return LLMResult(False, "", cfg.key, cfg.model_id, api_ms, wait_ms, in_tok, out_tok,
                             "invalid_response",
                             "The model returned an empty answer (it may have used its whole "
                             "token budget on reasoning). Try again or raise GEN_MAX_TOKENS.")
        return LLMResult(True, text, cfg.key, cfg.model_id, api_ms, wait_ms, in_tok, out_tok)
    except Exception as exc:  # noqa: BLE001 - converted to a structured result
        api_ms = (time.perf_counter() - start) * 1000.0
        return LLMResult(False, "", cfg.key, cfg.model_id, api_ms, wait_ms, None, None,
                         _classify(exc), _scrub(str(exc), cfg.api_key))
