
"""Streamlit interface used by BOTH chatbots, so the layouts are identical.
Only the theme and the pipeline (baseline vs preprocessing) differ.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import streamlit as st

from shared.config import PROVIDERS, RAG, ROOT
from shared.document_loader import SUPPORTED_EXTENSIONS, load_document, ocr_available
from shared.llm_client import chat
from shared.metrics import ResultLogger, format_duration
from shared.prompts import DOC_MODE, GENERAL_MODE
from shared.retrieval import VectorIndex, build_chunks


# --------------------------------------------------
# Persistent chat history configuration
# --------------------------------------------------

CHAT_HISTORY_DIR = ROOT / "data" / "chat_history"
MAX_HISTORY_MESSAGES = 500


def _current_username() -> str | None:
    """Return the logged-in username from the existing authentication state."""
    user = st.session_state.get("user")

    if isinstance(user, dict):
        username = user.get("username")
        if username:
            return str(username).strip()

    return None


def _user_hash() -> str:
    """Create a filesystem-safe identifier for the logged-in user."""
    username = _current_username()

    if not username:
        return "anonymous"

    return hashlib.sha256(
        username.casefold().encode("utf-8")
    ).hexdigest()


def _history_file(variant: str) -> Path | None:
    """Get the separate history file for a user and chatbot variant."""
    if not _current_username():
        return None

    safe_variant = "".join(
        c for c in variant if c.isalnum() or c in ("-", "_")
    )

    return CHAT_HISTORY_DIR / f"{_user_hash()}_{safe_variant}.json"


def _load_chat_history(variant: str) -> list[dict]:
    """Load saved messages for the current user and chatbot."""
    path = _history_file(variant)

    if path is None or not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            return []

        return [
            {
                "role": message["role"],
                "content": message["content"],
                **(
                    {"error": message["error"]}
                    if "error" in message else {}
                ),
                **(
                    {"meta": message["meta"]}
                    if isinstance(message.get("meta"), dict) else {}
                ),
            }
            for message in data
            if isinstance(message, dict)
            and message.get("role") in ("user", "assistant")
            and isinstance(message.get("content"), str)
        ][-MAX_HISTORY_MESSAGES:]

    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        st.warning(f"Could not load saved chat history: {exc}")
        return []


def _save_chat_history(variant: str, messages: list[dict]) -> None:
    """Save chat messages without exposing one user's file to another user."""
    path = _history_file(variant)

    # Do not save conversations without an authenticated user.
    if path is None:
        return

    try:
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        # Store only the fields required to restore the chat display.
        messages_to_save = []

        for message in messages[-MAX_HISTORY_MESSAGES:]:
            item = {
                "role": message.get("role"),
                "content": message.get("content", ""),
            }

            if "error" in message:
                item["error"] = message["error"]

            if isinstance(message.get("meta"), dict):
                item["meta"] = message["meta"]

            messages_to_save.append(item)

        temp_path = path.with_suffix(".tmp")

        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(
                messages_to_save,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temp_path, path)

    except (OSError, TypeError, ValueError) as exc:
        st.warning(f"Could not save chat history: {exc}")


def _clear_saved_history(variant: str) -> None:
    """Delete saved history for the current user and chatbot only."""
    path = _history_file(variant)

    if path is None:
        return

    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        st.error(f"Could not clear saved chat history: {exc}")


_CSS = """
<style>
:root { --p: __P__; --dark: __DARK__; --soft: __SOFT__; --line: __LINE__; }
[data-testid="stSidebar"] { background: var(--soft); border-right: 1px solid var(--line); }
.app-head { background: var(--p); color: #fff; padding: 1.1rem 1.4rem; border-radius: 12px;
            margin-bottom: 1rem; }
.app-head h1 { color: #fff; font-size: 1.5rem; margin: 0 0 .2rem 0; padding: 0; }
.app-head p { margin: 0; opacity: .92; font-size: .95rem; }
div.stButton > button, div.stDownloadButton > button {
    background: var(--p); color: #fff; border: 0; border-radius: 8px; }
div.stButton > button:hover, div.stDownloadButton > button:hover { background: var(--dark); color: #fff; }
[data-testid="stChatMessage"] { border: 1px solid var(--line); border-radius: 12px; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: var(--soft); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--dark); }
</style>
"""


def _inject_theme(t: dict) -> None:
    css = _CSS
    for k, v in (
        ("__P__", t["primary"]),
        ("__DARK__", t["dark"]),
        ("__SOFT__", t["soft"]),
        ("__LINE__", t["line"]),
    ):
        css = css.replace(k, v)
    st.markdown(css, unsafe_allow_html=True)


def _init_state(variant: str) -> None:
    """Initialize session state and restore the current user's chat."""
    ss = st.session_state

    # Separate in-memory messages for each user and chatbot.
    messages_key = f"messages_{_user_hash()}_{variant}"

    if messages_key not in ss:
        ss[messages_key] = _load_chat_history(variant)

    # Existing functions use the original 'messages' state key.
    ss["messages"] = ss[messages_key]

    ss.setdefault("index", None)
    ss.setdefault("index_meta", {})
    ss.setdefault("doc_status", [])
    ss.setdefault("retry", None)
    ss.setdefault(
        "run_id",
        datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
    )


def _ingest(files, pipeline) -> None:
    """Document ingestion: parse -> clean (common) -> chunk -> embed -> index."""
    status, segments = [], []

    for f in files:
        res = load_document(f.name, f.getvalue())
        status.append(
            {
                "name": f.name,
                "ok": res.ok,
                "error": res.error,
                "warnings": res.warnings,
                "segments": len(res.segments),
            }
        )
        if res.ok:
            segments += res.segments

    st.session_state.doc_status = status
    st.session_state.index, st.session_state.index_meta = None, {}

    if not segments:
        return

    chunks = build_chunks(
        segments,
        RAG["chunk_size"],
        RAG["chunk_overlap"],
        pipeline.split,
    )

    index = VectorIndex()
    index.build(chunks)

    st.session_state.index = index
    st.session_state.index_meta = {
        "chunks": len(chunks),
        "backend": index.backend,
        "signature": pipeline.ingestion_signature(),
        "files": [s["name"] for s in status if s["ok"]],
    }


def _history() -> list[dict]:
    """Last three successful exchanges for the current conversation."""
    out = []
    msgs = st.session_state.messages

    for i in range(0, len(msgs) - 1, 2):
        u, a = msgs[i], msgs[i + 1]

        if not a.get("error"):
            out += [
                {"role": "user", "content": u["content"]},
                {"role": "assistant", "content": a["content"]},
            ]

    return out[-6:]


def _caption(meta: dict, show_tokens: bool) -> str:
    bits = [f"Total {format_duration(meta['total_ms'])}"]

    if meta.get("api_ms") is not None:
        bits.append(f"model {format_duration(meta['api_ms'])}")

    if meta.get("retrieval_ms") is not None:
        bits.append(f"retrieval {format_duration(meta['retrieval_ms'])}")

    if meta.get("pre_ms"):
        bits.append(f"preprocessing {format_duration(meta['pre_ms'])}")

    if meta.get("wait_ms", 0) > 50:
        bits.append(f"rate-limit wait {format_duration(meta['wait_ms'])}")

    if show_tokens:
        tin = meta.get("in_tok")
        tout = meta.get("out_tok")
        bits.append(
            f"tokens in/out {tin if tin is not None else 'n/a'}/"
            f"{tout if tout is not None else 'n/a'}"
        )

    bits.append(f"{meta['provider']} · {meta['model']}")
    return " | ".join(bits)


def _render(
    msg: dict,
    i: int,
    last: bool,
    show_tokens: bool,
    show_details: bool,
) -> None:
    with st.chat_message(msg["role"]):
        if msg.get("error"):
            st.error(msg["content"])
        else:
            st.markdown(msg["content"])

        meta = msg.get("meta")
        if not meta:
            return

        # Older saved messages may not have all metadata fields.
        if "total_ms" in meta and "provider" in meta and "model" in meta:
            st.caption(_caption(meta, show_tokens))

        for warning in meta.get("warnings", []):
            st.warning(warning)

        if meta.get("sources"):
            with st.expander(f"Sources ({len(meta['sources'])})"):
                for n, source in enumerate(meta["sources"], 1):
                    st.markdown(
                        f"**[{n}] {source['ref']}** "
                        f"(similarity {source['score']:.2f})"
                    )
                    text = source["text"]
                    st.text(text[:400] + ("…" if len(text) > 400 else ""))

        if (
            show_details
            and meta.get("step_ms") is not None
            and meta.get("techniques")
        ):
            with st.expander("Preprocessing details"):
                st.markdown("**Original query**")
                st.code(meta.get("original", ""), language=None)

                st.markdown("**Processed query sent to the model**")
                st.code(meta.get("processed", ""), language=None)

                st.json(
                    {
                        "step_ms": {
                            k: round(v, 3)
                            for k, v in meta["step_ms"].items()
                        },
                        "details": meta.get("details", {}),
                    }
                )

        if msg.get("error") and last and st.button(
            "Retry", key=f"retry_{i}"
        ):
            st.session_state.messages = st.session_state.messages[:-2]
            st.session_state.retry = meta.get("original", "")
            st.rerun()


def _answer(question, pipeline, provider_key, mode, top_k, logger) -> dict:
    cfg = PROVIDERS[provider_key]
    t_start = time.perf_counter()
    step_ms: dict = {}

    base = {
        "provider": cfg.key,
        "model": cfg.model_id,
        "original": question,
        "techniques": pipeline.techniques(),
    }

    def finish(
        res,
        prep,
        built,
        hits,
        retrieval_ms,
        error_type="",
        error_message="",
        text="",
    ):
        total_ms = (time.perf_counter() - t_start) * 1000
        pre_ms = sum(step_ms.values())
        used = built.used_hits if built else hits

        sources = [
            {"ref": c.ref, "score": score, "text": c.text}
            for c, score in used
        ]

        warnings = list(prep.warnings) + (
            built.warnings if built else []
        )
        ok = res is not None and res.ok

        meta = {
            **base,
            "processed": prep.processed,
            "total_ms": total_ms,
            "pre_ms": pre_ms,
            "api_ms": res.api_ms if res else None,
            "retrieval_ms": retrieval_ms,
            "wait_ms": res.wait_ms if res else 0.0,
            "in_tok": res.input_tokens if res else None,
            "out_tok": res.output_tokens if res else None,
            "sources": sources,
            "step_ms": step_ms,
            "details": prep.details,
            "warnings": warnings,
        }

        index = st.session_state.index

        logger.log(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            run_id=st.session_state.run_id,
            variant=pipeline.name,
            mode=mode,
            provider=cfg.key,
            model_id=cfg.model_id,
            question_original=question,
            question_processed=prep.processed,
            answer=text if ok else "",
            total_ms=round(total_ms, 3),
            preprocessing_ms=round(pre_ms, 3),
            retrieval_ms=(
                None if retrieval_ms is None else round(retrieval_ms, 3)
            ),
            api_ms=(
                None if not res or res.api_ms is None
                else round(res.api_ms, 3)
            ),
            rate_limit_wait_ms=round(res.wait_ms, 3) if res else None,
            input_tokens=res.input_tokens if res else None,
            output_tokens=res.output_tokens if res else None,
            token_source=(
                "provider"
                if res and res.input_tokens is not None
                else ""
            ),
            success=ok,
            error_type=error_type,
            error_message=error_message,
            retrieved_sources="; ".join(
                source["ref"] for source in sources
            ),
            embedding_backend=(
                index.backend if index and mode == DOC_MODE else ""
            ),
            techniques_enabled=json.dumps(pipeline.techniques()),
            preprocessing_step_ms=json.dumps(
                {k: round(v, 3) for k, v in step_ms.items()}
            ),
        )

        return {
            "role": "assistant",
            "content": text if ok else error_message or "Request failed.",
            "error": not ok,
            "meta": meta,
        }

    prep = pipeline.process_query(question, cfg.max_input_tokens)
    step_ms.update(prep.step_ms)

    if prep.blocked:
        return finish(
            None,
            prep,
            None,
            [],
            None,
            "blocked_by_filter",
            prep.reason,
        )

    hits, retrieval_ms = [], None

    if mode == DOC_MODE:
        index = st.session_state.index

        if index is None:
            return finish(
                None,
                prep,
                None,
                [],
                None,
                "no_documents",
                "No processed documents. Upload files and choose Process documents.",
            )

        t = time.perf_counter()
        hits = index.search(prep.processed, top_k)
        retrieval_ms = (time.perf_counter() - t) * 1000

    built = pipeline.build_request(
        prep, hits, _history(), mode, cfg.max_input_tokens
    )

    for key, value in built.step_ms.items():
        step_ms[key] = step_ms.get(key, 0.0) + value

    res = chat(provider_key, built.messages)

    if res.ok:
        return finish(
            res, prep, built, hits, retrieval_ms, text=res.text
        )

    return finish(
        res,
        prep,
        built,
        hits,
        retrieval_ms,
        res.error_type,
        f"{res.error_type}: {res.error_message}",
    )


def render_app(
    variant: str,
    title: str,
    tagline: str,
    theme: dict,
    build_pipeline: Callable[[], object],
) -> None:
    st.set_page_config(
        page_title=title,
        page_icon=theme["icon"],
        layout="wide",
    )

    _inject_theme(theme)
    _init_state(variant)

    logger = ResultLogger(variant)

    with st.sidebar:
        st.subheader("Model")

        keys = list(PROVIDERS)

        provider_key = st.selectbox(
            "Provider",
            keys,
            label_visibility="collapsed",
            format_func=lambda k: (
                f"{PROVIDERS[k].label}"
                + (
                    ""
                    if PROVIDERS[k].configured
                    else "  (not set up)"
                )
            ),
        )

        cfg = PROVIDERS[provider_key]
        st.caption(f"Model ID: `{cfg.model_id or 'not set'}`")

        if not cfg.has_key:
            st.warning(f"Add {cfg.api_key_env} to your .env file.")
        elif not cfg.model_id:
            st.warning(
                f"Add {provider_key.upper()}_MODEL to your .env file."
            )

        if cfg.notes:
            st.caption(cfg.notes)

        st.subheader("Mode")

        if st.session_state.pop("_goto_docs", False):
            st.session_state["mode"] = DOC_MODE

        mode = st.radio(
            "Mode",
            [GENERAL_MODE, DOC_MODE],
            key="mode",
            label_visibility="collapsed",
        )

        pipeline = build_pipeline()

        st.subheader("Documents")

        files = st.file_uploader(
            "Upload files",
            accept_multiple_files=True,
            type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
            label_visibility="collapsed",
        )

        if not ocr_available():
            st.caption(
                "OCR is not available: scanned PDFs and images cannot be read."
            )

        top_k = st.slider("Passages to retrieve", 1, 8, RAG["top_k"])

        if st.button("Process documents", disabled=not files):
            with st.spinner("Reading, chunking and indexing…"):
                try:
                    t0 = time.perf_counter()
                    _ingest(files, pipeline)

                    if st.session_state.index_meta:
                        st.session_state.index_meta["seconds"] = (
                            time.perf_counter() - t0
                        )
                        st.session_state["_goto_docs"] = True

                except Exception as exc:  # noqa: BLE001
                    st.error(
                        f"Processing failed: {type(exc).__name__}: {exc}"
                    )

            if st.session_state.get("_goto_docs"):
                st.rerun()

        for status in st.session_state.doc_status:
            if status["ok"]:
                st.success(
                    f"{status['name']}: {status['segments']} section(s) read"
                )
            else:
                st.error(f"{status['name']}: {status['error']}")

            for warning in status["warnings"]:
                st.warning(f"{status['name']}: {warning}")

        meta = st.session_state.index_meta

        if meta:
            chunks_count = meta.get("chunks", 0)
            backend = meta.get("backend", "unknown")
            signature = meta.get("signature", "default")

            st.caption(
                f"{chunks_count} chunks · {backend} · {signature} chunking"
            )

            if backend == "tfidf-fallback":
                st.warning(
                    "The embedding model could not load, so keyword "
                    "(TF-IDF) retrieval is used."
                )

            if signature != pipeline.ingestion_signature():
                st.warning(
                    "Chunking settings changed. Process the documents "
                    "again to apply them."
                )

        st.subheader("Display")

        show_tokens = st.checkbox("Show token usage", value=True)
        show_details = st.checkbox(
            "Show preprocessing details", value=True
        )

        c1, c2 = st.columns(2)

        if c1.button("Clear chat"):
            st.session_state.messages = []

            messages_key = f"messages_{_user_hash()}_{variant}"
            st.session_state[messages_key] = []

            _clear_saved_history(variant)
            st.rerun()

        data = logger.read_bytes()

        c2.download_button(
            "Results CSV",
            data or b"",
            file_name=logger.path.name,
            mime="text/csv",
            disabled=data is None,
        )

    # --------------------------------------------------
    # Main chatbot interface
    # --------------------------------------------------

    st.markdown(
        f'<div class="app-head"><h1>{title}</h1>'
        f'<p>{tagline}</p></div>',
        unsafe_allow_html=True,
    )

    if mode == DOC_MODE and st.session_state.index is None:
        st.warning(
            "Document mode is on, but no documents are processed yet. "
            "Upload a file and click Process documents."
        )

    elif mode == DOC_MODE:
        st.success(
            "Answering from your documents: "
            + ", ".join(
                st.session_state.index_meta.get("files", [])
            )
        )

    elif st.session_state.index is not None:
        st.warning(
            "Your documents are processed, but you are in General Chat, "
            "so the model cannot see them. Switch Mode to "
            "'Ask Uploaded Documents'."
        )

    if not st.session_state.messages:
        st.info(
            "Ask anything in General Chat, or switch to Ask Uploaded "
            "Documents after processing files."
        )

    msgs = st.session_state.messages

    for i, message in enumerate(msgs):
        _render(
            message,
            i,
            i == len(msgs) - 1,
            show_tokens,
            show_details,
        )

    prompt = (
        st.chat_input("Type your question")
        or st.session_state.pop("retry", None)
    )

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Waiting for {cfg.label}…"):
                reply = _answer(
                    prompt,
                    pipeline,
                    provider_key,
                    mode,
                    top_k,
                    logger,
                )

        st.session_state.messages.extend([
            {"role": "user", "content": prompt},
            reply,
        ])

        # Keep the current user's in-memory and persistent histories aligned.
        messages_key = f"messages_{_user_hash()}_{variant}"
        st.session_state[messages_key] = st.session_state.messages

        _save_chat_history(variant, st.session_state.messages)

        st.rerun()
