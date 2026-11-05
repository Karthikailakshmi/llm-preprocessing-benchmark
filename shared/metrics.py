"""Timing helpers, duration formatting and the CSV result logger."""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import streamlit as st

from shared.config import RESULTS_DIR


COLUMNS = [
    "timestamp_utc",
    "username",
    "run_id",
    "variant",
    "case_id",
    "mode",
    "provider",
    "model_id",
    "question_original",
    "question_processed",
    "answer",
    "total_ms",
    "preprocessing_ms",
    "retrieval_ms",
    "api_ms",
    "rate_limit_wait_ms",
    "input_tokens",
    "output_tokens",
    "token_source",
    "success",
    "error_type",
    "error_message",
    "retrieved_sources",
    "embedding_backend",
    "techniques_enabled",
    "preprocessing_step_ms",
]


def format_duration(ms: Optional[float]) -> str:
    """<1 s -> ms, <1 min -> seconds, otherwise minutes. Raw values stay in ms."""
    if ms is None:
        return "n/a"
    if ms < 1000:
        return f"{ms:.0f} ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f} s"
    return f"{ms / 60_000:.2f} min"


@contextmanager
def timed(store: dict, key: str):
    """Add the elapsed milliseconds of the block to store[key]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (
            time.perf_counter() - start
        ) * 1000.0


def _current_username() -> str:
    """Get the logged-in username from Streamlit session state."""
    try:
        user = st.session_state.get("user")
        if isinstance(user, dict):
            return str(user.get("username") or "")
    except Exception:
        # Supports using the logger outside an active Streamlit session.
        pass

    return ""


def _migrate_csv_schema(path: Path) -> None:
    """Rewrite a results CSV so every row matches COLUMNS.

    Older files were written without the 'username' column, and newer rows
    (26 fields) were appended under the old 25-field header, which breaks
    pandas. Rows are realigned by length: 25 fields = old layout, 26 = new.
    """
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))

    if not rows:
        return

    header, body = rows[0], rows[1:]
    if header == COLUMNS and all(len(r) == len(COLUMNS) for r in body):
        return  # already clean

    old_columns = [c for c in COLUMNS if c != "username"]
    fixed = []
    for r in body:
        if not r:
            continue
        if len(r) == len(COLUMNS):
            record = dict(zip(COLUMNS, r))
        elif len(r) == len(old_columns):
            record = dict(zip(old_columns, r))
            record["username"] = ""
        else:
            continue  # unrecoverable row
        fixed.append(record)

    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(fixed)
    tmp.replace(path)


class ResultLogger:
    """Append one row per request (successful or failed) to a CSV file."""

    def __init__(
        self,
        variant: str,
        directory: Optional[Path] = None,
    ):
        self.variant = variant
        self.dir = Path(directory) if directory else Path(RESULTS_DIR)
        self.path = self.dir / f"{variant}_interactive.csv"

    def log(self, **row) -> None:
        """Save one request, including the current username when available."""
        self.dir.mkdir(parents=True, exist_ok=True)

        row.setdefault("variant", self.variant)

        if not row.get("username"):
            row["username"] = _current_username()

        _migrate_csv_schema(self.path)

        new_file = (
            not self.path.exists()
            or self.path.stat().st_size == 0
        )

        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=COLUMNS,
                extrasaction="ignore",
            )

            if new_file:
                writer.writeheader()

            writer.writerow({
                column: (
                    ""
                    if row.get(column) is None
                    else row.get(column)
                )
                for column in COLUMNS
            })

    def read_bytes(self) -> Optional[bytes]:
        """Return the CSV file contents for download, if available."""
        return self.path.read_bytes() if self.path.exists() else None

    def read_rows(
        self,
        username: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Read saved results, optionally filtering by username."""
        if not self.path.exists():
            return []

        with self.path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as fh:
            rows = list(csv.DictReader(fh))

        if username is not None:
            rows = [
                row
                for row in rows
                if (row.get("username") or "").casefold()
                == username.casefold()
            ]

        return rows