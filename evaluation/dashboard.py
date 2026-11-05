"""Evaluation dashboard: Baseline vs Preprocessed chatbot results.

Five interactive Plotly charts, each a different chart type and palette:
  1. Total response latency      - grouped bar   (blue / purple)
  2. Answer quality              - grouped bar   (teal / green / amber)
  3. Token usage                 - stacked bar   (orange / cyan)
  4. Reliability                 - donut         (green / red)
  5. Preprocessing effectiveness - scatter       (one colour per provider)

Only saved CSV records are read. Nothing is re-run and no value is invented:
missing numbers stay missing and are never treated as zero.
"""

from __future__ import annotations

import csv
import html
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from shared.config import RESULTS_DIR

try:
    from shared.config import PROVIDERS
except ImportError:  # pragma: no cover
    PROVIDERS = {}

try:
    from shared.metrics import COLUMNS as LOG_COLUMNS
except ImportError:  # pragma: no cover
    LOG_COLUMNS = []


st.set_page_config(
    page_title="Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

HARD_LIMIT_MS = 1500.0
FAST_LIMIT_MS = 1000.0

VARIANT_LABELS = {"baseline": "Baseline", "preprocessed": "Preprocessed"}
VARIANT_COLORS = {"Baseline": "#2563EB", "Preprocessed": "#7C3AED"}
VARIANT_PATTERNS = {"Baseline": "", "Preprocessed": "/"}

QUALITY_DIMENSIONS = {
    "relevance": "Relevance",
    "correctness": "Factual correctness",
    "faithfulness": "Faithfulness",
}
QUALITY_COLORS = {
    "Relevance": "#0D9488",
    "Factual correctness": "#22C55E",
    "Faithfulness": "#F59E0B",
}

# Separate palettes for the three latency charts in section 1.
LATENCY_AVG_COLORS = {"Baseline": "#0EA5E9", "Preprocessed": "#E11D48"}
LATENCY_TOTAL_COLORS = {"Baseline": "#F59E0B", "Preprocessed": "#4F46E5"}
ONE_REQUEST_COLORS = {
    "Preprocessing": "#D946EF",
    "Document retrieval": "#84CC16",
    "Model response": "#0F766E",
    "Rate-limit wait": "#78716C",
    "Other overhead": "#CBD5E1",
}

INPUT_COLOR = "#F97316"
OUTPUT_COLOR = "#06B6D4"
SUCCESS_COLOR = "#16A34A"
FAILURE_COLOR = "#DC2626"

PROVIDER_PALETTE = [
    "#0EA5E9", "#F43F5E", "#10B981", "#F59E0B", "#8B5CF6", "#64748B",
]

RATINGS_PATH = Path(RESULTS_DIR) / "quality_ratings.csv"
RATING_COLUMNS = [
    "username", "variant", "timestamp_utc",
    "relevance", "correctness", "faithfulness", "rated_at",
]

NUMERIC_COLUMNS = [
    "total_ms", "preprocessing_ms", "retrieval_ms", "api_ms",
    "rate_limit_wait_ms", "input_tokens", "output_tokens",
]
EXPECTED_COLUMNS = [
    "provider", "model_id", "mode", "question_original",
    "question_processed", "answer", "case_id", "error_type",
] + NUMERIC_COLUMNS

# Streamlit 1.50+ uses width="stretch"; older versions use use_container_width.
_VERSION = tuple(
    int(part) for part in st.__version__.split(".")[:2] if part.isdigit()
)
STRETCH = (
    {"width": "stretch"} if _VERSION >= (1, 50)
    else {"use_container_width": True}
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .dash-sub {opacity: .75; margin-top: -.6rem; margin-bottom: 1rem;}
    .kpi-card {
        border: 1px solid rgba(128,128,128,.28);
        border-left: 6px solid var(--accent, #2563EB);
        border-radius: 12px;
        padding: 14px 16px;
        background: rgba(128,128,128,.07);
        height: 100%;
    }
    .kpi-label {font-size: .78rem; text-transform: uppercase;
                letter-spacing: .05em; opacity: .75;}
    .kpi-value {font-size: 1.7rem; font-weight: 700; line-height: 1.25;}
    .kpi-sub {font-size: .78rem; opacity: .8; margin-top: 2px;}
    .section-bar {
        border-left: 6px solid var(--accent, #2563EB);
        padding-left: 12px; margin: 2rem 0 .3rem 0;
    }
    .section-bar h3 {margin: 0; padding: 0;}
    .section-bar p {margin: 2px 0 0 0; opacity: .75; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Evaluation Dashboard")
st.markdown(
    '<div class="dash-sub">Baseline vs Preprocessed chatbot: latency, '
    "answer quality, tokens, reliability and preprocessing effectiveness. "
    "Reads saved results only.</div>",
    unsafe_allow_html=True,
)


def section(number: int, title: str, subtitle: str, accent: str) -> None:
    """Render a coloured section heading."""
    st.markdown(
        f'<div class="section-bar" style="--accent:{accent}">'
        f"<h3>{number}. {html.escape(title)}</h3>"
        f"<p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str, accent: str) -> str:
    """Return the HTML of one KPI card."""
    return (
        f'<div class="kpi-card" style="--accent:{accent}">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div></div>'
    )


# ---------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------

def format_ms(value) -> str:
    """ms below 1 s, s below 1 min, otherwise min. Input is in ms."""
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if value < 1000:
        return f"{value:.0f} ms"
    if value < 60_000:
        return f"{value / 1000:.2f} s"
    return f"{value / 60_000:.2f} min"


def latency_axis(max_ms: float) -> dict:
    """Axis settings whose tick labels use the same ms/s/min rule."""
    top = max(float(max_ms) * 1.2, 1.0)
    ticks = np.linspace(0, top, 6)
    return {
        "tickmode": "array",
        "tickvals": ticks.tolist(),
        "ticktext": [format_ms(t) for t in ticks],
        "range": [0, top],
    }


def format_pct(value) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value:.1f}%"


def format_num(value, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def provider_label(key) -> str:
    config = PROVIDERS.get(key)
    return getattr(config, "label", None) or str(key)


def provider_order() -> list[str]:
    return [provider_label(key) for key in PROVIDERS]


def ordered_providers(labels) -> list[str]:
    present = set(labels)
    known = [p for p in provider_order() if p in present]
    return known + sorted(present - set(known))


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def _read_result_file(path: Path) -> pd.DataFrame:
    """Read one results CSV, realigning rows written by older versions.

    Files created before the 'username' column existed may contain 25-field
    and 26-field rows under one header. Rows are matched to LOG_COLUMNS by
    length so such files still load.
    """
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    if len(rows) < 2:
        return pd.DataFrame()

    full = list(LOG_COLUMNS)
    old = [c for c in full if c != "username"]
    header = rows[0]

    if full and header in (full, old):
        records = []
        for row in rows[1:]:
            if len(row) == len(full):
                records.append(dict(zip(full, row)))
            elif len(row) == len(old):
                record = dict(zip(old, row))
                record["username"] = ""
                records.append(record)
        return pd.DataFrame(records, columns=full)

    return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")


@st.cache_data(show_spinner=False)
def _load_results(files: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    """Load every *_interactive.csv. 'files' holds (path, mtime) for caching."""
    frames = []
    for path_text, _mtime in files:
        path = Path(path_text)
        try:
            frame = _read_result_file(path)
        except (OSError, UnicodeDecodeError, csv.Error, pd.errors.ParserError):
            continue
        if frame.empty:
            continue
        fallback = path.name.replace("_interactive.csv", "")
        if "variant" not in frame.columns:
            frame["variant"] = fallback
        else:
            frame["variant"] = (
                frame["variant"].replace("", np.nan).fillna(fallback)
            )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    data = pd.concat(frames, ignore_index=True, sort=False)

    for column in EXPECTED_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Success is three-valued: success, failure, or unknown. A missing or
    # unreadable status is never counted as a success.
    if "success" in data.columns:
        text = data["success"].astype(str).str.strip().str.lower()
    else:
        text = pd.Series("", index=data.index)
    data["ok"] = text.isin(["true", "1", "yes", "success"]).to_numpy()
    data["fail"] = text.isin(["false", "0", "no", "failed"]).to_numpy()
    data["known"] = data["ok"] | data["fail"]

    data["timestamp"] = (
        pd.to_datetime(data["timestamp_utc"], errors="coerce", utc=True)
        if "timestamp_utc" in data.columns else pd.NaT
    )
    return data


def load_results() -> pd.DataFrame:
    folder = Path(RESULTS_DIR)
    if not folder.exists():
        return pd.DataFrame()
    files = tuple(
        (str(p), p.stat().st_mtime)
        for p in sorted(folder.glob("*_interactive.csv"))
    )
    return _load_results(files) if files else pd.DataFrame()


def load_ratings(username: str) -> pd.DataFrame:
    """Human quality ratings saved by this dashboard (never auto-generated)."""
    empty = pd.DataFrame(columns=RATING_COLUMNS)
    if not RATINGS_PATH.exists():
        return empty
    try:
        saved = pd.read_csv(RATINGS_PATH, encoding="utf-8-sig")
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return empty
    if not set(RATING_COLUMNS).issubset(saved.columns):
        return empty
    saved = saved[
        saved["username"].fillna("").astype(str).str.casefold()
        == username.casefold()
    ].copy()
    for column in QUALITY_DIMENSIONS:
        saved[column] = pd.to_numeric(saved[column], errors="coerce")
    return saved


def save_rating(row: dict) -> None:
    """Add or replace one rating, keyed by user + variant + timestamp."""
    RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame(columns=RATING_COLUMNS)
    if RATINGS_PATH.exists():
        try:
            existing = pd.read_csv(RATINGS_PATH, encoding="utf-8-sig")
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            pass

    if not existing.empty:
        same = (
            (existing["username"].astype(str) == row["username"])
            & (existing["variant"].astype(str) == row["variant"])
            & (existing["timestamp_utc"].astype(str) == row["timestamp_utc"])
        )
        existing = existing[~same]

    updated = pd.concat(
        [existing, pd.DataFrame([row], columns=RATING_COLUMNS)],
        ignore_index=True,
    )
    updated.to_csv(RATINGS_PATH, index=False, encoding="utf-8")


def get_username() -> str | None:
    user = st.session_state.get("user")
    if isinstance(user, dict):
        username = user.get("username")
        return str(username) if username else None
    return None


# ---------------------------------------------------------
# Load data, enforce per-user access
# ---------------------------------------------------------

username = get_username()
if not username:
    st.warning("Please log in to view your saved evaluation results.")
    st.stop()

all_results = load_results()
if all_results.empty:
    st.info(
        "No saved results were found. Run your chatbot tests first; "
        "results will appear here when they have been logged."
    )
    st.stop()

if "username" not in all_results.columns:
    st.info("Saved results do not record usernames, so none can be shown.")
    st.stop()

# Users only ever see their own records.
results = all_results[
    all_results["username"].fillna("").astype(str).str.strip().str.casefold()
    == username.casefold()
].copy()

if results.empty:
    st.info(
        "No results are associated with your account yet. Older CSV "
        "records without a username cannot be assigned to your account."
    )
    st.stop()

results["variant"] = results["variant"].fillna("unknown").astype(str)
results["variant_label"] = results["variant"].map(
    lambda v: VARIANT_LABELS.get(v, v.title())
)
results["provider_label"] = results["provider"].map(
    lambda p: provider_label(p) if pd.notna(p) else "Unknown"
)


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.header("Dashboard Filters")

filtered = results.copy()

variant_options = [
    v for v in VARIANT_COLORS if v in set(filtered["variant_label"])
] + sorted(set(filtered["variant_label"]) - set(VARIANT_COLORS))
chosen = st.sidebar.multiselect(
    "Chatbot (Baseline / Preprocessed)", variant_options,
    default=variant_options,
)
filtered = filtered[filtered["variant_label"].isin(chosen)]


def _multiselect_filter(frame: pd.DataFrame, column: str, label: str):
    if frame[column].dropna().empty:
        return frame
    options = sorted(frame[column].dropna().astype(str).unique())
    picked = st.sidebar.multiselect(label, options, default=options)
    return frame[frame[column].astype(str).isin(picked)]


filtered = _multiselect_filter(filtered, "provider_label", "Provider")
filtered = _multiselect_filter(filtered, "model_id", "Model")
filtered = _multiselect_filter(filtered, "mode", "Chat mode")

if filtered["timestamp"].notna().any():
    low = filtered["timestamp"].min().date()
    high = filtered["timestamp"].max().date()
    picked_dates = st.sidebar.date_input(
        "Date range", value=(low, high), min_value=low, max_value=high
    )
    if isinstance(picked_dates, (tuple, list)):
        if len(picked_dates) == 2:
            start, end = picked_dates
        elif len(picked_dates) == 1:
            start = end = picked_dates[0]
        else:
            start, end = low, high
    else:
        start = end = picked_dates
    dates = filtered["timestamp"].dt.date
    filtered = filtered[(dates >= start) & (dates <= end)]

include_failed_latency = st.sidebar.checkbox(
    "Include failed requests in latency",
    value=False,
    help=(
        "Off (default): latency uses successful requests only, so timeouts "
        "and instant API-key errors do not distort the averages."
    ),
)

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

st.caption(f"Showing saved results for account: **{username}**")

successful = filtered[filtered["ok"]]
latency_rows = (filtered if include_failed_latency else successful).dropna(
    subset=["total_ms"]
)

ratings = load_ratings(username)
rated = pd.DataFrame()
if not ratings.empty:
    rated = filtered.merge(
        ratings.drop(columns=["rated_at"]),
        on=["username", "variant", "timestamp_utc"],
        how="inner",
    ).dropna(subset=list(QUALITY_DIMENSIONS), how="all")
if not rated.empty:
    rated["quality_overall"] = rated[list(QUALITY_DIMENSIONS)].mean(axis=1)


def per_variant(frame: pd.DataFrame, func) -> str:
    """Apply func to each variant's rows and join for a KPI sub-line."""
    parts = []
    for label in VARIANT_COLORS:
        part = frame[frame["variant_label"] == label]
        parts.append(f"{label}: {func(part) if not part.empty else 'N/A'}")
    return "  ·  ".join(parts)


# ---------------------------------------------------------
# KPI cards
# ---------------------------------------------------------

avg_latency = latency_rows["total_ms"].mean() if not latency_rows.empty else np.nan
known_count = int(filtered["known"].sum())
success_rate = (
    filtered["ok"].sum() / known_count * 100 if known_count else np.nan
)

token_rows = successful.dropna(subset=["input_tokens", "output_tokens"]).copy()
token_rows["total_tokens"] = (
    token_rows["input_tokens"] + token_rows["output_tokens"]
)
avg_tokens = token_rows["total_tokens"].mean() if not token_rows.empty else np.nan

pre_rows = filtered[
    (filtered["variant_label"] == "Preprocessed")
    & filtered["preprocessing_ms"].notna()
]
avg_pre = pre_rows["preprocessing_ms"].mean() if not pre_rows.empty else np.nan

quality_mean = rated["quality_overall"].mean() if not rated.empty else np.nan

cards = st.columns(5)
cards[0].markdown(
    kpi_card(
        "Avg total response latency", format_ms(avg_latency),
        per_variant(latency_rows, lambda d: format_ms(d["total_ms"].mean())),
        "#2563EB",
    ),
    unsafe_allow_html=True,
)
cards[1].markdown(
    kpi_card(
        "Answer quality (0-100)",
        format_num(quality_mean, 1) if pd.notna(quality_mean) else "Not rated",
        per_variant(rated, lambda d: format_num(d["quality_overall"].mean(), 1))
        if not rated.empty else "No ratings collected yet",
        "#0D9488",
    ),
    unsafe_allow_html=True,
)
cards[2].markdown(
    kpi_card(
        "Avg tokens / success", format_num(avg_tokens),
        per_variant(token_rows, lambda d: format_num(d["total_tokens"].mean()))
        if not token_rows.empty else "No token counts logged",
        "#F97316",
    ),
    unsafe_allow_html=True,
)
cards[3].markdown(
    kpi_card(
        "Success rate", format_pct(success_rate),
        per_variant(
            filtered,
            lambda d: format_pct(
                d["ok"].sum() / d["known"].sum() * 100
                if d["known"].sum() else np.nan
            ),
        ),
        "#16A34A",
    ),
    unsafe_allow_html=True,
)
cards[4].markdown(
    kpi_card(
        "Avg preprocessing time", format_ms(avg_pre),
        f"{len(pre_rows):,} preprocessed requests"
        if not pre_rows.empty else "No preprocessed rows in selection",
        "#7C3AED",
    ),
    unsafe_allow_html=True,
)


# =========================================================
# Chart 1: Total response latency (grouped bar)
# =========================================================

section(
    1, "Response Latency",
    "How fast each model answers (average), how much time all of its "
    "responses took (total), and the total time of one request with every "
    "step included, for Baseline and Preprocessed.",
    "#2563EB",
)

def mark_no_data(fig, present_labels):
    """Keep every configured model on the x axis. A model without saved
    responses gets a 'no data' note instead of a bar (never a zero bar)."""
    all_models = ordered_providers(list(present_labels) + provider_order())
    no_data = [m for m in all_models if m not in set(present_labels)]
    if no_data:
        fig.add_trace(go.Bar(
            x=no_data, y=[None] * len(no_data),
            showlegend=False, hoverinfo="skip",
        ))
        for model_name in no_data:
            fig.add_annotation(
                x=model_name, y=0, text="no data", showarrow=False,
                yshift=14, font=dict(color="#64748B"),
            )
    fig.update_xaxes(categoryorder="array", categoryarray=all_models)
    return no_data


if latency_rows.empty:
    st.info("No latency measurements are available for this selection.")
else:
    stats = (
        latency_rows.groupby(["provider_label", "variant_label"])["total_ms"]
        .agg(
            avg="mean",
            total="sum",
            median="median",
            p95=lambda s: s.quantile(0.95),
            n="size",
            within_1s=lambda s: (s <= FAST_LIMIT_MS).mean() * 100,
            within_limit=lambda s: (s <= HARD_LIMIT_MS).mean() * 100,
        )
        .reset_index()
    )
    stats["avg_text"] = stats["avg"].map(format_ms)
    stats["total_text"] = stats["total"].map(format_ms)
    stats["median_text"] = stats["median"].map(format_ms)
    stats["p95_text"] = stats["p95"].map(format_ms)

    st.markdown("#### Average response latency (how fast each answer arrives)")
    fig = px.bar(
        stats,
        x="provider_label", y="avg", color="variant_label",
        barmode="group", text="avg_text",
        color_discrete_map=LATENCY_AVG_COLORS,
        pattern_shape="variant_label",
        pattern_shape_map=VARIANT_PATTERNS,
        category_orders={
            "provider_label": ordered_providers(stats["provider_label"]),
            "variant_label": list(VARIANT_COLORS),
        },
        custom_data=[
            "avg_text", "median_text", "p95_text", "n",
            "within_1s", "within_limit",
        ],
        labels={
            "provider_label": "Provider / model",
            "avg": "Average response latency",
            "variant_label": "Chatbot",
        },
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b> (%{fullData.name})<br>"
            "Average: %{customdata[0]}<br>"
            "Median: %{customdata[1]}<br>"
            "P95: %{customdata[2]}<br>"
            "Within 1 s: %{customdata[4]:.1f}%<br>"
            "Within 1.5 s limit: %{customdata[5]:.1f}%<br>"
            "Requests: %{customdata[3]}<extra></extra>"
        ),
    )
    fig.add_hline(
        y=HARD_LIMIT_MS, line_dash="dash", line_color="#DC2626",
        annotation_text="1.5 s hard limit", annotation_position="top left",
    )
    no_data = mark_no_data(fig, stats["provider_label"])
    fig.update_yaxes(
        **latency_axis(max(stats["avg"].max(), HARD_LIMIT_MS / 1.2)),
        title="Average response latency",
    )
    fig.update_layout(
        height=430, bargap=0.25, legend_title_text="Chatbot",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, key="chart_latency", **STRETCH)
    shown_variants = [
        v for v in VARIANT_COLORS if v in set(filtered["variant_label"])
    ]
    present_pairs = set(zip(stats["provider_label"], stats["variant_label"]))
    missing_pairs = [
        f"{model_name} ({variant_name})"
        for model_name in ordered_providers(
            list(stats["provider_label"]) + provider_order()
        )
        for variant_name in shown_variants
        if (model_name, variant_name) not in present_pairs
    ]
    if missing_pairs:
        st.caption(
            "No bar is drawn where there are no saved responses: "
            + ", ".join(missing_pairs)
            + ". Chat with these models in that chatbot to compare them."
        )

    # ---- Total latency: all responses added up ----
    st.markdown("#### Total latency (all responses added up)")
    fig = px.bar(
        stats,
        x="provider_label", y="total", color="variant_label",
        barmode="group", text="total_text",
        color_discrete_map=LATENCY_TOTAL_COLORS,
        pattern_shape="variant_label",
        pattern_shape_map=VARIANT_PATTERNS,
        category_orders={
            "provider_label": ordered_providers(stats["provider_label"]),
            "variant_label": list(VARIANT_COLORS),
        },
        custom_data=["total_text", "n", "avg_text"],
        labels={
            "provider_label": "Provider / model",
            "total": "Total latency",
            "variant_label": "Chatbot",
        },
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b> (%{fullData.name})<br>"
            "Total latency: %{customdata[0]}<br>"
            "Requests: %{customdata[1]}<br>"
            "Average per request: %{customdata[2]}<extra></extra>"
        ),
    )
    mark_no_data(fig, stats["provider_label"])
    fig.update_yaxes(
        **latency_axis(stats["total"].max()),
        title="Total latency (sum of all responses)",
    )
    fig.update_layout(
        height=430, bargap=0.25, legend_title_text="Chatbot",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, key="chart_latency_total", **STRETCH)
    st.caption(
        "Total latency adds up the response times of every saved request, "
        "so it grows with the number of requests. Check the request counts "
        "(hover, or the details table below) before comparing Baseline and "
        "Preprocessed; the average chart above is the fair speed comparison."
    )

    detail = pd.DataFrame({
        "Provider / model": stats["provider_label"],
        "Chatbot": stats["variant_label"],
        "Requests": stats["n"],
        "Average": stats["avg_text"],
        "Total": stats["total_text"],
        "Median": stats["median_text"],
        "P95": stats["p95_text"],
        "Within 1 s": stats["within_1s"].map(format_pct),
        "Within 1.5 s": stats["within_limit"].map(format_pct),
    })
    with st.expander("Latency details (average, total, median, P95, limits)"):
        st.dataframe(detail, hide_index=True, **STRETCH)
        st.caption(
            "Latency includes failed requests."
            if include_failed_latency
            else "Latency uses successful requests only."
        )

    # ---- Total latency for ONE request (every step included) ----
    st.markdown(
        "#### Total latency for one request "
        "(preprocessing + retrieval + model)"
    )
    part_columns = {
        "preprocessing_ms": "Preprocessing",
        "retrieval_ms": "Document retrieval",
        "api_ms": "Model response",
        "rate_limit_wait_ms": "Rate-limit wait",
        "other_ms": "Other overhead",
    }
    parts = latency_rows.copy()
    # A step a request did not use (for example retrieval in General Chat)
    # took no time, so it counts as 0 ms and the parts add up to the total.
    measured = list(part_columns)[:-1]
    for column in measured:
        parts[column] = parts[column].fillna(0)
    parts["other_ms"] = (
        parts["total_ms"] - parts[measured].sum(axis=1)
    ).clip(lower=0)

    one_request = (
        parts.groupby(["provider_label", "variant_label"])[
            list(part_columns) + ["total_ms"]
        ].mean().reset_index()
    )
    model_rank = {
        name: i for i, name in enumerate(
            ordered_providers(list(one_request["provider_label"])
                              + provider_order())
        )
    }
    variant_rank = {name: i for i, name in enumerate(VARIANT_COLORS)}
    one_request["_m"] = one_request["provider_label"].map(model_rank)
    one_request["_v"] = one_request["variant_label"].map(variant_rank).fillna(9)
    one_request = one_request.sort_values(["_m", "_v"]).drop(
        columns=["_m", "_v"]
    )
    # One plain text label per bar: model, chatbot mode and total.
    one_request["bar_label"] = (
        one_request["provider_label"].astype(str) + " · "
        + one_request["variant_label"].astype(str)
        + " (" + one_request["total_ms"].map(format_ms) + ")"
    )
    bar_order = one_request["bar_label"].tolist()

    one_long = one_request.melt(
        id_vars=["provider_label", "variant_label", "bar_label", "total_ms"],
        value_vars=list(part_columns),
        var_name="component", value_name="ms",
    )
    one_long["component"] = one_long["component"].map(part_columns)
    one_long["ms_text"] = one_long["ms"].map(format_ms)
    one_long["total_text"] = one_long["total_ms"].map(format_ms)

    st.caption(
        "How to read this chart: each bar is one model in one chatbot "
        "(Baseline or Preprocessed). The bar's length is the time one "
        "typical request takes, from sending the question to receiving the "
        "answer, and the label at the left ends with that total. The "
        "colours show where the time went: preprocessing, document "
        "retrieval, the model's own response, and waiting."
    )
    fig = px.bar(
        one_long, y="bar_label", x="ms",
        color="component", orientation="h", barmode="relative",
        color_discrete_map=ONE_REQUEST_COLORS,
        pattern_shape="component",
        pattern_shape_map={
            "Preprocessing": "x", "Document retrieval": "/",
            "Model response": "", "Rate-limit wait": ".",
            "Other overhead": "+",
        },
        category_orders={
            "component": list(part_columns.values()),
            "bar_label": bar_order,
        },
        custom_data=["provider_label", "variant_label", "ms_text",
                     "total_text"],
        labels={"ms": "Total latency of one request", "bar_label": "",
                "component": "Step"},
    )
    fig.update_traces(hovertemplate=(
        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
        "%{fullData.name}: %{customdata[2]}<br>"
        "Total for one request: %{customdata[3]}<extra></extra>"
    ))
    fig.add_vline(
        x=FAST_LIMIT_MS, line_dash="dot", line_color="#16A34A",
        annotation_text="1 s target", annotation_position="top",
    )
    fig.add_vline(
        x=HARD_LIMIT_MS, line_dash="dash", line_color="#DC2626",
        annotation_text="1.5 s hard limit", annotation_position="top right",
    )
    fig.update_xaxes(
        **latency_axis(
            max(one_request["total_ms"].max(), HARD_LIMIT_MS / 1.2)
        ),
        title="Total latency of one request",
    )
    fig.update_yaxes(
        title="", autorange="reversed",
        categoryorder="array", categoryarray=bar_order,
    )
    fig.update_layout(
        height=max(380, 48 * len(bar_order) + 150),
        legend_title_text="Step", margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, key="chart_latency_one", **STRETCH)

    numbers = pd.DataFrame({
        "Model": one_request["provider_label"],
        "Chatbot": one_request["variant_label"],
        **{
            label: one_request[column].map(format_ms)
            for column, label in part_columns.items()
        },
        "Total (one request)": one_request["total_ms"].map(format_ms),
    })
    st.markdown("**Numbers behind the chart** (average for one request)")
    st.dataframe(numbers, hide_index=True, **STRETCH)
    st.caption(
        "A step a request did not use (for example retrieval in General "
        "Chat) counts as 0 ms, so the steps add up to the total."
    )


# =========================================================
# Chart 2: Answer quality (grouped bar)
# =========================================================

section(
    2, "Answer Quality",
    "Human-rated relevance, factual correctness and faithfulness "
    "(0-100, higher is better).",
    "#0D9488",
)

if rated.empty:
    st.info(
        "No quality scores have been collected yet, so this chart is empty. "
        "Scores are never generated automatically. Use **Rate answers** "
        "below to score saved answers; the chart fills in as you rate."
    )
else:
    quality_long = rated.melt(
        id_vars=["variant_label"],
        value_vars=list(QUALITY_DIMENSIONS),
        var_name="dimension", value_name="score",
    ).dropna(subset=["score"])
    quality_long["dimension"] = quality_long["dimension"].map(
        QUALITY_DIMENSIONS
    )
    quality_stats = (
        quality_long.groupby(["variant_label", "dimension"])["score"]
        .agg(avg="mean", n="size")
        .reset_index()
    )
    quality_stats["text"] = quality_stats["avg"].map(lambda v: f"{v:.0f}")
    fig = px.bar(
        quality_stats, x="variant_label", y="avg", color="dimension",
        barmode="group", text="text",
        color_discrete_map=QUALITY_COLORS,
        pattern_shape="dimension",
        pattern_shape_map={
            "Relevance": "", "Factual correctness": "/",
            "Faithfulness": "x",
        },
        category_orders={
            "variant_label": list(VARIANT_COLORS),
            "dimension": list(QUALITY_COLORS),
        },
        custom_data=["n"],
        labels={
            "variant_label": "Chatbot", "avg": "Average score (0-100)",
            "dimension": "Quality dimension",
        },
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b> - %{fullData.name}<br>"
            "Average score: %{y:.1f} / 100<br>"
            "Ratings: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_yaxes(range=[0, 110], title="Average score (0-100)")
    fig.update_layout(height=400, bargap=0.25, margin=dict(t=30, b=20))
    st.plotly_chart(fig, key="chart_quality", **STRETCH)
    st.caption(
        f"{len(rated):,} rated answers in the current selection. "
        "A dimension you left unrated is not counted as zero."
    )

with st.expander("Rate answers (collect quality scores)"):
    st.caption(
        "Score a saved answer from 0 to 100 on each dimension. Faithfulness "
        "means the answer sticks to the retrieved document or known facts "
        "and adds nothing invented. Ratings are saved to "
        "`evaluation/results/quality_ratings.csv` under your username."
    )
    candidates = successful[
        successful["answer"].fillna("").astype(str).str.strip() != ""
    ].sort_values("timestamp_utc", ascending=False).head(200)

    if candidates.empty:
        st.info("There are no successful answers to rate in this selection.")
    else:
        rated_keys = set(
            zip(ratings["variant"].astype(str),
                ratings["timestamp_utc"].astype(str))
        ) if not ratings.empty else set()

        def option_label(idx) -> str:
            row = candidates.loc[idx]
            done = "✓ " if (
                str(row["variant"]), str(row["timestamp_utc"])
            ) in rated_keys else ""
            question = str(row["question_original"])[:55]
            return (
                f"{done}{str(row['timestamp_utc'])[:16]} | "
                f"{row['variant_label']} | {row['provider_label']} | "
                f"{question}"
            )

        choice = st.selectbox(
            "Answer to rate (latest 200 successful answers)",
            options=list(candidates.index), format_func=option_label,
        )
        row = candidates.loc[choice]
        st.markdown(f"**Question:** {row['question_original']}")
        st.markdown("**Answer:**")
        st.write(str(row["answer"]))

        scores = {}
        slider_columns = st.columns(3)
        for column, (key, label) in zip(
            slider_columns, QUALITY_DIMENSIONS.items()
        ):
            scores[key] = column.slider(
                label, 0, 100, 50, key=f"rate_{key}_{choice}"
            )
        if st.button("Save rating", key="save_rating"):
            save_rating({
                "username": username,
                "variant": str(row["variant"]),
                "timestamp_utc": str(row["timestamp_utc"]),
                **scores,
                "rated_at": datetime.now(timezone.utc).isoformat(),
            })
            st.success("Rating saved. Rerun the page to update the charts.")


# =========================================================
# Chart 3: Token usage (stacked bar)
# =========================================================

section(
    3, "Token Usage",
    "Average input and output tokens per successful response, "
    "by provider and chatbot.",
    "#F97316",
)

if token_rows.empty:
    st.info("No token counts were logged for successful requests.")
else:
    token_stats = (
        token_rows.groupby(["provider_label", "variant_label"])
        .agg(
            input_avg=("input_tokens", "mean"),
            output_avg=("output_tokens", "mean"),
            n=("input_tokens", "size"),
            total=("total_tokens", "sum"),
        )
        .reset_index()
    )
    token_long = token_stats.melt(
        id_vars=["provider_label", "variant_label", "n", "total"],
        value_vars=["input_avg", "output_avg"],
        var_name="kind", value_name="tokens",
    )
    token_long["kind"] = token_long["kind"].map(
        {"input_avg": "Input tokens", "output_avg": "Output tokens"}
    )
    fig = px.bar(
        token_long, x="provider_label", y="tokens", color="kind",
        facet_col="variant_label", barmode="relative",
        color_discrete_map={
            "Input tokens": INPUT_COLOR, "Output tokens": OUTPUT_COLOR,
        },
        pattern_shape="kind",
        pattern_shape_map={"Input tokens": "", "Output tokens": "."},
        category_orders={
            "variant_label": list(VARIANT_COLORS),
            "provider_label": ordered_providers(token_long["provider_label"]),
            "kind": ["Input tokens", "Output tokens"],
        },
        custom_data=["n", "total"],
        labels={
            "provider_label": "Provider / model",
            "tokens": "Average tokens per response", "kind": "Token type",
        },
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(hovertemplate=(
        "<b>%{x}</b><br>%{fullData.name}: %{y:,.0f} tokens (average)<br>"
        "Successful responses: %{customdata[0]}<br>"
        "Total tokens used: %{customdata[1]:,.0f}<extra></extra>"
    ))
    fig.update_layout(height=430, margin=dict(t=40, b=20),
                      legend_title_text="Token type")
    st.plotly_chart(fig, key="chart_tokens", **STRETCH)

    token_summary = st.columns(len(VARIANT_COLORS) + 1)
    token_summary[0].metric(
        "Total tokens (selection)",
        format_num(token_rows["total_tokens"].sum()),
    )
    for column, label in zip(token_summary[1:], VARIANT_COLORS):
        part = token_rows[token_rows["variant_label"] == label]
        column.metric(
            f"{label}: avg tokens / success",
            format_num(part["total_tokens"].mean()) if not part.empty
            else "N/A",
        )
    st.caption(
        "Rows without token counts are skipped, not counted as zero. "
        "Provider pricing is not configured in this project, so cost "
        "estimates are not shown."
    )


# =========================================================
# Chart 4: Reliability (donut)
# =========================================================

section(
    4, "Reliability",
    "Share of successful and failed requests for each chatbot.",
    "#16A34A",
)

if known_count == 0:
    st.info("No success/failure status is available for this selection.")
else:
    scope_options = ["All providers"] + sorted(
        filtered["provider_label"].dropna().unique()
    )
    scope = st.selectbox("Provider scope", scope_options, key="rel_scope")
    scoped = (
        filtered if scope == "All providers"
        else filtered[filtered["provider_label"] == scope]
    )

    labels_present = [
        v for v in VARIANT_COLORS if v in set(scoped["variant_label"])
    ] + sorted(set(scoped["variant_label"]) - set(VARIANT_COLORS))

    if not labels_present:
        st.info("No requests for this provider in the selection.")
    else:
        fig = make_subplots(
            rows=1, cols=len(labels_present),
            specs=[[{"type": "domain"}] * len(labels_present)],
            subplot_titles=labels_present,
        )
        summary = []
        for position, label in enumerate(labels_present, start=1):
            part = scoped[scoped["variant_label"] == label]
            ok = int(part["ok"].sum())
            failed = int(part["fail"].sum())
            unknown = int((~part["known"]).sum())
            known = ok + failed
            summary.append((label, ok, failed, unknown, known))
            if known == 0:
                continue
            fig.add_trace(
                go.Pie(
                    labels=["Successful", "Failed"],
                    values=[ok, failed], hole=0.62,
                    marker=dict(colors=[SUCCESS_COLOR, FAILURE_COLOR]),
                    textinfo="label+percent", sort=False,
                    hovertemplate=(
                        "%{label}: %{value} requests (%{percent})"
                        f"<extra>{html.escape(label)}</extra>"
                    ),
                ),
                row=1, col=position,
            )
            domain = fig.data[-1].domain
            fig.add_annotation(
                text=f"<b>{ok / known * 100:.0f}%</b><br>success",
                x=sum(domain.x) / 2, y=sum(domain.y) / 2,
                xref="paper", yref="paper",
                showarrow=False, font=dict(size=18),
            )
        fig.update_layout(
            height=400, showlegend=True, margin=dict(t=60, b=10),
        )
        st.plotly_chart(fig, key="chart_reliability", **STRETCH)

        columns = st.columns(len(summary))
        for column, (label, ok, failed, unknown, known) in zip(
            columns, summary
        ):
            success_text = format_pct(ok / known * 100) if known else "N/A"
            failure_text = format_pct(failed / known * 100) if known else "N/A"
            column.metric(f"{label}: success rate", success_text)
            note = (
                f"Failure rate {failure_text}  ·  "
                f"{ok + failed + unknown:,} total requests"
            )
            if unknown:
                note += f"  ·  {unknown} without a recorded status (excluded)"
            column.caption(note)

        by_provider = (
            filtered[filtered["known"]]
            .groupby(["provider_label", "variant_label"])["ok"]
            .agg(Requests="size", Successful="sum")
            .reset_index()
        )
        by_provider["Success rate"] = (
            by_provider["Successful"] / by_provider["Requests"] * 100
        ).map(format_pct)
        with st.expander("Reliability by provider"):
            st.dataframe(
                by_provider.rename(columns={
                    "provider_label": "Provider / model",
                    "variant_label": "Chatbot",
                }),
                hide_index=True, **STRETCH,
            )


# =========================================================
# Chart 5: Preprocessing effectiveness (scatter)
# =========================================================

section(
    5, "Preprocessing Effectiveness",
    "Does the time spent preprocessing a question buy a better answer? "
    "Each dot is one question asked to both chatbots.",
    "#8B5CF6",
)


def add_match_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Key used to pair the same question across chatbots.

    Uses case_id when logged, otherwise the normalised question text.
    """
    frame = frame.copy()
    text_key = (
        frame["question_original"].fillna("").astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    case = frame["case_id"].fillna("").astype(str).str.strip()
    frame["match_key"] = case.where(case != "", text_key)
    return frame[frame["match_key"] != ""]


matched = add_match_key(filtered)
pairs_found = len(
    set(zip(
        matched.loc[matched["variant_label"] == "Baseline", "provider_label"],
        matched.loc[matched["variant_label"] == "Baseline", "match_key"],
    ))
    & set(zip(
        matched.loc[matched["variant_label"] == "Preprocessed", "provider_label"],
        matched.loc[matched["variant_label"] == "Preprocessed", "match_key"],
    ))
)

scatter = pd.DataFrame()
if not rated.empty:
    rated_matched = add_match_key(rated)
    keys = ["provider_label", "match_key"]
    left = (
        rated_matched[rated_matched["variant_label"] == "Baseline"]
        .groupby(keys).agg(base_quality=("quality_overall", "mean"))
    )
    right = (
        rated_matched[rated_matched["variant_label"] == "Preprocessed"]
        .groupby(keys).agg(
            pre_quality=("quality_overall", "mean"),
            preprocessing_ms=("preprocessing_ms", "mean"),
            retrieval_ms=("retrieval_ms", "mean"),
            original_length=(
                "question_original", lambda s: s.astype(str).str.len().mean()
            ),
            processed_length=(
                "question_processed", lambda s: s.astype(str).str.len().mean()
            ),
            mode=("mode", "first"),
            question=("question_original", "first"),
        )
    )
    scatter = left.join(right, how="inner").reset_index()
    scatter = scatter.dropna(subset=["preprocessing_ms"])

if scatter.empty:
    st.info(
        "This chart needs matched, rated question pairs, and none are "
        "available yet."
    )
    st.markdown(
        "**What to collect**\n"
        "1. Ask the **same question** in both the Baseline and Preprocessed "
        "chatbots using the same provider.\n"
        "2. Rate both answers in **Rate answers** (section 2).\n"
        "3. Questions are matched by identical question text (or by "
        "`case_id` when one is logged).\n\n"
        f"Currently: **{pairs_found}** question(s) asked in both chatbots "
        f"with the same provider, **{len(scatter)}** with ratings for both."
    )
else:
    scatter["improvement"] = scatter["pre_quality"] - scatter["base_quality"]
    scatter["pre_text"] = scatter["preprocessing_ms"].map(format_ms)
    scatter["ret_text"] = scatter["retrieval_ms"].map(format_ms)
    scatter["question_short"] = scatter["question"].astype(str).str.slice(0, 80)
    scatter["mode"] = scatter["mode"].fillna("Unknown")
    colors = {
        label: PROVIDER_PALETTE[i % len(PROVIDER_PALETTE)]
        for i, label in enumerate(sorted(scatter["provider_label"].unique()))
    }
    fig = px.scatter(
        scatter, x="preprocessing_ms", y="improvement",
        color="provider_label", symbol="mode",
        color_discrete_map=colors,
        custom_data=[
            "question_short", "pre_text", "ret_text", "original_length",
            "processed_length", "base_quality", "pre_quality",
        ],
        labels={
            "preprocessing_ms": "Preprocessing time",
            "improvement": "Quality improvement (Preprocessed - Baseline)",
            "provider_label": "Provider / model", "mode": "Chat mode",
        },
    )
    fig.update_traces(
        marker=dict(size=13, line=dict(width=1, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Preprocessing: %{customdata[1]}<br>"
            "Retrieval: %{customdata[2]}<br>"
            "Question length: %{customdata[3]:.0f} chars "
            "(processed %{customdata[4]:.0f})<br>"
            "Quality: baseline %{customdata[5]:.1f} -> "
            "preprocessed %{customdata[6]:.1f}<br>"
            "Improvement: %{y:+.1f} points<extra></extra>"
        ),
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color="#64748B",
        annotation_text="no change", annotation_position="top left",
    )
    fig.update_xaxes(
        **latency_axis(scatter["preprocessing_ms"].max()),
        title="Preprocessing time",
    )
    fig.update_yaxes(title="Quality improvement (points, 0-100 scale)")
    fig.update_layout(height=440, margin=dict(t=30, b=20))
    st.plotly_chart(fig, key="chart_scatter", **STRETCH)
    st.caption(
        f"{len(scatter)} matched question pair(s). Above the dashed line the "
        "Preprocessed answer scored higher than the Baseline answer."
    )


# =========================================================
# Extra latency view: by chat mode
# =========================================================

st.divider()
st.subheader("Average Latency by Chat Mode")

mode_rows = latency_rows.dropna(subset=["mode"])
if mode_rows.empty:
    st.info("No latency data with chat mode is available.")
else:
    mode_stats = (
        mode_rows.groupby(["mode", "variant_label"])["total_ms"]
        .agg(avg="mean", n="size").reset_index()
    )
    mode_stats["avg_text"] = mode_stats["avg"].map(format_ms)
    fig = px.bar(
        mode_stats, x="mode", y="avg", color="variant_label",
        barmode="group", text="avg_text",
        color_discrete_map=VARIANT_COLORS,
        pattern_shape="variant_label",
        pattern_shape_map=VARIANT_PATTERNS,
        custom_data=["avg_text", "n"],
        labels={"mode": "Chat mode", "avg": "Average total latency",
                "variant_label": "Chatbot"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b> (%{fullData.name})<br>"
            "Average: %{customdata[0]}<br>"
            "Requests: %{customdata[1]}<extra></extra>"
        ),
    )
    fig.update_yaxes(
        **latency_axis(mode_stats["avg"].max()),
        title="Average total latency",
    )
    fig.update_layout(height=380, margin=dict(t=30, b=20))
    st.plotly_chart(fig, key="chart_mode_latency", **STRETCH)


# =========================================================
# Records table and download
# =========================================================

st.divider()
st.subheader("Saved Evaluation Records")

display_columns = [
    c for c in [
        "timestamp_utc", "variant_label", "case_id", "mode",
        "provider_label", "model_id", "total_ms", "preprocessing_ms",
        "retrieval_ms", "api_ms", "input_tokens", "output_tokens",
        "error_type",
    ] if c in filtered.columns
]
table = filtered[display_columns].copy()
table["status"] = np.select(
    [filtered["ok"], filtered["fail"]], ["Success", "Failed"], "Unknown"
)
table = table.sort_values("timestamp_utc", ascending=False, na_position="last")
for column in ("total_ms", "preprocessing_ms", "retrieval_ms", "api_ms"):
    if column in table.columns:
        table[column] = table[column].map(format_ms)
table = table.rename(columns={
    "variant_label": "chatbot", "provider_label": "provider",
})
st.dataframe(table, hide_index=True, **STRETCH)

download = filtered.drop(
    columns=["variant_label", "provider_label", "timestamp",
             "ok", "fail", "known"],
    errors="ignore",
)
st.download_button(
    label="Download My Filtered Results (CSV)",
    data=download.to_csv(index=False).encode("utf-8"),
    file_name="my_evaluation_results.csv",
    mime="text/csv",
)
st.caption(
    "This dashboard reads saved CSV records only; it does not rerun tests. "
    "The download keeps raw latency values in milliseconds."
)