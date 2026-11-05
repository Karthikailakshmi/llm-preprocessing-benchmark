"""Compare saved Baseline and Preprocessed chatbot results.

Six features:
  1. Overall improvement summary (latency, tokens, success rate)
  2. Side-by-side answer comparison for matched questions
  3. Per-question performance differences (latency and tokens)
  4. Provider and model comparison (speed and reliability)
  5. Latency target analysis (1 s target, 1.5 s hard limit)
  6. Detailed comparison CSV export

Only saved CSV records are read. Nothing is re-run, and missing values are
never treated as zero or as success.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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
    page_title="Compare Chatbot Results",
    page_icon="⚖️",
    layout="wide",
)


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

TARGET_MS = 1000.0
HARD_LIMIT_MS = 1500.0

BASE_LABEL = "Baseline"
PRE_LABEL = "Preprocessed"
VARIANT_COLORS = {BASE_LABEL: "#2563EB", PRE_LABEL: "#7C3AED"}
VARIANT_PATTERNS = {BASE_LABEL: "", PRE_LABEL: "/"}

VERDICT_COLORS = {
    "Faster": "#16A34A", "Slower": "#DC2626", "Similar": "#64748B",
}

TARGET_CATEGORIES = {
    "meets": "Meets 1 s target",
    "over": "Over 1 s, within 1.5 s limit",
    "exceeds": "Exceeds 1.5 s hard limit",
}
TARGET_COLORS = {
    TARGET_CATEGORIES["meets"]: "#16A34A",
    TARGET_CATEGORIES["over"]: "#F59E0B",
    TARGET_CATEGORIES["exceeds"]: "#DC2626",
}
TARGET_PATTERNS = {
    TARGET_CATEGORIES["meets"]: "",
    TARGET_CATEGORIES["over"]: "/",
    TARGET_CATEGORIES["exceeds"]: "x",
}

NUMERIC_COLUMNS = [
    "total_ms", "preprocessing_ms", "retrieval_ms", "api_ms",
    "rate_limit_wait_ms", "input_tokens", "output_tokens",
]
EXPECTED_COLUMNS = [
    "provider", "model_id", "mode", "case_id", "question_original",
    "question_processed", "answer", "error_type", "error_message",
    "retrieved_sources",
] + NUMERIC_COLUMNS

_VERSION = tuple(
    int(part) for part in st.__version__.split(".")[:2] if part.isdigit()
)
STRETCH = (
    {"width": "stretch"} if _VERSION >= (1, 50)
    else {"use_container_width": True}
)


# ---------------------------------------------------------
# Styling and layout helpers
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .dash-sub {opacity: .75; margin-top: -.6rem; margin-bottom: 1rem;}
    .section-bar {
        border-left: 6px solid var(--accent, #2563EB);
        padding-left: 12px; margin: 2rem 0 .4rem 0;
    }
    .section-bar h3 {margin: 0; padding: 0;}
    .section-bar p {margin: 2px 0 0 0; opacity: .75; font-size: .9rem;}
    .answer-head {
        border-left: 6px solid var(--accent, #2563EB);
        padding: 4px 10px; margin-bottom: 6px; font-weight: 700;
        background: rgba(128,128,128,.07); border-radius: 0 8px 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Baseline vs Preprocessing")
st.markdown(
    '<div class="dash-sub">Does preprocessing make the chatbot faster, '
    "cheaper and more reliable? Compares saved results only; no tests "
    "are rerun.</div>",
    unsafe_allow_html=True,
)


def section(number: int, title: str, subtitle: str, accent: str) -> None:
    st.markdown(
        f'<div class="section-bar" style="--accent:{accent}">'
        f"<h3>{number}. {html.escape(title)}</h3>"
        f"<p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
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


def format_ms_signed(value) -> str:
    """Format a latency difference with an explicit sign."""
    if value is None or pd.isna(value):
        return "N/A"
    if round(float(value)) == 0:
        return "0 ms"
    sign = "+" if value > 0 else "-"
    return f"{sign}{format_ms(abs(value))}"


def format_pct(value, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.1f}%" if signed else f"{value:.1f}%"


def format_num(value, digits: int = 0, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"


def pct_change(base: float, new: float) -> float:
    """Percentage change from base to new; NaN when it is undefined."""
    if pd.isna(base) or pd.isna(new) or base == 0:
        return float("nan")
    return (new - base) / abs(base) * 100


def latency_axis(max_ms: float) -> dict:
    top = max(float(max_ms) * 1.2, 1.0)
    ticks = np.linspace(0, top, 6)
    return {
        "tickmode": "array", "tickvals": ticks.tolist(),
        "ticktext": [format_ms(t) for t in ticks], "range": [0, top],
    }


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
    """Read one results CSV, realigning rows from older log versions."""
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

    # Success is three-valued; an unknown status is never a success.
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


def load_all_results() -> pd.DataFrame:
    folder = Path(RESULTS_DIR)
    if not folder.exists():
        return pd.DataFrame()
    files = tuple(
        (str(p), p.stat().st_mtime)
        for p in sorted(folder.glob("*_interactive.csv"))
    )
    return _load_results(files) if files else pd.DataFrame()


def get_username() -> str | None:
    user = st.session_state.get("user")
    if isinstance(user, dict):
        username = user.get("username")
        return str(username) if username else None
    return None


# ---------------------------------------------------------
# Load data and enforce per-user access
# ---------------------------------------------------------

username = get_username()
if not username:
    st.warning("Please log in to compare your saved results.")
    st.stop()

all_results = load_all_results()
if all_results.empty:
    st.info(
        "No saved interactive results were found. "
        "Run the chatbot tests first, then return to this page."
    )
    st.stop()

if "username" not in all_results.columns:
    st.info("Saved results do not record usernames, so none can be shown.")
    st.stop()

user_results = all_results[
    all_results["username"].fillna("").astype(str).str.strip().str.casefold()
    == username.casefold()
].copy()

if user_results.empty:
    st.info(
        "No results associated with your account were found. Results "
        "recorded before username tracking was introduced cannot "
        "automatically be assigned to a user."
    )
    st.stop()

user_results["provider_key"] = user_results["provider"].fillna("Unknown").astype(str)
user_results["provider_label"] = user_results["provider"].map(
    lambda p: provider_label(p) if pd.notna(p) else "Unknown"
)
user_results["mode_key"] = user_results["mode"].fillna("Unknown").astype(str)


# ---------------------------------------------------------
# Sidebar: variants and filters
# ---------------------------------------------------------

st.sidebar.header("Comparison Settings")

variant_names = sorted(
    user_results["variant"].dropna().astype(str).unique().tolist()
)
baseline_candidates = [v for v in variant_names if "baseline" in v.casefold()]
preprocessed_candidates = [
    v for v in variant_names
    if "preprocess" in v.casefold() and "baseline" not in v.casefold()
]
if not baseline_candidates:
    st.warning("Could not identify a baseline variant from the saved names.")
if not preprocessed_candidates:
    st.warning("Could not identify a preprocessing variant from the saved names.")

default_baseline = (
    baseline_candidates[0] if baseline_candidates else variant_names[0]
)
default_preprocessed = (
    preprocessed_candidates[0] if preprocessed_candidates
    else next((v for v in variant_names if v != default_baseline),
              default_baseline)
)

baseline_variant = st.sidebar.selectbox(
    "Baseline variant", variant_names,
    index=variant_names.index(default_baseline),
)
preprocessed_variant = st.sidebar.selectbox(
    "Preprocessed variant", variant_names,
    index=variant_names.index(default_preprocessed),
)
if baseline_variant == preprocessed_variant:
    st.warning(
        "You selected the same variant twice. Select different variants "
        "for a meaningful comparison."
    )

scoped = user_results.copy()

mode_options = sorted(scoped["mode_key"].unique())
picked_modes = st.sidebar.multiselect(
    "Chat mode", mode_options, default=mode_options
)
scoped = scoped[scoped["mode_key"].isin(picked_modes)]

provider_options = ordered_providers(scoped["provider_label"].unique())
picked_providers = st.sidebar.multiselect(
    "Provider / model", provider_options, default=provider_options
)
scoped = scoped[scoped["provider_label"].isin(picked_providers)]

include_failed_latency = st.sidebar.checkbox(
    "Include failed requests in latency",
    value=False,
    help=(
        "Off (default): latency statistics use successful requests only, "
        "so timeouts and instant API-key errors do not distort them."
    ),
)
similar_pct = st.sidebar.slider(
    "Treat latency changes within ± (%) as similar", 0, 25, 5,
    help="Run-to-run API timing is noisy, so tiny changes are not "
         "reported as faster or slower.",
)

baseline = scoped[scoped["variant"].astype(str) == baseline_variant].copy()
preprocessed = scoped[scoped["variant"].astype(str) == preprocessed_variant].copy()

st.caption(f"Comparing saved results for account: **{username}**")

if baseline.empty or preprocessed.empty:
    st.warning(
        "One or both selected variants have no saved results for this "
        "account with the current filters."
    )
    st.stop()


def latency_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame if include_failed_latency else frame[frame["ok"]]
    return rows.dropna(subset=["total_ms"])


def token_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[frame["ok"]].dropna(subset=["input_tokens", "output_tokens"])
    return rows.assign(
        total_tokens=rows["input_tokens"] + rows["output_tokens"]
    )


def success_rate(frame: pd.DataFrame) -> float:
    known = int(frame["known"].sum())
    return frame["ok"].sum() / known * 100 if known else float("nan")


base_lat, pre_lat = latency_frame(baseline), latency_frame(preprocessed)
base_tok, pre_tok = token_frame(baseline), token_frame(preprocessed)


# =========================================================
# 1. Overall improvement summary
# =========================================================

section(
    1, "Overall Improvement Summary",
    "Average latency, token usage and success rate, Baseline vs "
    "Preprocessed.",
    "#2563EB",
)

base_avg_lat = base_lat["total_ms"].mean() if not base_lat.empty else np.nan
pre_avg_lat = pre_lat["total_ms"].mean() if not pre_lat.empty else np.nan
base_avg_tok = base_tok["total_tokens"].mean() if not base_tok.empty else np.nan
pre_avg_tok = pre_tok["total_tokens"].mean() if not pre_tok.empty else np.nan
base_rate, pre_rate = success_rate(baseline), success_rate(preprocessed)

lat_change = pct_change(base_avg_lat, pre_avg_lat)
tok_change = pct_change(base_avg_tok, pre_avg_tok)
rate_change = (
    pre_rate - base_rate if pd.notna(base_rate) and pd.notna(pre_rate)
    else np.nan
)


def summary_card(title, base_text, pre_text, delta_text, lower_is_better,
                 requests_note):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        left, right = st.columns(2)
        left.metric(BASE_LABEL, base_text)
        right.metric(
            PRE_LABEL, pre_text,
            delta=delta_text,
            delta_color="inverse" if lower_is_better else "normal",
        )
        st.caption(requests_note)


cols = st.columns(3)
with cols[0]:
    summary_card(
        "Average total latency", format_ms(base_avg_lat),
        format_ms(pre_avg_lat),
        f"{format_pct(lat_change, signed=True)} vs Baseline"
        if pd.notna(lat_change) else None,
        True,
        f"{len(base_lat):,} vs {len(pre_lat):,} requests. "
        + ("Includes failures." if include_failed_latency
           else "Successful requests only."),
    )
with cols[1]:
    summary_card(
        "Average tokens per response", format_num(base_avg_tok),
        format_num(pre_avg_tok),
        f"{format_pct(tok_change, signed=True)} vs Baseline"
        if pd.notna(tok_change) else None,
        True,
        f"{len(base_tok):,} vs {len(pre_tok):,} successful responses "
        "with token counts.",
    )
with cols[2]:
    summary_card(
        "Success rate", format_pct(base_rate), format_pct(pre_rate),
        f"{rate_change:+.1f} points vs Baseline"
        if pd.notna(rate_change) else None,
        False,
        f"{int(baseline['known'].sum()):,} vs "
        f"{int(preprocessed['known'].sum()):,} requests with a recorded "
        "status.",
    )

verdicts = []
if pd.notna(lat_change):
    verdicts.append(
        f"**Latency:** Preprocessed is {abs(lat_change):.1f}% "
        f"{'faster' if lat_change < 0 else 'slower'} on average."
        if round(lat_change, 1) != 0 else "**Latency:** no average change."
    )
if pd.notna(tok_change):
    verdicts.append(
        f"**Tokens:** Preprocessed uses {abs(tok_change):.1f}% "
        f"{'fewer' if tok_change < 0 else 'more'} tokens per response."
        if round(tok_change, 1) != 0 else "**Tokens:** no average change."
    )
if pd.notna(rate_change):
    verdicts.append(
        f"**Reliability:** success rate is {abs(rate_change):.1f} points "
        f"{'higher' if rate_change > 0 else 'lower'} with Preprocessed."
        if round(rate_change, 1) != 0 else
        "**Reliability:** no change in success rate."
    )
if verdicts:
    st.markdown("  \n".join(verdicts))

pre_time = preprocessed.loc[preprocessed["preprocessing_ms"].notna(),
                            "preprocessing_ms"].mean()
st.caption(
    "Averages mix different questions unless both chatbots were asked the "
    "same ones; see sections 2 and 3 for like-for-like comparisons. "
    f"Average preprocessing step time: {format_ms(pre_time)}."
)


# ---------------------------------------------------------
# Match the same question across the two chatbots
# ---------------------------------------------------------

PAIR_KEYS = ["provider_key", "mode_key", "match_key"]
PAIR_COLUMNS = [
    "provider_label", "model_id", "question_original", "question_processed",
    "answer", "total_ms", "input_tokens", "output_tokens", "ok", "fail",
    "known", "error_type", "error_message", "retrieved_sources",
    "timestamp_utc", "timestamp",
]


def add_match_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Key pairing the same question: case_id if logged, else its text."""
    frame = frame.copy()
    text_key = (
        frame["question_original"].fillna("").astype(str).str.lower()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    case = frame["case_id"].fillna("").astype(str).str.strip()
    frame["match_key"] = case.where(case != "", text_key)
    return frame[frame["match_key"] != ""]


def latest_per_question(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per question: the latest successful run, else the latest."""
    return (
        add_match_key(frame)
        .sort_values(["ok", "timestamp"], na_position="first")
        .groupby(PAIR_KEYS, sort=False)
        .tail(1)
        [PAIR_KEYS + PAIR_COLUMNS]
    )


def build_pairs(base: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    pairs = latest_per_question(base).merge(
        latest_per_question(pre), on=PAIR_KEYS, suffixes=("_base", "_pre")
    )
    if pairs.empty:
        return pairs

    pairs["tokens_base"] = pairs["input_tokens_base"] + pairs["output_tokens_base"]
    pairs["tokens_pre"] = pairs["input_tokens_pre"] + pairs["output_tokens_pre"]
    both_ok = pairs["ok_base"] & pairs["ok_pre"]

    # Differences are only meaningful when both runs succeeded.
    pairs["latency_change_ms"] = np.where(
        both_ok, pairs["total_ms_pre"] - pairs["total_ms_base"], np.nan
    )
    pairs["latency_change_pct"] = [
        pct_change(b, p) if ok else np.nan
        for b, p, ok in zip(pairs["total_ms_base"], pairs["total_ms_pre"], both_ok)
    ]
    pairs["token_change"] = np.where(
        both_ok, pairs["tokens_pre"] - pairs["tokens_base"], np.nan
    )
    pairs["token_change_pct"] = [
        pct_change(b, p) if ok else np.nan
        for b, p, ok in zip(pairs["tokens_base"], pairs["tokens_pre"], both_ok)
    ]

    pct = pairs["latency_change_pct"]
    pairs["verdict"] = np.select(
        [pct.isna(), pct.abs() <= similar_pct, pct < 0],
        ["Not comparable", "Similar", "Faster"],
        "Slower",
    )
    for side in ("base", "pre"):
        pairs[f"status_{side}"] = np.select(
            [pairs[f"ok_{side}"], pairs[f"fail_{side}"]],
            ["Success", "Failed"], "Unknown",
        )
    return pairs


pairs = build_pairs(baseline, preprocessed)


def no_pairs_message() -> None:
    st.info(
        "No questions have been asked in both chatbots yet, so there is "
        "nothing to compare like-for-like."
    )
    st.markdown(
        "Ask the **same question** in the Baseline and the Preprocessed "
        "chatbot, using the **same provider and chat mode**. Questions are "
        "matched by identical text (ignoring case and extra spaces), or by "
        "`case_id` when one is logged."
    )


# =========================================================
# 2. Side-by-side answer comparison
# =========================================================

section(
    2, "Side-by-Side Answer Comparison",
    "The same question, its original and processed forms, and both answers.",
    "#7C3AED",
)

if pairs.empty:
    no_pairs_message()
else:
    def pair_label(idx) -> str:
        row = pairs.loc[idx]
        question = str(row["question_original_base"]).replace("\n", " ")[:60]
        return (
            f"{row['provider_label_base']} | {row['mode_key']} | {question}"
        )

    chosen_pair = st.selectbox(
        f"Matched question ({len(pairs)} available)",
        options=list(pairs.index), format_func=pair_label,
        key="pair_choice",
    )
    pair = pairs.loc[chosen_pair]

    st.markdown(f"**Original question:** {pair['question_original_base']}")
    processed = pair["question_processed_pre"]
    original = pair["question_original_pre"]
    if pd.isna(processed) or str(processed).strip() == "":
        st.markdown("**Processed question:** not recorded")
    elif str(processed).strip() == str(original).strip():
        st.markdown("**Processed question:** unchanged by preprocessing")
    else:
        st.markdown(f"**Processed question:** {processed}")

    verdict = pair["verdict"]
    verdict_text = (
        f"Latency {format_ms_signed(pair['latency_change_ms'])} "
        f"({format_pct(pair['latency_change_pct'], signed=True)}), "
        f"tokens {format_num(pair['token_change'], signed=True)}"
        if verdict != "Not comparable"
        else "Not comparable because at least one run failed or has no "
             "recorded status"
    )
    st.caption(f"Result: **{verdict}**. {verdict_text}.")

    left, right = st.columns(2)
    for column, side, label, color in (
        (left, "base", BASE_LABEL, VARIANT_COLORS[BASE_LABEL]),
        (right, "pre", PRE_LABEL, VARIANT_COLORS[PRE_LABEL]),
    ):
        with column:
            st.markdown(
                f'<div class="answer-head" style="--accent:{color}">'
                f"{label}</div>",
                unsafe_allow_html=True,
            )
            status = pair[f"status_{side}"]
            if status == "Success":
                st.write(str(pair[f"answer_{side}"]))
            elif status == "Failed":
                st.error(
                    f"{pair[f'error_type_{side}'] or 'Error'}: "
                    f"{pair[f'error_message_{side}'] or 'no details logged'}"
                )
            else:
                st.warning("No recorded status for this request.")
            tokens = pair[f"tokens_{side}"]
            st.caption(
                f"{pair[f'model_id_{side}']}  ·  "
                f"{format_ms(pair[f'total_ms_{side}'])}  ·  "
                f"{format_num(tokens)} tokens  ·  {status}"
            )
            sources = pair[f"retrieved_sources_{side}"]
            if pd.notna(sources) and str(sources).strip():
                st.caption(f"Sources: {sources}")

    st.caption(
        "Matched by provider, chat mode and identical question text. If a "
        "question was asked more than once, the latest successful run of "
        "each chatbot is shown."
    )


# =========================================================
# 3. Per-question performance differences
# =========================================================

section(
    3, "Per-Question Performance Differences",
    "Which matching questions became faster or slower, and how their "
    "token usage changed.",
    "#F97316",
)

if pairs.empty:
    no_pairs_message()
else:
    counts = pairs["verdict"].value_counts()
    metrics = st.columns(4)
    metrics[0].metric("Matched questions", f"{len(pairs):,}")
    metrics[1].metric("Faster with preprocessing", int(counts.get("Faster", 0)))
    metrics[2].metric("Slower with preprocessing", int(counts.get("Slower", 0)))
    metrics[3].metric(
        "Similar / not comparable",
        f"{int(counts.get('Similar', 0))} / {int(counts.get('Not comparable', 0))}",
    )

    comparable = pairs[pairs["verdict"] != "Not comparable"].copy()
    if comparable.empty:
        st.info(
            "None of the matched questions succeeded in both chatbots, so "
            "no latency or token differences can be calculated."
        )
    else:
        comparable["question_short"] = (
            comparable["question_original_base"].astype(str)
            .str.replace("\n", " ").str.slice(0, 45)
            + " (" + comparable["provider_label_base"] + ")"
        )
        comparable["change_text"] = comparable["latency_change_ms"].map(
            format_ms_signed
        )
        comparable["base_text"] = comparable["total_ms_base"].map(format_ms)
        comparable["pre_text"] = comparable["total_ms_pre"].map(format_ms)
        comparable["tok_text"] = comparable["token_change"].map(
            lambda v: format_num(v, signed=True)
        )
        shown = comparable.reindex(
            comparable["latency_change_ms"].abs()
            .sort_values(ascending=False).head(25).index
        ).sort_values("latency_change_ms")

        fig = px.bar(
            shown, y="question_short", x="latency_change_ms", orientation="h",
            color="verdict", color_discrete_map=VERDICT_COLORS,
            pattern_shape="verdict",
            pattern_shape_map={"Faster": "", "Slower": "/", "Similar": "."},
            text="change_text",
            custom_data=["base_text", "pre_text", "change_text",
                         "latency_change_pct", "tok_text"],
            labels={"question_short": "Question",
                    "latency_change_ms": "Latency change (Preprocessed - Baseline)",
                    "verdict": "Result"},
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Baseline: %{customdata[0]}<br>"
                "Preprocessed: %{customdata[1]}<br>"
                "Change: %{customdata[2]} (%{customdata[3]:+.1f}%)<br>"
                "Token change: %{customdata[4]}<extra></extra>"
            ),
        )
        span = max(shown["latency_change_ms"].abs().max(), 1.0) * 1.3
        ticks = np.linspace(-span, span, 7)
        fig.update_xaxes(
            tickmode="array", tickvals=ticks.tolist(),
            ticktext=[format_ms_signed(t) for t in ticks],
            range=[-span, span], title="Latency change "
            "(left = faster, right = slower)",
        )
        fig.update_yaxes(title="", autorange="reversed")
        fig.update_layout(
            height=max(320, 38 * len(shown) + 120),
            margin=dict(t=20, b=20), legend_title_text="Result",
        )
        st.plotly_chart(fig, key="chart_pair_latency", **STRETCH)
        st.caption(
            f"Changes within ±{similar_pct}% count as similar. Only "
            "questions that succeeded in both chatbots are charted. "
            "Showing up to the 25 largest changes."
        )

    table = pd.DataFrame({
        "Provider / model": pairs["provider_label_base"],
        "Chat mode": pairs["mode_key"],
        "Question": pairs["question_original_base"].astype(str).str.slice(0, 80),
        "Result": pairs["verdict"],
        "Baseline latency": pairs["total_ms_base"].map(format_ms),
        "Preprocessed latency": pairs["total_ms_pre"].map(format_ms),
        "Latency change": pairs["latency_change_ms"].map(format_ms_signed),
        "Latency change %": pairs["latency_change_pct"].map(
            lambda v: format_pct(v, signed=True)),
        "Baseline tokens": pairs["tokens_base"].map(format_num),
        "Preprocessed tokens": pairs["tokens_pre"].map(format_num),
        "Token change": pairs["token_change"].map(
            lambda v: format_num(v, signed=True)),
        "Token change %": pairs["token_change_pct"].map(
            lambda v: format_pct(v, signed=True)),
    })
    with st.expander("Per-question table"):
        st.dataframe(table, hide_index=True, **STRETCH)


# =========================================================
# 4. Provider and model comparison
# =========================================================

section(
    4, "Provider and Model Comparison",
    "Response time and reliability for each provider and model.",
    "#0EA5E9",
)


def provider_summary(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    lat = latency_frame(frame).groupby("provider_label")["total_ms"].agg(
        avg_ms="mean", median_ms="median", latency_n="size"
    )
    rel = frame[frame["known"]].groupby("provider_label")["ok"].agg(
        requests="size", successes="sum"
    )
    model = frame.groupby("provider_label")["model_id"].first()
    out = rel.join(lat, how="outer").join(model, how="left").reset_index()
    out["success_rate"] = out["successes"] / out["requests"] * 100
    out["variant_label"] = label
    return out


provider_data = pd.concat(
    [provider_summary(baseline, BASE_LABEL),
     provider_summary(preprocessed, PRE_LABEL)],
    ignore_index=True,
)

if provider_data.empty:
    st.info("No provider or model records are available.")
else:
    order = ordered_providers(provider_data["provider_label"])
    chart_cols = st.columns(2)

    latency_chart = provider_data.dropna(subset=["avg_ms"]).copy()
    with chart_cols[0]:
        st.markdown("**Average response time**")
        if latency_chart.empty:
            st.info("No latency data for these providers.")
        else:
            latency_chart["avg_text"] = latency_chart["avg_ms"].map(format_ms)
            latency_chart["median_text"] = latency_chart["median_ms"].map(format_ms)
            fig = px.bar(
                latency_chart, x="provider_label", y="avg_ms",
                color="variant_label", barmode="group", text="avg_text",
                color_discrete_map=VARIANT_COLORS,
                pattern_shape="variant_label",
                pattern_shape_map=VARIANT_PATTERNS,
                category_orders={"provider_label": order,
                                 "variant_label": list(VARIANT_COLORS)},
                custom_data=["avg_text", "median_text", "latency_n"],
                labels={"provider_label": "Provider / model",
                        "avg_ms": "Average total latency",
                        "variant_label": "Chatbot"},
            )
            fig.update_traces(
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b> (%{fullData.name})<br>"
                    "Average: %{customdata[0]}<br>"
                    "Median: %{customdata[1]}<br>"
                    "Requests: %{customdata[2]}<extra></extra>"
                ),
            )
            fig.update_yaxes(**latency_axis(latency_chart["avg_ms"].max()),
                             title="Average total latency")
            fig.update_layout(height=400, margin=dict(t=20, b=20),
                              legend_title_text="Chatbot")
            st.plotly_chart(fig, key="chart_provider_latency", **STRETCH)

    reliability_chart = provider_data.dropna(subset=["success_rate"]).copy()
    with chart_cols[1]:
        st.markdown("**Reliability (success rate)**")
        if reliability_chart.empty:
            st.info("No success status data for these providers.")
        else:
            reliability_chart["rate_text"] = reliability_chart[
                "success_rate"].map(lambda v: f"{v:.0f}%")
            fig = px.bar(
                reliability_chart, x="provider_label", y="success_rate",
                color="variant_label", barmode="group", text="rate_text",
                color_discrete_map=VARIANT_COLORS,
                pattern_shape="variant_label",
                pattern_shape_map=VARIANT_PATTERNS,
                category_orders={"provider_label": order,
                                 "variant_label": list(VARIANT_COLORS)},
                custom_data=["requests", "successes"],
                labels={"provider_label": "Provider / model",
                        "success_rate": "Success rate (%)",
                        "variant_label": "Chatbot"},
            )
            fig.update_traces(
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b> (%{fullData.name})<br>"
                    "Success rate: %{y:.1f}%<br>"
                    "Successful: %{customdata[1]} of "
                    "%{customdata[0]} requests<extra></extra>"
                ),
            )
            fig.update_yaxes(range=[0, 115], title="Success rate (%)")
            fig.update_layout(height=400, margin=dict(t=20, b=20),
                              legend_title_text="Chatbot")
            st.plotly_chart(fig, key="chart_provider_success", **STRETCH)

    wide = provider_data.pivot(
        index="provider_label", columns="variant_label",
        values=["model_id", "requests", "avg_ms", "success_rate"],
    )
    rows = []
    for provider_name in order:
        if provider_name not in wide.index:
            continue
        row = wide.loc[provider_name]
        base_ms = row[("avg_ms", BASE_LABEL)] if ("avg_ms", BASE_LABEL) in row else np.nan
        pre_ms = row[("avg_ms", PRE_LABEL)] if ("avg_ms", PRE_LABEL) in row else np.nan
        base_sr = row[("success_rate", BASE_LABEL)] if ("success_rate", BASE_LABEL) in row else np.nan
        pre_sr = row[("success_rate", PRE_LABEL)] if ("success_rate", PRE_LABEL) in row else np.nan
        rows.append({
            "Provider / model": provider_name,
            "Baseline avg latency": format_ms(base_ms),
            "Preprocessed avg latency": format_ms(pre_ms),
            "Latency change": format_pct(pct_change(base_ms, pre_ms), signed=True),
            "Baseline success": format_pct(base_sr),
            "Preprocessed success": format_pct(pre_sr),
            "Success change": (
                f"{pre_sr - base_sr:+.1f} pts"
                if pd.notna(base_sr) and pd.notna(pre_sr) else "N/A"
            ),
        })
    with st.expander("Provider and model details"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
        st.caption(
            "Providers with very few requests give unreliable averages; "
            "check the request counts in the chart tooltips."
        )


# =========================================================
# 5. Latency target analysis
# =========================================================

section(
    5, "Latency Target Analysis",
    "How many responses meet the 1 s target and how many exceed the "
    "1.5 s hard limit, for each chatbot.",
    "#16A34A",
)


def target_counts(frame: pd.DataFrame) -> dict:
    lat = latency_frame(frame)["total_ms"]
    return {
        "n": len(lat),
        "meets": int((lat <= TARGET_MS).sum()),
        "over": int(((lat > TARGET_MS) & (lat <= HARD_LIMIT_MS)).sum()),
        "exceeds": int((lat > HARD_LIMIT_MS).sum()),
    }


target = {BASE_LABEL: target_counts(baseline), PRE_LABEL: target_counts(preprocessed)}

if min(info["n"] for info in target.values()) == 0:
    st.info("Latency measurements are missing for one of the chatbots.")
else:
    target_rows = []
    for label, info in target.items():
        for key, category in TARGET_CATEGORIES.items():
            target_rows.append({
                "Chatbot": label, "Category": category,
                "count": info[key], "percent": info[key] / info["n"] * 100,
                "n": info["n"],
            })
    target_df = pd.DataFrame(target_rows)
    target_df["text"] = [
        f"{c} ({p:.0f}%)" if c else ""
        for c, p in zip(target_df["count"], target_df["percent"])
    ]
    fig = px.bar(
        target_df, y="Chatbot", x="percent", color="Category",
        orientation="h", barmode="relative", text="text",
        color_discrete_map=TARGET_COLORS,
        pattern_shape="Category", pattern_shape_map=TARGET_PATTERNS,
        category_orders={
            "Chatbot": list(VARIANT_COLORS),
            "Category": list(TARGET_COLORS),
        },
        custom_data=["count", "n"],
        labels={"percent": "Share of responses (%)"},
    )
    fig.update_traces(
        textposition="inside",
        hovertemplate=(
            "<b>%{y}</b><br>%{fullData.name}<br>"
            "%{customdata[0]} of %{customdata[1]} responses "
            "(%{x:.1f}%)<extra></extra>"
        ),
    )
    fig.update_xaxes(range=[0, 100], title="Share of responses (%)")
    fig.update_yaxes(title="", autorange="reversed")
    fig.update_layout(height=280, margin=dict(t=20, b=20),
                      legend_title_text="Latency category")
    st.plotly_chart(fig, key="chart_target", **STRETCH)

    cols = st.columns(2)
    for column, (label, info) in zip(cols, target.items()):
        with column:
            with st.container(border=True):
                st.markdown(f"**{label}** ({info['n']:,} responses)")
                inner = st.columns(2)
                inner[0].metric(
                    "Meet 1 s target",
                    f"{info['meets']:,}",
                    f"{info['meets'] / info['n'] * 100:.1f}%",
                    delta_color="off",
                )
                inner[1].metric(
                    "Exceed 1.5 s limit",
                    f"{info['exceeds']:,}",
                    f"{info['exceeds'] / info['n'] * 100:.1f}%",
                    delta_color="off",
                )

    base_t, pre_t = target[BASE_LABEL], target[PRE_LABEL]
    base_meet = base_t["meets"] / base_t["n"] * 100
    pre_meet = pre_t["meets"] / pre_t["n"] * 100
    base_exc = base_t["exceeds"] / base_t["n"] * 100
    pre_exc = pre_t["exceeds"] / pre_t["n"] * 100
    st.markdown(
        f"**1 s target:** {PRE_LABEL} meets it for {pre_meet:.1f}% of "
        f"responses vs {base_meet:.1f}% for {BASE_LABEL} "
        f"({pre_meet - base_meet:+.1f} points).  \n"
        f"**1.5 s hard limit:** {PRE_LABEL} exceeds it for {pre_exc:.1f}% "
        f"vs {base_exc:.1f}% for {BASE_LABEL} "
        f"({pre_exc - base_exc:+.1f} points; lower is better)."
    )
    st.caption(
        "Failed requests are included in these counts."
        if include_failed_latency
        else "Counts use successful requests only; failed requests are "
             "excluded (turn on 'Include failed requests in latency' in "
             "the sidebar to count them)."
    )

    breakdown = []
    for label, frame in ((BASE_LABEL, baseline), (PRE_LABEL, preprocessed)):
        lat = latency_frame(frame)
        for name, part in lat.groupby("provider_label"):
            values = part["total_ms"]
            breakdown.append({
                "Provider / model": name, "Chatbot": label,
                "Responses": len(values),
                "Meet 1 s target": int((values <= TARGET_MS).sum()),
                "Over 1 s, within 1.5 s": int(
                    ((values > TARGET_MS) & (values <= HARD_LIMIT_MS)).sum()),
                "Exceed 1.5 s limit": int((values > HARD_LIMIT_MS).sum()),
                "% meeting target": format_pct(
                    (values <= TARGET_MS).mean() * 100),
            })
    if breakdown:
        with st.expander("Latency targets by provider and model"):
            st.dataframe(pd.DataFrame(breakdown), hide_index=True, **STRETCH)


# =========================================================
# 6. Detailed comparison export
# =========================================================

section(
    6, "Detailed Comparison Export",
    "CSV of matched questions with both answers, latency and token "
    "differences, and success status.",
    "#64748B",
)

if pairs.empty:
    no_pairs_message()
else:
    export = pd.DataFrame({
        "provider": pairs["provider_label_base"],
        "model_baseline": pairs["model_id_base"],
        "model_preprocessed": pairs["model_id_pre"],
        "chat_mode": pairs["mode_key"],
        "question_original": pairs["question_original_base"],
        "question_processed": pairs["question_processed_pre"],
        "baseline_answer": pairs["answer_base"],
        "preprocessed_answer": pairs["answer_pre"],
        "baseline_status": pairs["status_base"],
        "preprocessed_status": pairs["status_pre"],
        "baseline_error": pairs["error_type_base"],
        "preprocessed_error": pairs["error_type_pre"],
        "baseline_latency_ms": pairs["total_ms_base"],
        "preprocessed_latency_ms": pairs["total_ms_pre"],
        "latency_change_ms": pairs["latency_change_ms"],
        "latency_change_pct": pairs["latency_change_pct"],
        "latency_result": pairs["verdict"],
        "baseline_input_tokens": pairs["input_tokens_base"],
        "baseline_output_tokens": pairs["output_tokens_base"],
        "preprocessed_input_tokens": pairs["input_tokens_pre"],
        "preprocessed_output_tokens": pairs["output_tokens_pre"],
        "baseline_total_tokens": pairs["tokens_base"],
        "preprocessed_total_tokens": pairs["tokens_pre"],
        "token_change": pairs["token_change"],
        "token_change_pct": pairs["token_change_pct"],
        "baseline_timestamp": pairs["timestamp_utc_base"],
        "preprocessed_timestamp": pairs["timestamp_utc_pre"],
    })
    st.download_button(
        label=f"Download Matched Comparison ({len(export)} questions, CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="baseline_vs_preprocessed_matched.csv",
        mime="text/csv",
    )
    st.caption(
        "Latency values are raw milliseconds. Change columns are empty "
        "when either run failed, because a failed run's timing is not "
        "comparable."
    )

all_records = pd.concat(
    [baseline.assign(comparison_variant=BASE_LABEL),
     preprocessed.assign(comparison_variant=PRE_LABEL)],
    ignore_index=True,
).drop(columns=["ok", "fail", "known", "timestamp", "provider_key",
                "provider_label", "mode_key"], errors="ignore")
st.download_button(
    label="Download All Selected Records (CSV)",
    data=all_records.to_csv(index=False).encode("utf-8"),
    file_name="baseline_vs_preprocessed_all_records.csv",
    mime="text/csv",
    key="download_all",
)

st.caption(
    "All comparisons use saved CSV records. No tests are rerun."
)