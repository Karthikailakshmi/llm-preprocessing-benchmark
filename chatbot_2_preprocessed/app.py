"""Chatbot 2: WITH CUSTOM PREPROCESSING (blue and white).
Run from this folder:  streamlit run app.py --server.port 8502"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from chatbot_2_preprocessed.preprocessing import PreprocessConfig, PreprocessingPipeline  # noqa: E402
from shared.ui import render_app  # noqa: E402

THEME = {"primary": "#1D4ED8", "dark": "#1E3A8A", "soft": "#EFF6FF", "line": "#BFDBFE", "icon": "🧪"}


def build_pipeline() -> PreprocessingPipeline:
    with st.expander("Preprocessing techniques", expanded=False):
        cb = lambda label, key, help_=None, **kw: st.checkbox(label, value=kw.pop("value", True),  # noqa: E731
                                                              key=key, help=help_, **kw)
        cfg = PreprocessConfig(
            text_cleaning=cb("Text cleaning", "t_clean", "Removes control characters and extra spaces; keeps code."),
            normalization=cb("Text normalization", "t_norm", "Unicode NFC form and unusual spaces."),
            tokenization=cb("Tokenization (count)", "t_tok", "Approximate token count of the query."),
            data_filtering=cb("Data filtering", "t_filter", "Blocks empty or malformed input; warns on repetition."),
            deduplication=cb("Deduplication", "t_dedup", "Removes accidental repeats and duplicate chunks."),
            prompt_formatting=cb("Prompt formatting", "t_fmt", "Structured system, context and question layout."),
            truncation=cb("Truncation", "t_trunc", "Keeps input within the model limit, best content first."),
            chunking=cb("Chunking", "t_chunk", "Sentence-aware chunks. Process documents again after changing."),
            padding=cb("Padding (not used for API calls)", "t_pad", None, value=False, disabled=True),
        )
        st.caption("Dataset formatting is an offline step used by the evaluation scripts.")
    return PreprocessingPipeline(cfg)


render_app(
    variant="preprocessed",
    title="Preprocessing chatbot",
    tagline="Your question is cleaned and structured before it reaches the model.",
    theme=THEME,
    build_pipeline=build_pipeline,
)
