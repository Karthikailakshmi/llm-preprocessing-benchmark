"""Chatbot 1: BASELINE (purple and white). No custom preprocessing.
Run from this folder:  streamlit run app.py --server.port 8501"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.prompts import BaselinePipeline  # noqa: E402
from shared.ui import render_app  # noqa: E402

THEME = {"primary": "#6D28D9", "dark": "#4C1D95", "soft": "#F5F3FF", "line": "#DDD6FE", "icon": "💬"}

render_app(
    variant="baseline",
    title="Baseline chatbot",
    tagline="Your question goes to the model as written. No preprocessing.",
    theme=THEME,
    build_pipeline=BaselinePipeline,
)
