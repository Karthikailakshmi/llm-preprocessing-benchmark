# Comparative Analysis of LLM Chatbots With and Without Custom Preprocessing

Two independent Streamlit chatbots with identical features, models, retrieval and
generation settings. Only the preprocessing differs.

| | Chatbot 1 (baseline) | Chatbot 2 (preprocessed) |
|---|---|---|
| Theme | purple and white | blue and white |
| Folder | `chatbot_1_baseline/` | `chatbot_2_preprocessed/` |
| Port | 8501 | 8502 |
| Query handling | original query + common system prompt | nine configurable techniques (`preprocessing.py`) |

## Status

Built and tested so far (Stages 2 to 8): both chatbots, five provider integrations,
document Q&A (RAG), the nine preprocessing functions, per-request logging, the
evaluation dataset (26 questions), and an offline test suite.
**Not built yet:** `evaluation/run_evaluation.py` (Stage 9), `compare_results.py` and
`dashboard.py` (Stage 10). The latter three are clearly marked stubs.

## Setup

Windows PowerShell shown. On macOS/Linux use `python3` and `source .venv/bin/activate`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env          # then add your API keys
python verify_models.py         # lists model IDs your keys can use
python test_providers.py        # one tiny request per provider
python smoke_test.py            # offline checks, no keys needed
```

OCR (optional): install the Tesseract program, then `pytesseract` (already in
requirements) can read scanned PDFs and PNG/JPG files. Without it the app tells you.

## Run

```powershell
cd chatbot_1_baseline
streamlit run app.py --server.port 8501
```
```powershell
cd chatbot_2_preprocessed
streamlit run app.py --server.port 8502
```
Run each from inside its own folder so its `.streamlit/config.toml` theme applies.

Or start both from one terminal: `python run_both.py` (Ctrl+C stops both).

## Providers and models

One distinct model family per provider. IDs live in `shared/config.py` and can be
overridden in `.env` (e.g. `GROQ_MODEL=`). Confirm them with `verify_models.py`.

| Provider | Default model | Notes |
|---|---|---|
| Gemini | `gemini-2.5-flash` | free-tier inputs may be used by Google |
| NVIDIA | `nvidia/nvidia-nemotron-nano-9b-v2` | ~40 requests/min |
| Groq | `meta-llama/llama-4-scout-17b-16e-instruct` | |
| Cerebras | `gpt-oss-120b` | trial credit; card required |
| OpenRouter | set `OPENROUTER_MODEL` | choose a `:free` model; ~50 requests/day unless $10 credit bought |

Requests are paced per provider to respect free-tier limits. Waiting time is logged
as `rate_limit_wait_ms` and excluded from `api_ms`. There are no automatic retries;
use the Retry button.

## Supported files

PDF, DOCX, TXT, MD, CSV, XLSX, PPTX, HTML, JSON, PNG, JPG/JPEG (images and scanned PDF
pages need OCR). Other types are rejected with a message. PDF tables are read as page
text; DOCX/PPTX/XLSX/CSV tables are extracted as rows.

## Fair comparison

Shared by both bots (`shared/`): document parsing and cleaning, embedding model,
chunk size and overlap, top-k, system prompts, generation parameters, provider config,
conversation history and logging. Preprocessing affects only chatbot 2. Chunking
differs by design: baseline uses fixed windows, chatbot 2 uses sentence-aware chunks
with the same size and overlap. Turn each technique off in the sidebar to test its
individual contribution.

Folder layout differs from the suggested one on purpose: `llm_client`, `document_loader`,
`retrieval` and `metrics` are single shared modules so the two bots cannot drift apart.

## Logging

Every request, successful or failed, is appended to
`evaluation/results/<variant>_interactive.csv` with total, preprocessing, retrieval and
API durations (ms), provider-reported tokens (blank if unavailable), error type, sources,
and the enabled techniques. Logs contain your questions and answers locally; they are
git-ignored. API keys are never logged.

## Limitations

- Token counts shown by the preprocessing step are approximate; provider-reported usage is authoritative.
- The input limit is a uniform 6000 tokens for every model (override per provider with `<PROVIDER>_MAX_INPUT_TOKENS`).
- The first document run downloads the embedding model; if that fails the app falls back to TF-IDF and says so.
- Free tiers change; re-check limits before long runs.
