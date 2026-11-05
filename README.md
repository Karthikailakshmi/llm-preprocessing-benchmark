# Comparative Analysis of LLM Chatbots With and Without Custom Preprocessing

A comparative study of two independent Streamlit chatbots designed to evaluate how custom query preprocessing affects response quality, latency, and token efficiency across five LLM configurations.

Both chatbots share the same core functionality, document retrieval pipeline, generation settings, and evaluation approach. The primary difference is how user queries are processed before being sent to the language model.

## Chatbot Comparison

| Feature | Chatbot 1 — Baseline | Chatbot 2 — Preprocessed |
|---|---|---|
| Theme | Purple and white | Blue and white |
| Directory | `chatbot_1_baseline/` | `chatbot_2_preprocessed/` |
| Default port | `8501` | `8502` |
| Query handling | Original query with a common system prompt | Nine configurable preprocessing techniques |
| Preprocessing | No custom query preprocessing | Implemented in `preprocessing.py` |

## Project Status

### Implemented

- Two independent Streamlit chatbot applications
- Five LLM configurations across three providers
- Document question answering using Retrieval-Augmented Generation (RAG)
- Nine configurable query preprocessing techniques
- Per-request logging for performance analysis
- Evaluation dataset containing 26 questions
- Offline test suite
- Shared modules for document processing, retrieval, LLM communication, and metrics

### Pending Implementation

The following evaluation components are currently stubs and are not yet fully implemented:

- `evaluation/run_evaluation.py` — automated evaluation execution
- `evaluation/compare_results.py` — comparison of baseline and preprocessed results
- `evaluation/dashboard.py` — visualization of evaluation results

Evaluation and comparison results should be considered preliminary until these components are implemented and tested.

## Technology Stack

- **Programming language:** Python
- **Frontend:** Streamlit
- **LLM providers:** Google Gemini, NVIDIA, and Groq
- **Document processing:** Multiple document formats with optional OCR
- **Retrieval:** Shared document-processing and retrieval pipeline
- **Evaluation metrics:** Response latency, token usage, preprocessing time, retrieval time, and request outcomes

## LLM Providers and Models

The project uses five LLM configurations across three providers.

| No. | Provider | Model ID | Environment variable |
|---:|---|---|---|
| 1 | Google Gemini | `gemini-3.5-flash-lite` | As configured in `shared/config.py` |
| 2 | NVIDIA | `nvidia/nemotron-3.5-bolt-e4b` | As configured in `shared/config.py` |
| 3 | Groq | `openai/gpt-oss-20b` | `GROQ_MODEL` |
| 4 | Groq | `qwen/qwen3.8-27b` | `GROQ_MODEL_2` |
| 5 | Groq | `openai/gpt-oss-120b` | `GROQ_MODEL_3` |

Model availability and exact API identifiers should be verified against the respective provider's current API.

## Prerequisites

- Python installed and available from the terminal
- API keys for the providers you intend to use
- Internet connectivity for API requests
- Tesseract OCR installed separately if scanned-document OCR is required

## Installation and Setup

The commands below use Windows PowerShell. On macOS or Linux, use `python3` where appropriate and activate the virtual environment with `source .venv/bin/activate`.

### 1. Clone the repository

```powershell
git clone https://github.com/Karthikailakshmi/llm-preprocessing-benchmark.git
cd llm-preprocessing-benchmark
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution-policy setting for your environment or activate the environment through your preferred terminal.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a template, if that template is available.

Add your provider API keys and configure the Groq model IDs as follows:

```dotenv
GROQ_MODEL=openai/gpt-oss-20b
GROQ_MODEL_2=qwen/qwen3.8-27b
GROQ_MODEL_3=openai/gpt-oss-120b
```

Configure the Gemini and NVIDIA model IDs using the variable names supported by `shared/config.py`.

**Security:** Never commit `.env` or API keys to GitHub. The repository's `.gitignore` should exclude local credentials.

### 5. Verify model configuration

```powershell
python verify_models.py
```

Use this script to check model configuration and availability as supported by your implementation.

### 6. Run the offline test suite

```powershell
python smoke_test.py
```

The smoke tests are intended to run without API keys. Provider-specific tests may require valid credentials and network access.

## Running the Chatbots

Run each chatbot from its own directory so its Streamlit theme configuration is applied.

### Chatbot 1 — Baseline

```powershell
cd chatbot_1_baseline
streamlit run app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Chatbot 2 — Preprocessed

Open another terminal from the project root and run:

```powershell
cd chatbot_2_preprocessed
streamlit run app.py --server.port 8502
```

Open [http://localhost:8502](http://localhost:8502) in your browser.

### Run both chatbots together

Alternatively, launch both applications from the project root:

```powershell
python run_both.py
```

Press `Ctrl+C` to stop the processes.

## Supported Document Formats

The document question-answering functionality supports:

- PDF
- DOCX
- TXT
- Markdown
- CSV
- XLSX
- PPTX
- HTML
- JSON
- PNG
- JPG and JPEG

OCR is required to extract text from images and scanned PDF pages. Install Tesseract OCR separately and ensure it is available on your system PATH. The Python `pytesseract` dependency is included in the requirements.

Unsupported file formats are rejected with an explanatory message.

PDF tables are extracted as page text. Tables in DOCX, PPTX, XLSX, and CSV files are processed according to the corresponding extraction logic.

## Fair Comparison Methodology

The two chatbots share the same core modules under `shared/`, including:

- Document parsing and cleaning
- Embedding configuration
- Retrieval settings, including top-k
- System prompts and generation parameters
- Provider configuration
- Conversation history handling
- Metrics and request logging

The principal experimental difference is custom query preprocessing in Chatbot 2.

Chunking differs intentionally: the baseline uses fixed-size windows, while the preprocessed chatbot uses sentence-aware chunks with the same configured chunk size and overlap.

The preprocessing techniques can be individually disabled through the sidebar to investigate their contributions.

This design aims to minimize unrelated implementation differences so the impact of query preprocessing can be evaluated more consistently.

## Logging and Performance Metrics

Each request, successful or unsuccessful, is logged locally under `evaluation/results/`.

The logs can include:

- Total request duration
- Preprocessing duration
- Retrieval duration
- API duration
- Provider-reported token usage, when available
- Error information
- Retrieved sources
- Enabled preprocessing techniques

Time spent waiting for provider rate limits is recorded as `rate_limit_wait_ms` and excluded from `api_ms`.

Preprocessing token counts are approximate; provider-reported usage is preferred when available. Automatic retries are not implemented, so use the application's Retry option when appropriate.

### Privacy and data handling

Interactive logs may contain user questions, generated answers, and retrieved-source information. Keep these files private if they contain sensitive data.

API keys should never be logged or committed to the repository. Review chat-history files and generated evaluation outputs before publishing the project.

## Limitations

- Automated evaluation, result comparison, and dashboard visualization are still pending implementation.
- The evaluation dataset currently contains 26 questions.
- Preprocessing token counts are estimates; provider-reported usage is authoritative when available.
- The default input limit is 6,000 tokens across models, with provider-specific overrides supported through configuration.
- The first document-processing run may download an embedding model.
- If embedding-model loading fails, the application may fall back to TF-IDF retrieval and display a notification.
- Provider availability, model identifiers, pricing, rate limits, and free-tier policies may change.
- Responses and measured latency can vary depending on provider load, network conditions, model behavior, and configuration.
- Results should not be interpreted as conclusive until the automated evaluation pipeline is implemented and run.

## Future Work

- Implement automated evaluation using the 26-question dataset.
- Compare baseline and preprocessed chatbot response quality, latency, and token efficiency.
- Build a dashboard for visualizing evaluation results.
- Analyze the contribution of each preprocessing technique.
- Expand testing across query types, document collections, and model configurations.
- Improve reproducibility through consistent test conditions and recorded configuration settings.

## Repository Structure

```text
llm-preprocessing-benchmark/
├── chatbot_1_baseline/
│   ├── .streamlit/
│   ├── app.py
│   └── __init__.py
├── chatbot_2_preprocessed/
│   ├── .streamlit/
│   ├── app.py
│   ├── preprocessing.py
│   └── __init__.py
├── data/
│   └── chat_history/
├── evaluation/
│   ├── sample_docs/
│   ├── results/
│   ├── run_evaluation.py
│   ├── compare_results.py
│   └── dashboard.py
├── shared/
│   ├── config.py
│   ├── document_loader.py
│   ├── llm_client.py
│   ├── metrics.py
│   ├── prompts.py
│   ├── retrieval.py
│   └── ui.py
├── .gitignore
├── README.md
├── requirements.txt
├── run_both.py
├── smoke_test.py
└── verify_models.py
```

Additional scripts and files may be present in the repository. The evaluation files listed above include planned components that are not yet fully implemented.

## Author

**Karthikailakshmi**

Developed as a comparative study of LLM chatbot behavior with and without custom query preprocessing.
