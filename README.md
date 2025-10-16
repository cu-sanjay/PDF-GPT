# PDF‑GPT v2.0 · Chat With Your PDFs Using AI

> **What's New in v2.0:** Performance optimization with caching, export functionality, chat history and improved error handling.

[![Open on Render](https://img.shields.io/badge/Open%20App-Render-3a3a3a)](https://chatwithpdf-289m.onrender.com/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Version](https://img.shields.io/badge/version-2.0-blue)](https://github.com/cu-sanjay/PDF-GPT)

## Try now
<p align="center">
  <a href="https://chatwithpdf-289m.onrender.com/">
    <img src="https://github.com/user-attachments/assets/dcf43d6f-0509-4d64-911a-61feb2ddd3ea" alt="PDF‑GPT live demo" width="720" />
  </a>
</p>

## Overview

PDF‑GPT v2.0 is a professional Streamlit application that lets you ask questions about your PDF files and generate summaries, practice questions, MCQs, and study notes. It uses Google Gemini 2.0 Flash for language reasoning, LangChain for text processing, and FAISS for vector search. Version 2.0 features a complete UI overhaul, performance optimizations, and new export capabilities.

## Key Features

### Core Features

* Multiple PDF upload and processing
* Chat over documents with Gemini 2.0 Flash
* Vector search using FAISS for relevant answers
* **[NEW]** Chat history with conversation tracking
* **[NEW]** Session state management

### Study Tools

* Document summarization
* Question generation with answers
* MCQ generation
* Structured study notes
* Instant answers for quick lookups
* **[NEW]** Export functionality (download summaries, notes, MCQs as text files)

### v2.0 Improvements

* **Performance Optimization:** Model and embeddings caching for faster responses
* **Better UX:** Improved error messages and user feedback
* **Export Capability:** Download generated content as text files
* **Chat History:** View previous Q&A sessions
* **Code Quality:** Removed unused imports, cleaner code structure

## Architecture (high level)

* **UI**: Streamlit
* **PDF parsing**: PyPDF2 (text extraction)
* **Indexing**: LangChain text splitters + FAISS vector store
* **LLM**: Google Gemini (via google-generativeai and langchain-google-genai)
* **Config**: .env for secrets and environment settings

## Quick start

### 1) Clone and set up

```bash
git clone https://github.com/cu-sanjay/PDF-GPT.git
cd PDF-GPT

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment

Create a `.env` file in the project root:

```env
# Required
GOOGLE_API_KEY=your_google_ai_api_key

# Optional
GEMINI_MODEL=gemini-2.0-flash
EMBEDDINGS_MODEL=text-embedding-004
```

Get a free Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3) Run locally

```bash
# Option A
streamlit run app.py

# Option B (works even if streamlit is not on PATH)
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Deployment

### One‑click on Render

Use the button below to deploy your own copy on Render. Set the `GOOGLE_API_KEY` environment variable in the Render dashboard.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

**Start command** (Render):

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

**Environment variables** (Render):

* `GOOGLE_API_KEY` (required)
* `GEMINI_MODEL` (optional, default `gemini-2.0-flash`)
* `EMBEDDINGS_MODEL` (optional, default `text-embedding-004`)

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Create a new app on Streamlit Cloud and select this repo.
3. Main file: `app.py`.
4. Add secrets in **Settings → Secrets**:

```toml
GOOGLE_API_KEY = "your_google_ai_api_key"
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDINGS_MODEL = "text-embedding-004"
```

> Do not commit the real `.env`. Commit `.env.example` only.

## Commands

Local development:

```bash
python -m streamlit run app.py
```

Container or PaaS environments:

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
```

## Requirements

* Python 3.8 or later
* A valid Google AI API key

## Troubleshooting

* **GOOGLE\_API\_KEY not found**: create a `.env` file or set the variable on the host platform.
* **Streamlit not found**: run with `python -m streamlit run app.py` and ensure dependencies are installed.
* **PDF cannot be read**: the file may be image‑only or password protected.
* **No text extracted**: OCR is not included. Use text‑based PDFs or add OCR before upload.
* **Large files**: split large PDFs or process fewer files at a time.
* **Cold start on Render**: wait for the free instance to wake.

## Security and privacy

* PDFs are processed in memory during a session.
* Do not commit confidential files or keys.
* Review and follow your organisation policies when handling documents.

## Acknowledgements

The project uses Google Gemini, LangChain, FAISS, and Streamlit.
