# 📚 Research Copilot

**Chat with my Papers**

Explore my scientific publications using structured metadata, semantic search, Retrieval-Augmented Generation, and local language models.

> 🚧 **Work in progress:** The current version provides a deployable semantic-search interface and an extended local mode with AI-generated summaries and paper comparisons.

---

## Live Demo

A public Streamlit demo is planned for:

`https://research-copilot-rxtnzmc8abdfjvan38nnvo.streamlit.app/`

The public version supports:

- browsing the publication catalogue;
- searching semantically across the complete paper collection;
- inspecting relevant passages with paper and page references.

The full local version additionally supports:

- structured summaries of individual papers;
- comparisons between two selected papers;
- local answer generation with Qwen through Ollama.

No paid AI API is required.

---

## Motivation

Scientific publications contain detailed information about research questions, data, methods, and findings, but reading an entire publication portfolio takes time.

Research Copilot provides an accessible way to explore my scientific work. It also serves as a practical AI-engineering project combining scientific document processing, Transformer embeddings, semantic retrieval, structured metadata, and local language-model inference.

The application deliberately separates reliable structured information from generative AI:

- publication titles, years, journals, and publication types come from curated metadata;
- semantic search retrieves relevant passages directly from the papers;
- local language-model generation is used only for controlled workflows such as single-paper summaries and two-paper comparisons.

---

## Features

###  Publication catalogue

Browse the publications by:

- title;
- publication year;
- journal;
- publication type.

This information is stored in a structured JSON file and does not depend on language-model inference.

###  Semantic search

Search across the complete publication collection for:

- research topics;
- statistical and machine-learning methods;
- datasets and empirical settings;
- financial markets and asset classes;
- findings and conclusions.

Each result includes:

- the PDF filename;
- page number;
- chunk number;
- cosine similarity;
- the original supporting passage.

###  Paper summaries — local mode

Select one paper and generate a structured summary covering:

1. research question and motivation;
2. data and empirical setting;
3. methods actually used by the authors;
4. main findings;
5. contribution.

###  Paper comparisons — local mode

Select two papers and compare their:

- research questions;
- datasets and empirical settings;
- methods;
- findings;
- similarities;
- differences.

###  Transparent evidence

Retrieved passages remain visible beneath search results and generated responses. This makes it possible to inspect the evidence and identify the original paper and page.

---

## Public and Local Modes

The same Streamlit application supports two operating modes.

### Public demo mode

When Ollama is unavailable, the app automatically provides:

- publication browsing;
- structured publication metadata;
- semantic search;
- source passages and page references.

This mode is designed for deployment on Streamlit Community Cloud.

### Full local mode

When a local Ollama server is detected, the app additionally enables:

- AI-generated paper summaries;
- comparisons between selected papers;
- grounded generation using retrieved paper passages.

The local version uses `qwen2.5:7b` and does not require a paid API.

---

## Architecture

### Shared document pipeline

```text
PDF publications
        │
        ▼
PyMuPDF text extraction
        │
        ▼
Custom text cleaning
        │
        ▼
Overlapping text chunks
        │
        ▼
BAAI/bge-small-en-v1.5
        │
        ▼
Precomputed embedding matrix
        │
        ▼
Cosine-similarity retrieval
        │
        ▼
Streamlit interface
```

### Optional local generation

```text
User selects a paper or comparison
        │
        ▼
Relevant passages are retrieved
        │
        ▼
Passages + controlled instruction
        │
        ▼
Qwen 2.5 7B through Ollama
        │
        ▼
Grounded summary or comparison
        │
        ▼
Supporting passages and page references
```

The deployment version uses a precomputed NumPy index rather than requiring a running vector-database service.

---

## Technology Stack

- **Python**
- **PyMuPDF** — PDF text extraction
- **Sentence Transformers**
- **BAAI/bge-small-en-v1.5** — local embedding model
- **NumPy** — precomputed embedding matrix and cosine-similarity retrieval
- **Streamlit** — interactive application
- **Ollama** — local LLM runtime
- **Qwen 2.5 7B** — local summaries and comparisons
- **ChromaDB** — used during earlier development and retained for experimental retrieval workflows

No paid AI service is required.

---

## Project Structure

```text
research-copilot/
├── data/
│   └── publications.json
├── papers/
│   └── publication PDFs
├── search_index/
│   ├── chunks.json
│   └── embeddings.npy
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── metadata.py
│   ├── pdf_loader.py
│   ├── text_cleaner.py
│   └── vector_store.py
├── tests/
├── app.py
├── build_search_index.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/FJPet/research-copilot.git
cd research-copilot
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```powershell
pip install -r requirements.txt
```

---

## Build the Search Index

The repository can include a precomputed search index for deployment. To rebuild it after changing the PDFs, preprocessing, or embedding model, run:

```powershell
python build_search_index.py
```

This creates:

```text
search_index/
├── embeddings.npy
└── chunks.json
```

The embedding matrix contains normalized vectors for the paper chunks. At query time, the application embeds the search query and ranks the stored chunks using cosine similarity.

---

## Run in Public Demo Mode

Ollama is not required for publication browsing and semantic search.

Run:

```powershell
streamlit run streamlit_app.py
```

The application will normally open at:

```text
http://localhost:8501
```

When Ollama is not detected, the summary and comparison tabs explain that these features are available only in full local mode.

---

## Enable Full Local Mode

### 1. Install Ollama

Install Ollama for your operating system.

### 2. Download Qwen

```powershell
ollama pull qwen2.5:7b
```

### 3. Start the Ollama server

```powershell
ollama serve
```

Leave this terminal open.

### 4. Start the application

In another terminal:

```powershell
streamlit run streamlit_app.py
```

The app detects the Ollama server at:

```text
http://localhost:11434
```

When detected, paper summaries and comparison features are enabled automatically.

---

## Deploy on Streamlit Community Cloud

Use the following deployment configuration:

```text
Repository: FJPet/research-copilot
Branch: main
Main file path: streamlit_app.py
```

The deployed application automatically runs in public demo mode because a local Ollama server is not available on Streamlit Community Cloud.

The following files must be committed:

```text
data/publications.json
search_index/chunks.json
search_index/embeddings.npy
papers/
```

The `search_index/` directory must therefore **not** be included in `.gitignore`.

---

## Example Searches

Try queries such as:

- `transfer entropy`
- `volatility forecasting`
- `machine learning`
- `cryptocurrency price discovery`
- `zombie firms`
- `Wikipedia searches and stock returns`
- `information flows between financial markets`
- `convolutional neural networks`
- `implied volatility`
- `market microstructure`

---

## Current Limitations

- Mathematical equations may not always be reconstructed perfectly from PDF text layers.
- Some special characters can be affected by PDF encoding.
- Character-based chunking may divide content across section or equation boundaries.
- Semantic search retrieves relevant passages but does not by itself guarantee a complete corpus-level answer.
- Compact local language models can misinterpret ambiguous passages.
- Summary generation can be slow on CPU-only systems.
- The current version intentionally avoids unrestricted corpus-wide generative chat.
- Public redistribution of individual PDFs depends on the rights associated with the included manuscript versions.

---

## Work in Progress

Planned improvements include:

- intelligent question routing;
- a structured research knowledge graph;
- curated method, dataset, and research-area profiles;
- section-aware and formula-aware chunking;
- improved distinction between methods used by the authors and methods mentioned in cited literature;
- faster local summarization;
- more reliable cross-paper synthesis;
- automated testing;
- Docker support;
- improved visual design and screenshots.

---

## Roadmap

- [x] PDF ingestion
- [x] Text extraction with PyMuPDF
- [x] Text cleaning
- [x] Overlapping chunk generation
- [x] Local Transformer embeddings
- [x] Semantic retrieval
- [x] Structured publication metadata
- [x] Precomputed deployment index
- [x] Streamlit interface
- [x] Local Qwen integration
- [x] Single-paper summary workflow
- [x] Two-paper comparison workflow
- [ ] Public Streamlit deployment
- [ ] Research knowledge graph
- [ ] Intelligent intent routing
- [ ] Structured method and dataset profiles
- [ ] Formula-aware document processing
- [ ] Automated tests
- [ ] Docker configuration

---

## Why This Project?

This project connects my academic research background with practical AI engineering.

It demonstrates:

- scientific document ingestion;
- PDF preprocessing;
- text chunking;
- Transformer embeddings;
- semantic search;
- Retrieval-Augmented Generation;
- local language-model deployment;
- structured metadata;
- transparent source attribution;
- Streamlit application development;
- hybrid public and local application design.

It also provides an interactive way to explore my publications without having to read every paper individually.

---

## Author

**Franziska J. Peter**

Research interests include financial econometrics, time-series analysis, market microstructure, volatility forecasting, information flow, cryptocurrency markets, machine learning, and explainable artificial intelligence.