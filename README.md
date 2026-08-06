# Research Copilot

**Chat with my Papers**

Explore my scientific publications using structured metadata, semantic search, Retrieval-Augmented Generation (RAG), and local language models.

**Live Demo:**  
[Research Copilot](https://research-copilot-rxtnzmc8abdfjvan38nnvo.streamlit.app/)

**Source Code:**  
[GitHub Repository](https://github.com/FJPet/research-copilot)

---

> **Work in Progress**
>
> Research Copilot is under active development. The current version provides a deployable semantic-search interface together with an extended local mode supporting AI-generated paper summaries and comparisons using Ollama and Qwen 2.5.

---

## Motivation

Scientific publications contain detailed information about research questions, data, methods, and findings, but reading an entire publication portfolio takes time.

Research Copilot provides an interactive way to explore my scientific work. At the same time, it serves as a practical AI engineering project combining scientific document processing, Transformer embeddings, semantic retrieval, structured metadata, and local language models.

The application deliberately separates reliable structured information from generative AI:

- publication titles, years, journals, and publication types come from curated metadata;
- semantic search retrieves relevant passages directly from the papers;
- local language-model generation is used only for controlled workflows such as single-paper summaries and paper comparisons.

---

## Features

### Publication Catalogue

Browse the publication portfolio by:

- title;
- publication year;
- journal;
- publication type.

Publication metadata are stored in a structured JSON file rather than inferred by a language model.

### Semantic Search

Search across the complete paper collection for:

- research topics;
- statistical and machine-learning methods;
- datasets;
- empirical findings;
- financial markets;
- asset classes.

Each result displays:

- paper title;
- page number;
- similarity score;
- supporting passage.

### Paper Summaries (Local Mode)

Generate structured summaries covering:

1. research question;
2. data and empirical setting;
3. methods actually used;
4. main findings;
5. contribution.

### Paper Comparisons (Local Mode)

Compare two selected papers with respect to:

- research question;
- datasets;
- methodology;
- findings;
- similarities;
- differences.

### Transparent Evidence

Every generated answer is grounded in retrieved passages that remain visible to the user.

---

## Public and Local Modes

The application supports two execution modes.

### Public Demo

The public Streamlit version provides:

- publication browsing;
- semantic search;
- transparent source passages.

No paid AI API is required.

### Local Version

When Ollama is running locally, the application additionally enables:

- AI-generated summaries;
- paper comparisons;
- Retrieval-Augmented Generation using Qwen 2.5.

---

## Architecture

### Shared Retrieval Pipeline

```text
PDF publications
        │
        ▼
PyMuPDF
        │
        ▼
Text cleaning
        │
        ▼
Overlapping chunks
        │
        ▼
Sentence Transformers
(BAAI/bge-small-en-v1.5)
        │
        ▼
Precomputed embedding index
        │
        ▼
Cosine similarity search
        │
        ▼
Streamlit interface
```

### Local AI Pipeline

```text
Retrieved passages
        │
        ▼
Qwen 2.5
via Ollama
        │
        ▼
Grounded summaries
and comparisons
```

---

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- BAAI/bge-small-en-v1.5
- NumPy
- Ollama
- Qwen 2.5
- ChromaDB (development version)

No paid AI service is required.

---

## Project Structure

```text
research-copilot/
├── data/
│   └── publications.json
├── papers/
├── search_index/
│   ├── embeddings.npy
│   └── chunks.json
├── src/
├── tests/
├── app.py
├── build_search_index.py
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/FJPet/research-copilot.git
cd research-copilot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Build the Search Index

```bash
python build_search_index.py
```

This generates:

```text
search_index/
├── embeddings.npy
└── chunks.json
```

---

## Run the Application

Public mode:

```bash
streamlit run streamlit_app.py
```

Enable the local AI features:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Then start the application:

```bash
streamlit run streamlit_app.py
```

---

## Example Queries

- transfer entropy
- volatility forecasting
- machine learning
- cryptocurrency price discovery
- zombie firms
- implied volatility
- market microstructure
- information flow
- convolutional neural networks

---

## Current Limitations

- Mathematical expressions extracted from PDFs are sometimes imperfect.
- Character-based chunking can split sections.
- Local language models occasionally misinterpret ambiguous passages.
- Summary generation is slower on CPU-only systems.
- The current version intentionally avoids unrestricted corpus-wide chat.

---

## Roadmap

- [x] PDF ingestion
- [x] Text cleaning
- [x] Semantic search
- [x] Structured metadata
- [x] Local LLM integration
- [x] Paper summaries
- [x] Paper comparisons
- [x] Public Streamlit demo
- [ ] Research knowledge graph
- [ ] Intelligent question routing
- [ ] Structured method profiles
- [ ] Formula-aware processing
- [ ] Automated tests
- [ ] Docker support

---

## Why This Project?

Research Copilot combines my academic background in econometrics and machine learning with modern AI engineering techniques.

The project demonstrates:

- scientific document processing;
- Transformer embeddings;
- semantic retrieval;
- Retrieval-Augmented Generation;
- local language-model deployment;
- interactive application development;
- transparent evidence-based AI.

It also provides an intuitive way to explore my publication portfolio without reading every paper individually.

---

## Author

**Franziska J. Peter**

Research interests include financial econometrics, time-series analysis, market microstructure, volatility forecasting, information flow, cryptocurrency markets, machine learning, and explainable artificial intelligence.