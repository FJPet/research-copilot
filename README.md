# Chat with my Papers

An interactive application for exploring my scientific publications using semantic search, local language models, and Retrieval-Augmented Generation (RAG).

The application provides a structured publication catalogue and allows users to:

- browse publications by year, journal, and publication type;
- semantically search across the paper collection;
- summarize an individual paper;
- compare two selected papers;
- inspect the supporting passages and page references used for an answer.

The project runs locally without a paid API service.

---

## Work in Progress

> **Work in progress.**

The current version demonstrates a fully local Retrieval-Augmented Generation pipeline for exploring a collection of scientific publications using semantic search, ChromaDB, Ollama, and Streamlit.

Planned improvements include:

- a structured research knowledge graph;
- intelligent question routing;
- structured information about methods, datasets, and research areas;
- improved handling of mathematical formulas;
- faster local retrieval and summarization;
- more reliable cross-paper comparisons;
- automated tests and additional deployment options.

The present version deliberately uses controlled workflows—publication browsing, single-paper summaries, two-paper comparisons, and transparent semantic search—instead of unrestricted corpus-wide chat. This improves reliability when using a compact local language model.

---

## Current Features

### Publication catalogue

Publication metadata are stored in a structured JSON file and include:

- title;
- publication year;
- journal;
- publication type;
- associated PDF filename.

This enables reliable factual views without relying on a language model to infer publication metadata.

### Paper summaries

Users select one publication. The application retrieves passages covering:

- research question and motivation;
- data;
- methods;
- findings;
- contribution.

A local language model then generates a structured summary based on the retrieved evidence.

### Paper comparison

Users select two publications and receive a comparison of:

- research questions;
- data and empirical settings;
- methods;
- findings;
- similarities;
- differences.

### Semantic search

The application searches all papers using local Transformer embeddings and displays relevant passages with:

- PDF filename;
- page number;
- chunk number;
- cosine distance.

---

## Architecture

```text
PDF publications
      │
      ▼
PyMuPDF text extraction
      │
      ▼
Text cleaning
      │
      ▼
Overlapping text chunks
      │
      ▼
BGE-small Transformer embeddings
      │
      ▼
ChromaDB vector database
      │
      ▼
Semantic retrieval
      │
      ▼
Qwen 2.5 via Ollama
      │
      ▼
Streamlit interface