import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import streamlit as st
from ollama import Client
from sentence_transformers import SentenceTransformer

from src.metadata import load_publications


MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_HOST = "http://localhost:11434"

INDEX_FOLDER = Path("search_index")
EMBEDDINGS_PATH = INDEX_FOLDER / "embeddings.npy"
CHUNKS_PATH = INDEX_FOLDER / "chunks.json"


st.set_page_config(
    page_title="Research Copilot",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """
    Load the local embedding model once.
    """

    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_search_index() -> tuple[
    np.ndarray,
    list[dict[str, Any]],
]:
    """
    Load the precomputed document embeddings and chunks.
    """

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            "The precomputed embedding index is missing."
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "The chunk metadata file is missing."
        )

    embeddings = np.load(
        EMBEDDINGS_PATH,
    )

    chunks = json.loads(
        CHUNKS_PATH.read_text(
            encoding="utf-8",
        )
    )

    if len(embeddings) != len(chunks):
        raise ValueError(
            "The numbers of embeddings and chunks do not match."
        )

    return embeddings, chunks


@st.cache_data
def get_publications() -> list[dict[str, Any]]:
    """
    Load structured publication metadata.
    """

    return load_publications()


def ollama_is_available() -> bool:
    """
    Check whether the local Ollama server is running.
    """

    try:
        with urlopen(
            f"{OLLAMA_HOST}/api/tags",
            timeout=1,
        ):
            return True

    except (URLError, TimeoutError, OSError):
        return False


def create_title_mapping(
    publications: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Map display titles to PDF filenames.
    """

    return {
        (
            f"{publication['year']} — "
            f"{publication['title']}"
        ): publication["source"]
        for publication in publications
    }


def semantic_search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    number_of_results: int = 10,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the precomputed embedding index.
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    ).astype("float32")

    similarities = embeddings @ query_embedding

    candidate_indices = np.argsort(
        similarities
    )[::-1]

    results: list[dict[str, Any]] = []

    for index in candidate_indices:
        chunk = chunks[int(index)]

        if source is not None:
            if chunk["source"] != source:
                continue

        results.append(
            {
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk": chunk["chunk"],
                "text": chunk["text"],
                "similarity": float(
                    similarities[index]
                ),
            }
        )

        if len(results) >= number_of_results:
            break

    return results


def retrieve_paper_overview(
    source: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Retrieve passages covering several dimensions of one paper.
    """

    queries = [
        "research question motivation and contribution",
        "data sample and empirical setting used by the authors",
        "methods models and estimation actually used by the authors",
        "main empirical findings results and conclusion",
    ]

    unique_results: dict[
        tuple[str, int, int],
        dict[str, Any],
    ] = {}

    for query in queries:
        results = semantic_search(
            query=query,
            model=model,
            embeddings=embeddings,
            chunks=chunks,
            number_of_results=3,
            source=source,
        )

        for result in results:
            identifier = (
                result["source"],
                result["page"],
                result["chunk"],
            )

            unique_results[identifier] = result

    return sorted(
        unique_results.values(),
        key=lambda result: (
            result["page"],
            result["chunk"],
        ),
    )


def format_context(
    passages: list[dict[str, Any]],
) -> str:
    """
    Format retrieved passages for the local language model.
    """

    blocks: list[str] = []

    for passage in passages:
        blocks.append(
            "\n".join(
                [
                    f"Paper: {passage['source']}",
                    f"Page: {passage['page']}",
                    f"Passage: {passage['text']}",
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)


def generate_local_answer(
    instruction: str,
    passages: list[dict[str, Any]],
) -> str:
    """
    Generate a grounded answer using local Qwen through Ollama.
    """

    context = format_context(passages)

    system_prompt = """
You are an assistant for exploring scientific publications.

Use only the supplied paper passages.

Rules:
- Do not invent facts.
- Do not attribute methods from cited literature to the focal paper.
- Report methods only when the passages indicate that the authors
  actually use, estimate, train, implement, or compare them.
- Mention uncertainty when the evidence is incomplete.
- Keep different papers clearly separated in comparisons.
- Write clearly for a technically informed reader.
"""

    client = Client(
        host=OLLAMA_HOST,
    )

    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"{instruction}\n\n"
                    f"Retrieved passages:\n{context}"
                ),
            },
        ],
        options={
            "temperature": 0.0,
        },
    )

    return response["message"]["content"]


def display_sources(
    passages: list[dict[str, Any]],
    heading: str = "Sources",
) -> None:
    """
    Show retrieved passages transparently.
    """

    st.markdown(f"### {heading}")

    for rank, passage in enumerate(
        passages,
        start=1,
    ):
        label = (
            f"{rank}. {passage['source']} "
            f"— page {passage['page']}"
        )

        with st.expander(
            label,
            expanded=(rank <= 2),
        ):
            st.caption(
                f"Chunk {passage['chunk']} · "
                f"Cosine similarity: "
                f"{passage['similarity']:.4f}"
            )

            st.write(
                passage["text"]
            )


def show_home(
    publications: list[dict[str, Any]],
    number_of_chunks: int,
    local_llm_available: bool,
) -> None:
    """
    Display the project landing page.
    """

    st.title("📚 Research Copilot")

    st.subheader("Chat with my Papers")

    st.write(
        "Explore my scientific publications using structured "
        "metadata, semantic search, and—when run locally—a "
        "local language model."
    )

    journal_articles = sum(
        publication["publication_type"]
        == "Journal article"
        for publication in publications
    )

    working_papers = sum(
        publication["publication_type"]
        == "Working paper"
        for publication in publications
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Publications",
        len(publications),
    )

    col2.metric(
        "Journal articles",
        journal_articles,
    )

    col3.metric(
        "Working papers",
        working_papers,
    )

    st.markdown("---")

    if local_llm_available:
        st.success(
            "Full local mode: semantic search, summaries, "
            "and comparisons are available."
        )
    else:
        st.info(
            "Public demo mode: publication browsing and semantic "
            "search are available. AI summaries and comparisons "
            "require the local Ollama version."
        )

    st.markdown(
        """
        ### Features

        **Browse publications**  
        Explore titles, years, journals, and publication types.

        **Semantic search**  
        Find passages on topics, methods, datasets, and findings.

        **Summarize a paper**  
        Available locally when Qwen is running through Ollama.

        **Compare papers**  
        Available locally when Qwen is running through Ollama.

        **Transparent evidence**  
        Search results and generated answers include paper and page references.
        """
    )

    st.caption(
        f"Search index: {number_of_chunks} text chunks"
    )


def show_publications(
    publications: list[dict[str, Any]],
) -> None:
    """
    Display the structured publication catalogue.
    """

    st.header("📚 Publications")

    selected_type = st.selectbox(
        "Publication type",
        [
            "All publications",
            "Journal articles",
            "Working papers",
        ],
    )

    filtered = publications

    if selected_type == "Journal articles":
        filtered = [
            publication
            for publication in publications
            if publication["publication_type"]
            == "Journal article"
        ]

    elif selected_type == "Working papers":
        filtered = [
            publication
            for publication in publications
            if publication["publication_type"]
            == "Working paper"
        ]

    for publication in filtered:
        with st.container(border=True):
            st.markdown(
                f"### {publication['title']}"
            )

            col1, col2, col3 = st.columns(
                [1, 2, 1]
            )

            col1.write(
                f"**Year:** {publication['year']}"
            )

            col2.write(
                f"**Journal:** "
                f"{publication['journal'] or 'Not yet published'}"
            )

            col3.write(
                f"**Type:** "
                f"{publication['publication_type']}"
            )


def show_search(
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
) -> None:
    """
    Display public semantic search.
    """

    st.header("🔍 Search the Papers")

    st.write(
        "Search for research topics, methods, datasets, "
        "markets, or empirical findings."
    )

    st.caption(
        "Examples: transfer entropy · volatility forecasting · "
        "machine learning · cryptocurrency price discovery · zombie firms"
    )

    query = st.text_input(
        "Search query",
        placeholder=(
            "For example: Which passages discuss "
            "volatility transmission?"
        ),
    )

    number_of_results = st.slider(
        "Number of passages",
        min_value=5,
        max_value=20,
        value=10,
    )

    if st.button(
        "Search",
        type="primary",
    ):
        if not query.strip():
            st.warning(
                "Please enter a search query."
            )
            return

        with st.spinner(
            "Searching the publication collection..."
        ):
            results = semantic_search(
                query=query,
                model=model,
                embeddings=embeddings,
                chunks=chunks,
                number_of_results=number_of_results,
            )

        st.markdown("## Results")
        display_sources(
            results,
            heading="Retrieved passages",
        )


def show_summary(
    publications: list[dict[str, Any]],
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    local_llm_available: bool,
) -> None:
    """
    Display the local summary workflow.
    """

    st.header("📝 Summarize a Paper")

    if not local_llm_available:
        st.warning(
            "Summaries require the local version with Ollama "
            f"and {OLLAMA_MODEL} running."
        )
        return

    title_mapping = create_title_mapping(
        publications
    )

    selected_title = st.selectbox(
        "Select a publication",
        list(title_mapping.keys()),
    )

    source = title_mapping[selected_title]

    if st.button(
        "Generate summary",
        type="primary",
    ):
        with st.spinner(
            "Retrieving relevant passages..."
        ):
            passages = retrieve_paper_overview(
                source=source,
                model=model,
                embeddings=embeddings,
                chunks=chunks,
            )

        instruction = """
Summarize the selected paper under these headings:

1. Research question and motivation
2. Data and empirical setting
3. Methods actually used by the authors
4. Main findings
5. Contribution

Do not report methods that occur only in the literature review.
If a category is not supported, state that briefly.
"""

        with st.spinner(
            "Generating the summary locally..."
        ):
            answer = generate_local_answer(
                instruction=instruction,
                passages=passages,
            )

        st.markdown("## Summary")
        st.write(answer)

        display_sources(passages)


def show_comparison(
    publications: list[dict[str, Any]],
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict[str, Any]],
    local_llm_available: bool,
) -> None:
    """
    Display the local paper-comparison workflow.
    """

    st.header("⚖️ Compare Papers")

    if not local_llm_available:
        st.warning(
            "Comparisons require the local version with Ollama "
            f"and {OLLAMA_MODEL} running."
        )
        return

    title_mapping = create_title_mapping(
        publications
    )

    titles = list(
        title_mapping.keys()
    )

    col1, col2 = st.columns(2)

    with col1:
        first_title = st.selectbox(
            "First paper",
            titles,
            index=0,
        )

    with col2:
        second_title = st.selectbox(
            "Second paper",
            titles,
            index=min(1, len(titles) - 1),
        )

    if st.button(
        "Compare papers",
        type="primary",
    ):
        if first_title == second_title:
            st.warning(
                "Please select two different papers."
            )
            return

        with st.spinner(
            "Retrieving both papers..."
        ):
            first_passages = retrieve_paper_overview(
                source=title_mapping[first_title],
                model=model,
                embeddings=embeddings,
                chunks=chunks,
            )

            second_passages = retrieve_paper_overview(
                source=title_mapping[second_title],
                model=model,
                embeddings=embeddings,
                chunks=chunks,
            )

            passages = (
                first_passages
                + second_passages
            )

        instruction = """
Compare the two selected papers under these headings:

1. Research questions
2. Data and empirical settings
3. Methods actually used
4. Main findings
5. Similarities
6. Differences

Keep the papers clearly separated.
Do not attribute cited methods to the focal papers.
"""

        with st.spinner(
            "Generating the comparison locally..."
        ):
            answer = generate_local_answer(
                instruction=instruction,
                passages=passages,
            )

        st.markdown("## Comparison")
        st.write(answer)

        display_sources(passages)


def show_about(
    local_llm_available: bool,
) -> None:
    """
    Explain the public and full local versions.
    """

    st.header("ℹ️ About")

    st.markdown(
        """
        **Research Copilot — Chat with my Papers** is a portfolio
        project for exploring a collection of scientific publications.

        ### Public mode

        - structured publication metadata;
        - precomputed document embeddings;
        - semantic search using cosine similarity;
        - transparent paper and page references.

        ### Full local mode

        - all public-mode features;
        - local Qwen summaries;
        - two-paper comparisons;
        - no paid AI API.

        ### Technology

        - Python
        - PyMuPDF
        - Sentence Transformers
        - BAAI/bge-small-en-v1.5
        - NumPy
        - Ollama and Qwen 2.5 7B
        - Streamlit
        """
    )

    if local_llm_available:
        st.success(
            "Ollama detected: full local features are enabled."
        )
    else:
        st.info(
            "Ollama was not detected: the app is running "
            "in public demo mode."
        )

    st.warning(
        "Work in progress: planned additions include intelligent "
        "question routing, structured method and dataset profiles, "
        "formula-aware processing, and a research knowledge graph."
    )


def main() -> None:
    publications = get_publications()

    try:
        embeddings, chunks = load_search_index()

    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))

        st.write(
            "Build the search index locally with:"
        )

        st.code(
            "python build_search_index.py"
        )
        return

    model = load_embedding_model()
    local_llm_available = ollama_is_available()

    (
        home_tab,
        publications_tab,
        search_tab,
        summary_tab,
        comparison_tab,
        about_tab,
    ) = st.tabs(
        [
            "Home",
            "Publications",
            "Search",
            "Summarize",
            "Compare",
            "About",
        ]
    )

    with home_tab:
        show_home(
            publications=publications,
            number_of_chunks=len(chunks),
            local_llm_available=local_llm_available,
        )

    with publications_tab:
        show_publications(
            publications=publications,
        )

    with search_tab:
        show_search(
            model=model,
            embeddings=embeddings,
            chunks=chunks,
        )

    with summary_tab:
        show_summary(
            publications=publications,
            model=model,
            embeddings=embeddings,
            chunks=chunks,
            local_llm_available=local_llm_available,
        )

    with comparison_tab:
        show_comparison(
            publications=publications,
            model=model,
            embeddings=embeddings,
            chunks=chunks,
            local_llm_available=local_llm_available,
        )

    with about_tab:
        show_about(
            local_llm_available=local_llm_available,
        )


if __name__ == "__main__":
    main()