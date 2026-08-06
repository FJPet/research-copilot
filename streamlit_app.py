from typing import Any

import streamlit as st

from src.embeddings import load_embedding_model
from src.llm import generate_answer
from src.metadata import load_publications
from src.vector_store import (
    get_collection_size,
    search_all_papers,
    search_paper,
)


st.set_page_config(
    page_title="Chat with my Papers",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def get_embedding_model():
    """
    Load the local embedding model once.
    """

    return load_embedding_model()


@st.cache_data
def get_publications() -> list[dict[str, Any]]:
    """
    Load publication metadata once.
    """

    return load_publications()


def create_title_mapping(
    publications: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Map displayed publication titles to PDF filenames.
    """

    return {
        (
            f"{publication['year']} — "
            f"{publication['title']}"
        ): publication["source"]
        for publication in publications
    }


def format_passages_for_llm(
    passages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Keep only the fields required by the LLM.
    """

    return [
        {
            "source": passage["source"],
            "page": passage["page"],
            "chunk": passage["chunk"],
            "distance": passage["distance"],
            "text": passage["text"],
        }
        for passage in passages
    ]


def retrieve_paper_overview(
    source: str,
    model,
) -> list[dict[str, Any]]:
    """
    Retrieve passages covering the main parts of one paper.
    """

    queries = [
        "What is the research question, purpose, and contribution of this paper?",
        "Which methods and models are actually used by the authors?",
        "Which data and sample are used in the empirical analysis?",
        "What are the main empirical findings and conclusions?",
    ]

    unique_passages: dict[
        tuple[str, int, int],
        dict[str, Any],
    ] = {}

    for query in queries:
        passages = search_paper(
            query=query,
            source=source,
            model=model,
            number_of_results=3,
        )

        for passage in passages:
            identifier = (
                passage["source"],
                passage["page"],
                passage["chunk"],
            )

            unique_passages[identifier] = passage

    return sorted(
        unique_passages.values(),
        key=lambda passage: (
            passage["page"],
            passage["chunk"],
        ),
    )


def display_sources(
    passages: list[dict[str, Any]],
) -> None:
    """
    Display supporting passages with paper and page information.
    """

    st.markdown("### Sources")

    for rank, passage in enumerate(
        passages,
        start=1,
    ):
        label = (
            f"{rank}. {passage['source']} "
            f"— page {passage['page']}"
        )

        with st.expander(label):
            st.caption(
                f"Chunk {passage['chunk']} · "
                f"Cosine distance: "
                f"{passage['distance']:.4f}"
            )

            st.write(passage["text"])


def show_home(
    publications: list[dict[str, Any]],
    stored_chunks: int,
) -> None:
    """
    Display the landing page.
    """

    st.title("📚 Chat with my Papers")

    st.subheader(
        "Explore my scientific publications "
        "using semantic search and a local language model."
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

    st.markdown(
        """
        ### What you can do

        **Browse publications**  
        View titles, publication years, journals, and publication types.

        **Summarize a paper**  
        Select one paper and generate a grounded summary from its text.

        **Compare two papers**  
        Compare their research questions, methods, data, and findings.

        **Search the collection**  
        Find semantically relevant passages across all publications.
        """
    )

    st.caption(
        f"The local vector database currently contains "
        f"{stored_chunks} text chunks."
    )


def show_publications(
    publications: list[dict[str, Any]],
) -> None:
    """
    Display the structured publication catalogue.
    """

    st.header("📚 Publications")

    publication_type = st.selectbox(
        "Filter by publication type",
        [
            "All publications",
            "Journal articles",
            "Working papers",
        ],
    )

    filtered = publications

    if publication_type == "Journal articles":
        filtered = [
            publication
            for publication in publications
            if publication["publication_type"]
            == "Journal article"
        ]

    elif publication_type == "Working papers":
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

            journal = (
                publication["journal"]
                or "Not yet published"
            )

            col2.write(
                f"**Journal:** {journal}"
            )

            col3.write(
                f"**Type:** "
                f"{publication['publication_type']}"
            )


def show_summary(
    publications: list[dict[str, Any]],
    model,
) -> None:
    """
    Generate a summary of one selected paper.
    """

    st.header("📝 Summarize a Paper")

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
            "Retrieving relevant sections..."
        ):
            passages = retrieve_paper_overview(
                source=source,
                model=model,
            )

        question = """
Summarize this paper using the supplied passages.

Structure the answer under these headings:

1. Research question and motivation
2. Data
3. Methods actually used by the authors
4. Main findings
5. Contribution

Do not report methods that appear only in cited literature.
If one category is not supported by the passages, state that briefly.
"""

        with st.spinner(
            "Generating the summary locally..."
        ):
            answer = generate_answer(
                question=question,
                retrieved_passages=(
                    format_passages_for_llm(
                        passages
                    )
                ),
            )

        st.markdown("## Summary")
        st.write(answer)

        display_sources(passages)


def show_comparison(
    publications: list[dict[str, Any]],
    model,
) -> None:
    """
    Compare two selected papers.
    """

    st.header("⚖️ Compare Papers")

    title_mapping = create_title_mapping(
        publications
    )

    titles = list(title_mapping.keys())

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

        first_source = title_mapping[
            first_title
        ]

        second_source = title_mapping[
            second_title
        ]

        with st.spinner(
            "Retrieving both papers..."
        ):
            first_passages = (
                retrieve_paper_overview(
                    source=first_source,
                    model=model,
                )
            )

            second_passages = (
                retrieve_paper_overview(
                    source=second_source,
                    model=model,
                )
            )

            passages = (
                first_passages
                + second_passages
            )

        question = """
Compare the two papers represented in the supplied passages.

Use these headings:

1. Research questions
2. Data and empirical setting
3. Methods actually used
4. Main findings
5. Similarities
6. Differences

Keep the two papers clearly separated.
Do not attribute methods from cited literature to either paper.
"""

        with st.spinner(
            "Generating the comparison locally..."
        ):
            answer = generate_answer(
                question=question,
                retrieved_passages=(
                    format_passages_for_llm(
                        passages
                    )
                ),
            )

        st.markdown("## Comparison")
        st.write(answer)

        display_sources(passages)


def show_search(model) -> None:
    """
    Run transparent semantic search across all papers.
    """

    st.header("🔍 Search Papers")

    st.write(
        "Search for concepts, methods, datasets, "
        "topics, or findings across the collection."
    )

    query = st.text_input(
        "Search query",
        placeholder=(
            "For example: transfer entropy, "
            "volatility forecasting, zombie firms..."
        ),
    )

    number_of_results = st.slider(
        "Passages per paper",
        min_value=1,
        max_value=3,
        value=1,
    )

    if st.button(
        "Search collection",
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
            passages = search_all_papers(
                query=query,
                model=model,
                passages_per_paper=(
                    number_of_results
                ),
            )

        passages = sorted(
            passages,
            key=lambda passage: (
                passage["distance"]
            ),
        )

        st.markdown(
            f"## Most relevant passages"
        )

        display_sources(
            passages[:20]
        )


def show_about() -> None:
    """
    Explain the technical architecture.
    """

    st.header("ℹ️ About")

    st.markdown(
        """
        **Chat with my Papers** is a local
        Retrieval-Augmented Generation application
        for exploring a collection of scientific papers.

        ### Technical pipeline

        1. PDFs are read with **PyMuPDF**
        2. Extracted text is cleaned and split into overlapping chunks
        3. **BGE Small** creates local Transformer embeddings
        4. **ChromaDB** stores and retrieves the embedding vectors
        5. **Qwen 2.5 7B**, running through Ollama, generates summaries
           and comparisons from retrieved evidence
        6. **Streamlit** provides the interactive interface

        All embeddings and language-model inference run locally.
        No paid API service is required.
        """
    )


def main() -> None:
    publications = get_publications()
    stored_chunks = get_collection_size()

    if stored_chunks == 0:
        st.error(
            "The vector database is empty."
        )

        st.info(
            "Index the papers before starting "
            "the application."
        )
        return

    model = get_embedding_model()

    (
        tab_home,
        tab_publications,
        tab_summary,
        tab_comparison,
        tab_search,
        tab_about,
    ) = st.tabs(
        [
            "Home",
            "Publications",
            "Summarize",
            "Compare",
            "Search",
            "About",
        ]
    )

    with tab_home:
        show_home(
            publications=publications,
            stored_chunks=stored_chunks,
        )

    with tab_publications:
        show_publications(
            publications=publications,
        )

    with tab_summary:
        show_summary(
            publications=publications,
            model=model,
        )

    with tab_comparison:
        show_comparison(
            publications=publications,
            model=model,
        )

    with tab_search:
        show_search(
            model=model,
        )

    with tab_about:
        show_about()


if __name__ == "__main__":
    main()