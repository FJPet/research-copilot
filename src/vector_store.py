from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "research_papers"


def get_client():
    """
    Connect to the separately running local ChromaDB server.
    """

    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
    )


def create_collection():
    """
    Create or load the research-paper collection.
    """

    client = get_client()

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )


def get_collection_size() -> int:
    """
    Return the number of indexed chunks.
    """

    return create_collection().count()


def get_paper_names() -> list[str]:
    """
    Return all PDF filenames stored in ChromaDB.
    """

    result = create_collection().get(
        include=["metadatas"],
    )

    metadatas = result.get("metadatas") or []

    sources = {
        metadata["source"]
        for metadata in metadatas
        if metadata and "source" in metadata
    }

    return sorted(sources)


def clear_collection() -> None:
    """
    Delete the collection and create a fresh one.
    """

    client = get_client()

    try:
        client.delete_collection(
            name=COLLECTION_NAME,
        )
    except Exception:
        pass

    client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )


def add_documents(
    documents: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> None:
    """
    Embed and store document chunks in batches.
    """

    collection = create_collection()

    for batch_start in range(
        0,
        len(documents),
        batch_size,
    ):
        batch = documents[
            batch_start:batch_start + batch_size
        ]

        texts = [
            document["text"]
            for document in batch
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        ids = [
            (
                f"{document['source']}"
                f"_page_{document['page']}"
                f"_chunk_{document['chunk']}"
            )
            for document in batch
        ]

        metadatas = [
            {
                "source": document["source"],
                "page": document["page"],
                "chunk": document["chunk"],
            }
            for document in batch
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

        completed = min(
            batch_start + batch_size,
            len(documents),
        )

        print(
            f"Stored {completed} of "
            f"{len(documents)} chunks."
        )


def search_paper(
    query: str,
    source: str,
    model: SentenceTransformer,
    number_of_results: int = 3,
) -> list[dict[str, Any]]:
    """
    Search only within one selected paper.
    """

    collection = create_collection()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=number_of_results,
        where={
            "source": source,
        },
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    passages: list[dict[str, Any]] = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        passages.append(
            {
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": metadata["chunk"],
                "distance": float(distance),
                "text": document,
            }
        )

    return passages


def search_all_papers(
    query: str,
    model: SentenceTransformer,
    passages_per_paper: int = 1,
) -> list[dict[str, Any]]:
    """
    Search separately within every indexed paper.
    """

    all_passages: list[dict[str, Any]] = []

    for source in get_paper_names():
        passages = search_paper(
            query=query,
            source=source,
            model=model,
            number_of_results=passages_per_paper,
        )

        all_passages.extend(passages)

    return all_passages