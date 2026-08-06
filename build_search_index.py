import json
from pathlib import Path

import numpy as np

from src.chunking import chunk_documents
from src.embeddings import load_embedding_model
from src.pdf_loader import load_pdfs
from src.text_cleaner import clean_text


INDEX_FOLDER = Path("search_index")
EMBEDDINGS_PATH = INDEX_FOLDER / "embeddings.npy"
CHUNKS_PATH = INDEX_FOLDER / "chunks.json"


def prepare_chunks() -> list[dict]:
    """
    Load, clean, and chunk all PDF pages.
    """

    pages = load_pdfs()
    cleaned_pages: list[dict] = []

    for page in pages:
        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        cleaned_pages.append(
            {
                "source": page["source"],
                "page": page["page"],
                "text": cleaned_text,
            }
        )

    return chunk_documents(
        documents=cleaned_pages,
        chunk_size=800,
        overlap=100,
        minimum_chunk_size=200,
    )


def main() -> None:
    INDEX_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = prepare_chunks()

    print(f"Created {len(chunks)} chunks.")

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings.astype("float32"),
    )

    CHUNKS_PATH.write_text(
        json.dumps(
            chunks,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Saved embeddings to: {EMBEDDINGS_PATH}")
    print(f"Saved chunk data to: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()