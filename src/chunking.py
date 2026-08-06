from typing import Any


def split_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100,
    minimum_chunk_size: int = 200,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Small final chunks are merged into the previous chunk.
    Line breaks and mathematical symbols are preserved.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap must not be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    if minimum_chunk_size < 0:
        raise ValueError("minimum_chunk_size must not be negative.")

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end

        # Prefer splitting at a paragraph or line boundary.
        if proposed_end < len(text):
            search_start = start + chunk_size // 2
            possible_breaks = [
                text.rfind("\n\n", search_start, proposed_end),
                text.rfind("\n", search_start, proposed_end),
                text.rfind(". ", search_start, proposed_end),
                text.rfind(" ", search_start, proposed_end),
            ]

            valid_breaks = [
                position
                for position in possible_breaks
                if position > start
            ]

            if valid_breaks:
                end = max(valid_breaks)

                # Include the punctuation or line break.
                if text[end:end + 2] == ". ":
                    end += 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    # Avoid a very small final chunk.
    if (
        len(chunks) >= 2
        and len(chunks[-1]) < minimum_chunk_size
    ):
        chunks[-2] = (
            chunks[-2]
            + "\n\n"
            + chunks[-1]
        ).strip()
        chunks.pop()

    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 800,
    overlap: int = 100,
    minimum_chunk_size: int = 200,
) -> list[dict[str, Any]]:
    """
    Split page-level documents into chunks while preserving metadata.
    """

    chunked_documents: list[dict[str, Any]] = []

    for document in documents:
        text_chunks = split_text(
            text=document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            minimum_chunk_size=minimum_chunk_size,
        )

        for chunk_index, chunk_text in enumerate(
            text_chunks,
            start=1,
        ):
            chunked_documents.append(
                {
                    "source": document["source"],
                    "page": document["page"],
                    "chunk": chunk_index,
                    "text": chunk_text,
                }
            )

    return chunked_documents