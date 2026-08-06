from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_embedding_model() -> SentenceTransformer:
    """
    Load the local embedding model.
    """

    return SentenceTransformer(MODEL_NAME)


def create_embedding(
    text: str,
    model: SentenceTransformer,
) -> list[float]:
    """
    Convert text into an embedding vector.
    """

    vector = model.encode(
        text,
        normalize_embeddings=True,
    )

    return vector.tolist()