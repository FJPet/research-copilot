from src.embeddings import load_embedding_model
from src.paper_profiles import build_all_profiles


def main() -> None:
    embedding_model = load_embedding_model()

    profiles = build_all_profiles(
        embedding_model=embedding_model,
    )

    print(
        f"\nCreated structured profiles "
        f"for {len(profiles)} papers."
    )

    print(
        "\nSaved to: data/paper_profiles.json"
    )


if __name__ == "__main__":
    main()