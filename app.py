from pathlib import Path

from src.metadata import load_publications


PAPERS_PATH = Path("papers")


def main() -> None:
    publications = load_publications()

    pdf_names = {
        path.name
        for path in PAPERS_PATH.glob("*.pdf")
    }

    metadata_names = {
        publication["source"]
        for publication in publications
    }

    missing_pdfs = sorted(
        metadata_names - pdf_names
    )

    missing_metadata = sorted(
        pdf_names - metadata_names
    )

    print(
        f"\nMetadata records: {len(publications)}"
    )

    print(
        f"PDF files: {len(pdf_names)}"
    )

    if missing_pdfs:
        print(
            "\nMetadata entries without matching PDF:"
        )

        for filename in missing_pdfs:
            print(f"- {filename}")

    if missing_metadata:
        print(
            "\nPDFs without metadata:"
        )

        for filename in missing_metadata:
            print(f"- {filename}")

    if not missing_pdfs and not missing_metadata:
        print(
            "\nAll PDF filenames match "
            "the publication metadata."
        )


if __name__ == "__main__":
    main()