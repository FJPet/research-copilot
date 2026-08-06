from pathlib import Path

import fitz  # PyMuPDF


def load_pdfs(folder: str = "papers") -> list[dict]:
    """
    Load all PDF files from a folder.

    Returns one dictionary per PDF page with:
    - source: PDF file name
    - page: page number
    - text: extracted page text
    """

    pdf_folder = Path(folder)

    if not pdf_folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {pdf_folder.resolve()}"
        )

    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in: {pdf_folder.resolve()}"
        )

    documents: list[dict] = []

    for pdf_path in pdf_files:
        try:
            with fitz.open(pdf_path) as pdf_document:
                for page_index in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_index)
                    text = page.get_text("text").strip()

                    if not text:
                        continue

                    documents.append(
                        {
                            "source": pdf_path.name,
                            "page": page_index + 1,
                            "text": text,
                        }
                    )

        except Exception as error:
            print(f"Could not process {pdf_path.name}: {error}")

    return documents