import re


NOISE_PHRASES = [
    "Preprint not peer reviewed",
    "This preprint research paper has not been peer reviewed.",
    "Electronic copy available at:",
]


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving paragraphs,
    formulas, and mathematical symbols as much as possible.
    """

    cleaned = text

    for phrase in NOISE_PHRASES:
        cleaned = cleaned.replace(phrase, "")

    # Join words split across lines with a hyphen.
    cleaned = re.sub(r"(\w)-\n(\w)", r"\1\2", cleaned)

    # Remove lines that contain only a page number.
    cleaned = re.sub(
        r"(?m)^\s*\d+\s*$",
        "",
        cleaned,
    )

    # Remove unnecessary spaces before punctuation.
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

    # Replace repeated spaces and tabs.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Preserve paragraphs but remove excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()