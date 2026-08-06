import json
from pathlib import Path
from typing import Any


PUBLICATIONS_PATH = Path("data/publications.json")


def load_publications() -> list[dict[str, Any]]:
    """
    Load structured publication metadata.
    """

    if not PUBLICATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {PUBLICATIONS_PATH.resolve()}"
        )

    publications = json.loads(
        PUBLICATIONS_PATH.read_text(encoding="utf-8")
    )

    return sorted(
        publications,
        key=lambda publication: (
            publication["year"] is None,
            -(publication["year"] or 0),
            publication["title"],
        ),
    )


def get_publication_by_source(
    source: str,
) -> dict[str, Any] | None:
    """
    Find publication metadata using its PDF filename.
    """

    for publication in load_publications():
        if publication["source"] == source:
            return publication

    return None


def get_publication_titles() -> list[str]:
    """
    Return all publication titles.
    """

    return [
        publication["title"]
        for publication in load_publications()
    ]


def format_publication(
    publication: dict[str, Any],
) -> str:
    """
    Create a readable citation-like description.
    """

    year = publication.get("year") or "Year unknown"
    journal = publication.get("journal")
    publication_type = publication.get(
        "publication_type",
        "Publication",
    )

    if journal:
        outlet = journal
    else:
        outlet = publication_type

    return (
        f"{publication['title']} "
        f"({year}) — {outlet}"
    )