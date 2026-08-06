import json
import re
from pathlib import Path
from typing import Any

from ollama import Client
from sentence_transformers import SentenceTransformer

from src.vector_store import get_paper_names, search_paper


PROFILE_PATH = Path("data/paper_profiles.json")
LLM_MODEL = "qwen2.5:7b"


METHOD_QUERIES = [
    (
        "Which statistical, econometric, machine-learning, deep-learning, "
        "network, forecasting, or information-theoretic methods do the "
        "authors actually apply in their own empirical analysis?"
    ),
    (
        "What model or models are estimated, trained, or implemented by "
        "the authors? Focus on phrases such as we use, we estimate, "
        "we apply, our model, and empirical analysis."
    ),
    (
        "Which prediction, classification, feature-selection, forecasting, "
        "explainability, inference, or data-representation methods are "
        "actually implemented in this paper?"
    ),
    (
        "Describe the methodology used by the authors, excluding methods "
        "that appear only in the literature review or reference list."
    ),
]


def is_reference_like(text: str) -> bool:
    """
    Identify passages that appear to be reference-list material.
    """

    normalized = text.strip().lower()

    if normalized.startswith(
        (
            "references",
            "bibliography",
            "literature cited",
        )
    ):
        return True

    citation_years = re.findall(
        r"\((?:19|20)\d{2}[a-z]?\)",
        text,
    )

    doi_count = text.lower().count("doi")
    et_al_count = text.lower().count("et al.")

    return (
        len(citation_years) >= 5
        or doi_count >= 3
        or et_al_count >= 5
    )


def collect_method_passages(
    source: str,
    embedding_model: SentenceTransformer,
) -> list[dict[str, Any]]:
    """
    Retrieve method-focused passages from one paper.
    """

    collected: dict[
        tuple[str, int, int],
        dict[str, Any],
    ] = {}

    for query in METHOD_QUERIES:
        passages = search_paper(
            query=query,
            source=source,
            model=embedding_model,
            number_of_results=4,
        )

        for passage in passages:
            if is_reference_like(passage["text"]):
                continue

            identifier = (
                passage["source"],
                passage["page"],
                passage["chunk"],
            )

            collected[identifier] = passage

    return sorted(
        collected.values(),
        key=lambda passage: (
            passage["page"],
            passage["chunk"],
        ),
    )


def build_evidence_text(
    passages: list[dict[str, Any]],
) -> str:
    """
    Format retrieved evidence for the LLM.
    """

    blocks = []

    for passage in passages:
        blocks.append(
            "\n".join(
                [
                    f"PAGE: {passage['page']}",
                    f"TEXT: {passage['text']}",
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)


def extract_paper_profile(
    source: str,
    embedding_model: SentenceTransformer,
) -> dict[str, Any]:
    """
    Extract methods actually used by the authors of one paper.
    """

    passages = collect_method_passages(
        source=source,
        embedding_model=embedding_model,
    )

    evidence = build_evidence_text(passages)

    system_prompt = """
You extract structured methodological information from scientific papers.

Critical rules:

1. Include only methods actually used, estimated, trained, implemented,
   or applied by the authors in the focal paper.

2. Exclude methods that appear only:
   - in cited literature,
   - in the references,
   - as unrelated background,
   - as possible future work.

3. Benchmark models may be included, but mark them as benchmarks.

4. Distinguish categories such as:
   - machine-learning model,
   - deep-learning model,
   - econometric model,
   - statistical model,
   - feature-selection method,
   - explainability method,
   - network or embedding method,
   - information-theoretic method,
   - data representation.

5. Every reported method must include supporting page numbers.

6. When the evidence is insufficient, omit the method rather than guessing.

Return valid JSON only.
"""

    user_prompt = f"""
PAPER FILE:
{source}

METHOD-FOCUSED PASSAGES:
{evidence}

Return this JSON structure:

{{
  "source": "{source}",
  "methods_used": [
    {{
      "name": "method name",
      "category": "method category",
      "role": "how it is used in this paper",
      "status": "primary method or benchmark",
      "evidence_pages": [1, 2]
    }}
  ],
  "method_summary": "brief summary of the paper's actual methodology",
  "uncertainties": []
}}
"""

    client = Client(
        host="http://localhost:11434",
    )

    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        format="json",
        options={
            "temperature": 0.0,
        },
    )

    profile = json.loads(
        response["message"]["content"]
    )

    profile["retrieved_pages"] = sorted(
        {
            passage["page"]
            for passage in passages
        }
    )

    return profile


def build_all_profiles(
    embedding_model: SentenceTransformer,
) -> list[dict[str, Any]]:
    """
    Build and save one structured profile per paper.
    """

    profiles = []
    paper_names = get_paper_names()

    PROFILE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, source in enumerate(
        paper_names,
        start=1,
    ):
        print(
            f"Processing paper {index} of "
            f"{len(paper_names)}: {source}"
        )

        profile = extract_paper_profile(
            source=source,
            embedding_model=embedding_model,
        )

        profiles.append(profile)

        PROFILE_PATH.write_text(
            json.dumps(
                profiles,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return profiles


def load_profiles() -> list[dict[str, Any]]:
    """
    Load previously generated paper profiles.
    """

    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            "Paper profiles have not been created yet."
        )

    return json.loads(
        PROFILE_PATH.read_text(
            encoding="utf-8",
        )
    )