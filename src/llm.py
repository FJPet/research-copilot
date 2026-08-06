from collections import defaultdict
from typing import Any

from ollama import Client


MODEL_NAME = "qwen2.5:7b"


def build_context(
    retrieved_passages: list[dict[str, Any]],
) -> str:
    """
    Group retrieved passages by paper.
    """

    passages_by_paper = defaultdict(list)

    for passage in retrieved_passages:
        passages_by_paper[
            passage["source"]
        ].append(passage)

    context_blocks = []

    for source, passages in passages_by_paper.items():
        paper_lines = [
            f"PAPER: {source}",
        ]

        for passage in passages:
            paper_lines.extend(
                [
                    f"PAGE: {passage['page']}",
                    f"PASSAGE: {passage['text']}",
                    "",
                ]
            )

        context_blocks.append(
            "\n".join(paper_lines)
        )

    return "\n\n====================\n\n".join(
        context_blocks
    )


def generate_answer(
    question: str,
    retrieved_passages: list[dict[str, Any]],
) -> str:
    """
    Generate a paper-aware answer.
    """

    context = build_context(
        retrieved_passages
    )

    system_prompt = """
You are an assistant for exploring the scientific publications of
Franziska J. Peter.

Use only the supplied passages.

Critical rules:

1. Report only methods, data, results, or claims that are actually used
   or presented in the focal paper.

2. Do not attribute methods from cited literature to the focal paper.

3. Distinguish clearly between:
   - methods used by the authors,
   - benchmark models,
   - methods mentioned only in the literature review,
   - future work or suggestions.

4. If a passage is ambiguous, do not make a strong claim.

5. For broad questions, organize the answer by paper.

6. Omit papers for which the supplied passages do not provide relevant
   supporting evidence.

7. Use the PDF filename as the paper identifier unless the full title is
   clearly available in the passage.

8. Never reinterpret acronyms without evidence. For example, CNN in this
   corpus must not be expanded as human activity recognition unless the
   passage explicitly says so.

9. End with a short synthesis across the supported papers.

Write clearly and accurately for a technically informed reader.
"""

    user_prompt = f"""
QUESTION:
{question}

PAPER-AWARE EVIDENCE:
{context}
"""

    client = Client(
        host="http://localhost:11434",
    )

    response = client.chat(
        model=MODEL_NAME,
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
        options={
            "temperature": 0.0,
        },
    )

    return response["message"]["content"]