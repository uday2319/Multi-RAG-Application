import re

import requests

from configure import (
    OLLAMA_CHAT_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT
)


NOT_FOUND = "Answer not found in the papers."


SYSTEM_PROMPT = f"""You are a research paper question-answering assistant.
Answer the question ONLY using the numbered evidence passages.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or numbers. Copy numbers exactly as written.
3. If the evidence does not contain the answer, reply exactly:
{NOT_FOUND}
4. Give a concise and direct answer (1-4 sentences).
5. After each sentence, cite the passage(s) you used by number, e.g. [1] or [2][3].
6. If the question compares papers, state what each paper says.
"""


STOP_SEQUENCES = [
    "\nInstruction:",
    "\nQuestion:",
    "\nQUESTION:",
    "\nEVIDENCE:",
    "\nInput:",
    "\nOutput:",
]


def build_context(context_chunks):
    """
    Number the passages [1], [2], ... and label each with
    its paper and page. The model cites the short number;
    we map it back to (file, page) afterwards.
    """

    context_parts = []

    for number, chunk in enumerate(context_chunks, start=1):

        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")

        context_parts.append(
            f"[{number}] ({source}, page {page})\n{chunk['content']}"
        )

    return "\n\n".join(context_parts)


CITE_NUMBERS_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def resolve_citations(answer, context_chunks):
    """
    Replace numeric citations like [2] or [1, 3] with
    [file.pdf, p.N]. Numbers that do not match a passage
    are dropped, since they point to nothing.
    """

    def replace(match):

        labels = []

        for number in match.group(1).split(","):

            index = int(number) - 1

            if 0 <= index < len(context_chunks):
                chunk = context_chunks[index]
                labels.append(
                    f"[{chunk.get('source', 'unknown')}, p.{chunk.get('page', '?')}]"
                )

        return "".join(labels)

    return CITE_NUMBERS_RE.sub(replace, answer)


def generate_answer(query, context_chunks):

    context = build_context(context_chunks)

    user_message = f"""EVIDENCE:
{context}

QUESTION:
{query}"""

    response = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {
                "num_predict": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE,
                # Small models sometimes continue by inventing the
                # next prompt turn; stop as soon as that starts
                "stop": STOP_SEQUENCES
            }
        },
        timeout=LLM_TIMEOUT
    )

    response.raise_for_status()

    answer = response.json()["message"]["content"].strip()

    return resolve_citations(answer, context_chunks)
