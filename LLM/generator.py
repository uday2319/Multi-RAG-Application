import requests

from configure import (
    OLLAMA_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT
)


def build_context(context_chunks):

    context_parts = []

    for chunk in context_chunks:

        source = chunk.get("source", "unknown")
        page = chunk.get("page", "?")

        context_parts.append(
            f"[{source}, p.{page}]\n{chunk['content']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query, context_chunks):

    context = build_context(context_chunks)

    prompt = f"""
You are a research paper question-answering assistant.

Answer the question ONLY using the provided evidence.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or numbers.
3. If the evidence does not contain the answer, say exactly:
Answer not found in the paper.
4. Give a concise and direct answer.
5. Cite the evidence you used with its label, e.g. [rag_lewis2020.pdf, p.3].

EVIDENCE:
{context}

QUESTION:
{query}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": LLM_MAX_TOKENS,
                "temperature": LLM_TEMPERATURE
            }
        },
        timeout=LLM_TIMEOUT
    )

    response.raise_for_status()

    return response.json()["response"].strip()
