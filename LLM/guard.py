"""
Hallucination guard.

Step 1 - evidence gate (before the LLM):
    If even the best retrieved chunk is a poor match for the
    question (low cross-encoder score), refuse without calling
    the LLM.

Step 2 - answer check (after the LLM):
    Split the answer into sentences. For each sentence, take the
    words it adds beyond the question and check that (a) enough
    of them appear in the retrieved evidence and (b) every number
    in it appears in the evidence. Unsupported sentences are
    removed; if nothing is left, the answer becomes a refusal.
"""

import re

from configure import (
    GUARD_MIN_RERANK_SCORE,
    GUARD_MIN_SUPPORT
)
from LLM.generator import NOT_FOUND


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CITATION_RE = re.compile(r"\[[^\[\]]*?\.pdf\s*,\s*p\.?\s*\d+\s*\]", re.IGNORECASE)
# Split after . ! ? and also on line breaks
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
REFUSAL_RE = re.compile(r"not found", re.IGNORECASE)

STOPWORDS = set("""
a an the and or but if then than so of to in on at by for with from into
over under as is are was were be been being has have had do does did it
its this that these those which who whom whose what when where why how
not no can could should would may might will shall also both each such
their there they them we our you your he she his her i me my
uses use used using based paper papers model models answer evidence
passage passages according described stated mentioned shown provided
""".split())


def normalize(token):
    """Very light stemming so 'tokens' matches 'token'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


# Scale words are not checked as words: "21 million" vs "21M"
# is a formatting difference. The number itself is still
# checked by the separate number rule.
NUMBER_WORDS = {"thousand", "million", "billion", "trillion", "percent", "%"}


def content_tokens(text):
    return {
        normalize(token)
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS
        and token not in NUMBER_WORDS
        and not NUMBER_RE.fullmatch(token)  # numbers: see number rule
        and (len(token) > 1 or token.isdigit())
    }


def numbers(text):
    return set(NUMBER_RE.findall(text))


def evidence_gate(results):
    """
    Returns (passed, best_score). Skipped (passes) when the
    retriever has no reranker scores.
    """

    scores = [
        r["rerank_score"]
        for r in results
        if r.get("rerank_score") is not None
    ]

    if not scores:
        return True, None

    best = max(scores)

    return best >= GUARD_MIN_RERANK_SCORE, best


def check_sentence(sentence, question_tokens, evidence_tokens, evidence_numbers):

    text = CITATION_RE.sub(" ", sentence)

    new_tokens = content_tokens(text) - question_tokens
    missing_numbers = numbers(text) - evidence_numbers

    if new_tokens:
        support = len(new_tokens & evidence_tokens) / len(new_tokens)
    else:
        # Sentence only repeats the question: nothing new to verify
        support = 1.0

    return {
        "sentence": sentence,
        "support": round(support, 3),
        "unsupported_words": sorted(new_tokens - evidence_tokens),
        "missing_numbers": sorted(missing_numbers),
        "supported": support >= GUARD_MIN_SUPPORT and not missing_numbers
    }


def split_sentences(answer):
    """
    Split into sentences. A fragment that is only citations
    (e.g. "[rag_lewis2020.pdf, p.4]" after the full stop) is
    attached to the sentence before it, so citations survive.
    """

    sentences = []

    for part in SENTENCE_RE.split(answer.strip()):

        part = part.strip()

        if not part:
            continue

        if not CITATION_RE.sub("", part).strip():
            if sentences:
                sentences[-1] += " " + part
            continue

        sentences.append(part)

    return sentences


def verify_answer(question, answer, results):
    """
    Check every sentence of the answer against the evidence.
    Returns the cleaned answer and per-sentence details.
    """

    if REFUSAL_RE.search(answer):
        return {"answer": NOT_FOUND, "refused": True, "reason": "model_refused", "sentences": []}

    evidence_text = "\n".join(r["content"] for r in results)

    evidence_tokens = content_tokens(evidence_text)
    evidence_numbers = numbers(evidence_text)
    question_tokens = content_tokens(question)

    sentences = split_sentences(answer)

    checks = [
        check_sentence(s, question_tokens, evidence_tokens, evidence_numbers)
        for s in sentences
    ]

    kept = [c["sentence"] for c in checks if c["supported"]]

    if not kept:
        return {"answer": NOT_FOUND, "refused": True, "reason": "unsupported_answer", "sentences": checks}

    return {
        "answer": " ".join(kept),
        "refused": False,
        "reason": "removed_sentences" if len(kept) < len(checks) else "ok",
        "sentences": checks
    }
