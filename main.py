from configure import (
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    TOP_K,
    RETRIEVAL_MODE,
    CANDIDATES,
    RRF_K,
    RERANKER_MODEL,
    PAPER_CATALOG,
    PAPER_ROUTING,
    GUARD_ENABLED
)

from embeddings.embedder import Embedder
from vectorestore.faiss_store import FAISSStore
from retrival.retreiver import Retriever
from retrival.paper_router import PaperRouter
from LLM.generator import generate_answer, NOT_FOUND
from LLM.guard import evidence_gate, verify_answer


def create_rag(
    mode=RETRIEVAL_MODE,
    reranker_model=RERANKER_MODEL,
    routing=PAPER_ROUTING
):
    """
    Build the retriever. Arguments default to configure.py;
    the evaluation overrides them to compare configurations.
    """

    embedder = Embedder(EMBEDDING_MODEL)

    store = FAISSStore(embedder.dimension)
    store.load(VECTORSTORE_DIR)

    retriever = Retriever(
        embedder,
        store,
        mode=mode,
        candidates=CANDIDATES,
        rrf_k=RRF_K,
        reranker_model=reranker_model,
        router=PaperRouter(PAPER_CATALOG),
        routing=routing
    )

    return retriever


def answer_question(retriever, query, top_k=TOP_K, guard=GUARD_ENABLED):
    """
    Full pipeline: retrieve -> evidence gate -> generate -> answer check.
    Returns a dict with the final answer, the raw LLM answer,
    the evidence used and what the guard decided.
    """

    results = retriever.retrieve(query, top_k=top_k)

    output = {
        "answer": None,
        "raw_answer": None,
        "results": results,
        "guard": {"enabled": guard}
    }

    if guard:

        passed, best_score = evidence_gate(results)
        output["guard"]["best_rerank_score"] = best_score

        if not passed:
            output["answer"] = NOT_FOUND
            output["guard"].update(refused=True, reason="weak_evidence")
            return output

    raw_answer = generate_answer(query, results)
    output["raw_answer"] = raw_answer

    if not guard:
        output["answer"] = raw_answer
        return output

    check = verify_answer(query, raw_answer, results)

    output["answer"] = check["answer"]
    output["guard"].update(
        refused=check["refused"],
        reason=check["reason"],
        sentences=check["sentences"]
    )

    return output


def main():

    retriever = create_rag()

    print("\nResearch Paper RAG")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nAsk: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        output = answer_question(retriever, query)

        print("\nAnswer:")
        print(output["answer"])

        guard = output["guard"]
        if guard.get("reason") not in (None, "ok"):
            print(f"\n[guard: {guard['reason']}]")

        print("\nSources:")

        for result in output["results"]:

            print(
                f"- {result.get('source', 'unknown')} "
                f"p.{result.get('page', '?')} "
                f"(score={result.get('score', 0):.3f})"
            )


if __name__ == "__main__":
    main()
