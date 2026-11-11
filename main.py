from configure import (
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    TOP_K
)

from embeddings.embedder import Embedder
from vectorestore.faiss_store import FAISSStore
from retrival.retreiver import Retriever
from LLM.generator import generate_answer


def create_rag():

    embedder = Embedder(EMBEDDING_MODEL)

    store = FAISSStore(embedder.dimension)
    store.load(VECTORSTORE_DIR)

    retriever = Retriever(
        embedder,
        store
    )

    return retriever


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

        results = retriever.retrieve(
            query,
            top_k=TOP_K
        )

        print(f"\nRetrieved {len(results)} chunks.")

        answer = generate_answer(
            query,
            results
        )

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        for result in results:

            print(
                f"- {result.get('source', 'unknown')} "
                f"p.{result.get('page', '?')} "
                f"(score={result.get('score', 0):.3f})"
            )


if __name__ == "__main__":
    main()