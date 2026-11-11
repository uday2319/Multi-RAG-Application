import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from configure import EMBEDDING_MODEL, VECTORSTORE_DIR, TOP_K
from embeddings.embedder import Embedder
from vectorestore.faiss_store import FAISSStore
from retrival.retreiver import Retriever
from LLM.generator import generate_answer
from logs.logger_config import setup_logger


logger = setup_logger()


app = FastAPI(
    title="Research Paper RAG",
    description="Text-based Research Paper Question Answering System",
    version="1.0"
)


class QueryRequest(BaseModel):
    question: str


def create_rag():

    logger.info("Initializing RAG system")

    embedder = Embedder(EMBEDDING_MODEL)

    store = FAISSStore(embedder.dimension)
    store.load(VECTORSTORE_DIR)

    logger.info(
        f"Vector store loaded: {store.index.ntotal} vectors"
    )

    retriever = Retriever(
        embedder,
        store
    )

    logger.info("RAG system initialized successfully")

    return retriever


retriever = create_rag()


@app.get("/")
def root():

    return {
        "message": "Research Paper RAG API is running"
    }


@app.post("/ask")
def ask_question(request: QueryRequest):

    start_time = time.time()

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        # Retrieve relevant chunks
        results = retriever.retrieve(
            question,
            top_k=TOP_K
        )

        logger.info(
            f"Retrieved {len(results)} chunks"
        )

        # Generate answer
        answer = generate_answer(
            question,
            results
        )

        # Sources
        sources = []

        for result in results:

            sources.append({
                "source": result.get(
                    "source",
                    "unknown"
                ),
                "page": result.get("page"),
                "score": round(
                    result.get("score", 0),
                    4
                )
            })

        latency = time.time() - start_time

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "latency_seconds": round(latency, 2)
        }

    except Exception as e:

        logger.exception(
            f"RAG request failed: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing the question."
        )