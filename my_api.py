import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from configure import TOP_K
from main import create_rag as build_retriever, answer_question
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

    retriever = build_retriever()

    logger.info(
        f"Vector store loaded: "
        f"{retriever.vectorstore.index.ntotal} vectors, "
        f"mode={retriever.mode}, "
        f"reranker={'on' if retriever.reranker else 'off'}"
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

        # Retrieve -> evidence gate -> generate -> answer check
        output = answer_question(
            retriever,
            question,
            top_k=TOP_K
        )

        guard = output["guard"]

        logger.info(
            f"Retrieved {len(output['results'])} chunks, "
            f"guard={guard.get('reason', 'off')}"
        )

        # Sources
        sources = []

        for result in output["results"]:

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
            "answer": output["answer"],
            "sources": sources,
            "guard": {
                "refused": guard.get("refused", False),
                "reason": guard.get("reason"),
                "best_rerank_score": guard.get("best_rerank_score")
            },
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