from pathlib import Path


# Project root
BASE_DIR = Path(__file__).resolve().parent


# Data directories
DATA_DIR = BASE_DIR / "data"

PAPERS_DIR = DATA_DIR / "papers"
FIGURES_DIR = DATA_DIR / "figures"


# FAISS vector database
VECTORSTORE_DIR = BASE_DIR / "vectorestore"


# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# PDF extraction: "text" (plain PyMuPDF, mixes columns)
# or "markdown" (pymupdf4llm, column-aware with markdown tables)
PDF_EXTRACTOR = "markdown"


# Chunking: "fixed" (character windows) or "paragraph"
CHUNKING = "fixed"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Retrieval
TOP_K = 3

# "dense" (FAISS only) or "hybrid" (FAISS + BM25, fused with RRF)
RETRIEVAL_MODE = "hybrid"

# How many candidates each retriever returns before fusion/reranking
CANDIDATES = 30

# Reciprocal Rank Fusion constant (standard value from the RRF paper)
RRF_K = 60

# Cross-encoder reranker; set to None to disable
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paper routing: when a question names papers (aliases in
# data/papers/papers.json), search inside each named paper.
# "off", "multi" (only when 2+ papers are named) or "all"
PAPER_CATALOG = PAPERS_DIR / "papers.json"
PAPER_ROUTING = "all"


# LLM (Ollama chat API)
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "phi3"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 180


# Hallucination guard
GUARD_ENABLED = True

# Step 1: refuse before the LLM if the best cross-encoder score is
# below this. Chosen from eval score distributions: answerable
# questions scored >= -0.83, three unanswerable ones <= -2.84.
GUARD_MIN_RERANK_SCORE = -2.0

# Step 2: share of a sentence's new words that must appear in
# the retrieved evidence for the sentence to be kept.
GUARD_MIN_SUPPORT = 0.5
