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


# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Retrieval
TOP_K = 3


# LLM (Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "phi3"
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 180
