from configure import (
    EMBEDDING_MODEL,
    PAPERS_DIR,
    VECTORSTORE_DIR
)

from ingestion.pdf_processor import chunk_pdf

from embeddings.embedder import Embedder
from vectorestore.faiss_store import FAISSStore


# 1. Find PDFs
pdf_paths = sorted(PAPERS_DIR.glob("*.pdf"))

if not pdf_paths:
    raise SystemExit(f"No PDFs found in {PAPERS_DIR}")


# 2. Read and chunk every PDF (each chunk keeps source + page)
chunks = []

for pdf_path in pdf_paths:

    pdf_chunks = chunk_pdf(pdf_path)

    print(f"{pdf_path.name}: {len(pdf_chunks)} chunks")

    chunks.extend(pdf_chunks)


print("Total chunks:", len(chunks))


# 3. Create embeddings
embedder = Embedder(EMBEDDING_MODEL)

texts = [
    chunk["content"]
    for chunk in chunks
]

embeddings = embedder.embed(texts)

print("Embeddings:", embeddings.shape)


# 4. Create FAISS index
store = FAISSStore(embedder.dimension)

store.add(
    embeddings,
    chunks
)


# 5. Save vector store
store.save(VECTORSTORE_DIR)

print("Vector store created successfully.")
