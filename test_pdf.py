from ingestion.pdf_processor import read_pdf_pages, chunk_pdf

pdf_path = "data/papers/rag_lewis2020.pdf"

pages = read_pdf_pages(pdf_path)

print("Pages extracted:", len(pages))
print("Characters extracted:", sum(len(p["text"]) for p in pages))

chunks = chunk_pdf(pdf_path)

print("Chunks created:", len(chunks))

for chunk in chunks[:3]:

    print("\n----------------")

    print(f"Chunk ({chunk['source']}, p.{chunk['page']}):")
    print(chunk["content"][:500])
