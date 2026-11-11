from ingestion.figure_processor import extract_figures


pdf_path = "data/papers/rag_lewis2020.pdf"

figures = extract_figures(pdf_path)

print("Figures extracted:", len(figures))

for figure in figures:
    print("\n----------------")
    print("Page:", figure["page"])
    print("Figure:", figure["figure"])
    print("Path:", figure["path"])
