from pathlib import Path

import pymupdf

from configure import CHUNK_SIZE, CHUNK_OVERLAP


def read_text(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Text file not found: {file_path}"
        )

    return file_path.read_text(
        encoding="utf-8"
    )


def clean_text(text):
    """
    Collapse layout whitespace: runs of spaces inside a line
    and empty lines, which otherwise waste chunk space.
    """

    lines = (
        " ".join(line.split())
        for line in text.splitlines()
    )

    return "\n".join(
        line for line in lines if line
    )


def read_pdf_pages(pdf_path):
    """
    Extract text from a PDF, one entry per page.
    Returns: [{"page": 1, "text": "..."}, ...]
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    pages = []

    with pymupdf.open(str(pdf_path)) as document:

        for page_number, page in enumerate(document, start=1):

            text = clean_text(
                page.get_text("text", sort=True)
            )

            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def chunk_text(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:

            chunks.append({
                "content": chunk,
                "type": "text"
            })

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def chunk_pdf(pdf_path):
    """
    Chunk a PDF page by page so every chunk keeps
    its source file and page number.
    """

    pdf_path = Path(pdf_path)

    chunks = []

    for page in read_pdf_pages(pdf_path):

        for chunk in chunk_text(page["text"]):

            chunk["source"] = pdf_path.name
            chunk["page"] = page["page"]

            chunks.append(chunk)

    return chunks
