import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from configure import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTRACTOR,
    CHUNKING
)


def read_text(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Text file not found: {file_path}"
        )

    return file_path.read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------

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


PICTURE_TEXT_RE = re.compile(
    r"<!-- Start of picture text -->.*?<!-- End of picture text -->",
    re.DOTALL
)
HTML_TAG_RE = re.compile(r"</?(sup|sub|span)[^>]*>")


def clean_markdown(text):
    """
    Light cleanup of pymupdf4llm markdown:
    - drop text scraped from inside figures (axis labels etc.)
    - keep sup/sub content but drop the tags
    - turn <br> inside table cells into spaces
    - drop bold markers, collapse 3+ blank lines
    Paragraph breaks (blank lines) are kept for chunking.
    """

    text = PICTURE_TEXT_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    text = text.replace("<br>", " ")
    text = text.replace("**", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------

def read_pdf_pages_text(pdf_path):
    """
    Baseline extractor: PyMuPDF plain text.
    Mixes the two columns of two-column papers.
    """

    pages = []

    with pymupdf.open(str(pdf_path)) as document:

        for page_number, page in enumerate(document, start=1):

            pages.append({
                "page": page_number,
                "text": clean_text(
                    page.get_text("text", sort=True)
                )
            })

    return pages


def read_pdf_pages_markdown(pdf_path):
    """
    Layout-aware extractor: pymupdf4llm reads columns in order
    and turns tables into markdown tables.
    """

    with pymupdf.open(str(pdf_path)) as document:

        page_chunks = pymupdf4llm.to_markdown(
            document,
            page_chunks=True,
            show_progress=False
        )

    pages = []

    for page_chunk in page_chunks:

        pages.append({
            "page": page_chunk["metadata"]["page_number"],
            "text": clean_markdown(page_chunk["text"])
        })

    return pages


def read_pdf_pages(pdf_path, extractor=PDF_EXTRACTOR):
    """
    Extract text from a PDF, one entry per page.
    Returns: [{"page": 1, "text": "..."}, ...]
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if extractor == "markdown":
        return read_pdf_pages_markdown(pdf_path)

    return read_pdf_pages_text(pdf_path)


# ---------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------

def chunk_text(text):
    """
    Baseline chunker: fixed-size character windows with overlap.
    """

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


def chunk_paragraphs(text):
    """
    Paragraph-aware chunker: packs whole paragraphs (blank-line
    separated blocks, so tables stay in one block) into chunks
    of up to CHUNK_SIZE characters. A paragraph longer than
    CHUNK_SIZE falls back to the fixed-size chunker.
    The last paragraph of a chunk is repeated at the start of
    the next one as overlap.
    """

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = []
    current_len = 0

    def flush():
        if current:
            chunks.append({
                "content": "\n\n".join(current),
                "type": "text"
            })

    for paragraph in paragraphs:

        if len(paragraph) > CHUNK_SIZE:

            flush()
            current, current_len = [], 0

            chunks.extend(chunk_text(paragraph))
            continue

        if current_len + len(paragraph) > CHUNK_SIZE and current:

            flush()

            overlap = current[-1]

            if len(overlap) + len(paragraph) <= CHUNK_SIZE:
                current, current_len = [overlap], len(overlap)
            else:
                current, current_len = [], 0

        current.append(paragraph)
        current_len += len(paragraph)

    flush()

    return chunks


def chunk_pdf(pdf_path, chunking=CHUNKING):
    """
    Chunk a PDF page by page so every chunk keeps
    its source file and page number.
    """

    pdf_path = Path(pdf_path)

    chunker = (
        chunk_paragraphs
        if chunking == "paragraph"
        else chunk_text
    )

    chunks = []

    for page in read_pdf_pages(pdf_path):

        for chunk in chunker(page["text"]):

            chunk["source"] = pdf_path.name
            chunk["page"] = page["page"]

            chunks.append(chunk)

    return chunks
