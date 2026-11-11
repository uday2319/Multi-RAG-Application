import pymupdf
from pathlib import Path

from configure import FIGURES_DIR


def extract_figures(pdf_path, output_dir=FIGURES_DIR):
    """
    Extract images/figures from a PDF and preserve page information.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(str(pdf_path))

    figures = []

    for page_number, page in enumerate(document, start=1):

        images = page.get_images(full=True)

        for figure_number, image_info in enumerate(images, start=1):

            xref = image_info[0]

            image_data = document.extract_image(xref)

            image_bytes = image_data["image"]
            image_ext = image_data["ext"]

            filename = (
                f"{pdf_path.stem}_"
                f"page_{page_number}_"
                f"figure_{figure_number}.{image_ext}"
            )

            image_path = output_dir / filename

            with open(image_path, "wb") as file:
                file.write(image_bytes)

            figures.append({
                "type": "figure",
                "source": pdf_path.name,
                "page": page_number,
                "figure": figure_number,
                "path": str(image_path)
            })

    document.close()

    return figures