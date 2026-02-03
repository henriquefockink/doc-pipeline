"""PDF to image converter using PyMuPDF."""

import io
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image


class PDFConverter:
    """Converts PDF pages to PIL Images."""

    def __init__(self, dpi: int = 200):
        """
        Initialize converter.

        Args:
            dpi: Resolution for rendering PDF pages (default 200)
        """
        self.dpi = dpi
        self.zoom = dpi / 72  # PDF default is 72 DPI

    def convert(
        self,
        pdf_path: str | Path,
        max_pages: int | None = None,
    ) -> list[Image.Image]:
        """
        Convert PDF to list of PIL Images.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum number of pages to convert (None = all)

        Returns:
            List of PIL Images, one per page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        images = []
        doc = fitz.open(pdf_path)

        try:
            num_pages = len(doc)
            if max_pages is not None:
                num_pages = min(num_pages, max_pages)

            for page_num in range(num_pages):
                page = doc[page_num]

                # Render page to pixmap
                mat = fitz.Matrix(self.zoom, self.zoom)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)

        finally:
            doc.close()

        return images

    def get_page_count(self, pdf_path: str | Path) -> int:
        """Get number of pages in PDF without converting."""
        doc = fitz.open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()


def is_pdf(file_path: str | Path) -> bool:
    """Check if file is a PDF based on magic bytes."""
    with open(file_path, "rb") as f:
        header = f.read(4)
        return header == b"%PDF"
