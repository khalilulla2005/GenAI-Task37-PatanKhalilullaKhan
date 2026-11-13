from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class PdfDocumentLoader(BaseLoader):
    """Load searchable PDF pages as LangChain Documents using pypdf."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        if not self.file_path.is_file():
            raise FileNotFoundError(f"PDF not found: {self.file_path}")

        from pypdf import PdfReader

        reader = PdfReader(str(self.file_path))
        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDFs are not supported: {self.file_path.name}")

        for page_number, page in enumerate(reader.pages):
            yield Document(
                page_content=page.extract_text() or "",
                metadata={
                    "source": self.file_path.name,
                    "page": page_number,
                },
            )
