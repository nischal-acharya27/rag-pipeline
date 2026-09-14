"""Document loading and chunking utilities."""

from pathlib import Path
from typing import Optional, Union

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_text_file(path: Union[str, Path]) -> list:
    """Load a single text file into LangChain Documents."""
    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def load_pdf(path: Union[str, Path]) -> list:
    """Load a PDF file, returning one Document per page."""
    loader = PyPDFLoader(str(path))
    return loader.load()


def load_directory(
    path: Union[str, Path],
    glob: str = "**/*.*",
    show_progress: bool = False,
) -> list:
    """Load all supported files from a directory.

    Supports .txt and .pdf files. Each file becomes one or more Documents
    with source metadata automatically attached.
    """
    loaders = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
    }

    docs = []
    for ext, loader_cls in loaders.items():
        dir_loader = DirectoryLoader(
            str(path),
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            show_progress=show_progress,
            use_multithreading=True,
        )
        docs.extend(dir_loader.load())

    return docs


def chunk_documents(
    documents: list,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: Optional[list] = None,
) -> list:
    """Split documents into chunks using recursive character splitting.

    Each chunk inherits its parent document's metadata plus chunk-specific
    fields (chunk_index, chunk_total) for traceability.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_documents([doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)
        all_chunks.extend(chunks)

    return all_chunks


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    metadata: Optional[dict] = None,
) -> list:
    """Chunk a raw string into Documents.

    Convenience wrapper when you have text but not a Document object.
    """
    doc = Document(page_content=text, metadata=metadata or {})
    return chunk_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
