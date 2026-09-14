"""ChromaDB vector store operations."""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document


def get_client(persist_dir: str | Path | None = None) -> chromadb.ClientAPI:
    """Create a ChromaDB client.

    If persist_dir is given, data is saved to disk and survives restarts.
    Otherwise, data lives only in memory.
    """
    if persist_dir is not None:
        return chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
    return chromadb.EphemeralClient(
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.ClientAPI,
    name: str = "documents",
    distance_fn: str = "cosine",
) -> chromadb.Collection:
    """Get an existing collection or create a new one.

    distance_fn: "cosine" (default), "l2", or "ip" (inner product).
    """
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": distance_fn},
    )


def add_documents(
    collection: chromadb.Collection,
    documents: list[Document],
    embeddings: list[list[float]],
    id_prefix: str = "doc",
) -> list[str]:
    """Add chunked documents with their embeddings to the collection.

    Returns the list of generated IDs.
    """
    ids = [f"{id_prefix}_{i}" for i in range(len(documents))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents],
    )

    return ids


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Query the collection with an embedding vector.

    Returns a list of result dicts, each with keys:
        id, document, metadata, distance
    sorted by relevance (closest first).
    """
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    if collection.count() == 0:
        return []

    results = collection.query(**kwargs)

    return [
        {
            "id": results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]


def delete_collection(client: chromadb.ClientAPI, name: str) -> None:
    """Delete a collection by name."""
    client.delete_collection(name)


def collection_stats(collection: chromadb.Collection) -> dict:
    """Return basic stats about a collection."""
    return {
        "name": collection.name,
        "count": collection.count(),
        "metadata": collection.metadata,
    }
