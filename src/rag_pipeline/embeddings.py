"""Embedding utilities for converting text to vectors."""

from sentence_transformers import SentenceTransformer, util
import numpy as np


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a sentence-transformer embedding model."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Embed a list of texts into vectors.

    Returns an (N, D) array where N = number of texts and D = embedding dimension.
    """
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 50)


def compute_similarity(query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query and a set of documents.

    Returns a 1-D array of similarity scores.
    """
    return util.cos_sim(query_embedding, doc_embeddings)[0].numpy()


def rank_by_similarity(
    query: str,
    documents: list[str],
    model: SentenceTransformer,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Embed a query and documents, return top-k documents ranked by similarity.

    Returns a list of (document_text, similarity_score) tuples, highest first.
    """
    query_emb = model.encode(query, convert_to_numpy=True)
    doc_embs = embed_texts(model, documents)
    scores = compute_similarity(query_emb, doc_embs)

    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [(documents[i], float(scores[i])) for i in ranked_indices]
