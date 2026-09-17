"""Advanced retrieval: keyword search, hybrid search, and reranking."""

import math
import re
from collections import Counter
from typing import Optional

from .chunking import chunk_documents, load_directory
from .embeddings import embed_texts, load_model
from .vectorstore import (
    add_documents,
    collection_stats,
    get_client,
    get_or_create_collection,
    query_collection,
)


# ---------------------------------------------------------------------------
# BM25 keyword search (pure Python, no extra dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Okapi BM25 keyword scoring over a corpus of documents.

    This is a from-scratch implementation so there are no extra dependencies.
    Production systems use libraries like rank-bm25 or Elasticsearch.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_freqs: Counter = Counter()
        self.doc_lens: list[int] = []

    def index(self, documents: list[str]) -> None:
        """Build the BM25 index from a list of document strings."""
        self.corpus = [_tokenize(doc) for doc in documents]
        self.doc_count = len(self.corpus)
        self.doc_lens = [len(doc) for doc in self.corpus]
        self.avg_dl = sum(self.doc_lens) / max(self.doc_count, 1)

        self.doc_freqs = Counter()
        for doc_tokens in self.corpus:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self.doc_freqs[token] += 1

    def _idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        df = self.doc_freqs.get(term, 0)
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """Score every document against a query. Returns list of floats."""
        query_tokens = _tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.corpus):
            tf_map = Counter(doc_tokens)
            doc_len = self.doc_lens[i]
            s = 0.0
            for term in query_tokens:
                tf = tf_map.get(term, 0)
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_dl
                )
                s += idf * numerator / denominator
            scores.append(s)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        """Return (doc_index, score) pairs sorted by score descending."""
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Each ranked_list is [(doc_index, score), ...] in descending score order.
    k is a smoothing constant (default 60, from the original RRF paper).

    Returns [(doc_index, rrf_score), ...] sorted by fused score.
    """
    fused_scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked_list):
            fused_scores[doc_idx] = fused_scores.get(doc_idx, 0.0) + 1.0 / (
                k + rank + 1
            )
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    documents: list[str],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: Optional[int] = None,
) -> list[tuple[int, float]]:
    """Rerank documents using a cross-encoder model.

    A cross-encoder scores (query, document) pairs jointly, which is more
    accurate than bi-encoder similarity but too slow to run on the full corpus.
    That's why we retrieve first (fast, approximate) then rerank (slow, precise).

    Returns [(doc_index, relevance_score), ...] sorted by score descending.
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        ranked = ranked[:top_k]
    return [(idx, float(score)) for idx, score in ranked]


# ---------------------------------------------------------------------------
# HybridRetriever — puts it all together
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Retriever that combines semantic search, keyword search, and reranking.

    Supports three retrieval modes:
        "semantic"  — embedding similarity only (what you had before)
        "keyword"   — BM25 keyword scoring only
        "hybrid"    — fuses semantic + keyword results with RRF

    Optionally reranks the top results with a cross-encoder for maximum quality.

    Usage:
        retriever = HybridRetriever(persist_dir="./chroma_db")
        retriever.index("data/sample/")
        results = retriever.retrieve("remote work policy", mode="hybrid", rerank=True)
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.model = load_model(embedding_model)
        self.client = get_client(persist_dir)
        self.collection = get_or_create_collection(self.client, collection_name)

        self.bm25 = BM25()
        self._chunks: list[dict] = []

    def index(self, data_path: str, clear_existing: bool = True) -> dict:
        """Index documents for both semantic and keyword search."""
        if clear_existing and self.collection.count() > 0:
            from .vectorstore import delete_collection

            delete_collection(self.client, self.collection_name)
            self.collection = get_or_create_collection(
                self.client, self.collection_name
            )

        docs = load_directory(data_path)
        chunks = chunk_documents(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        texts = [c.page_content for c in chunks]
        embeddings = embed_texts(self.model, texts)

        for chunk in chunks:
            for key, val in list(chunk.metadata.items()):
                if not isinstance(val, (str, int, float, bool)):
                    chunk.metadata[key] = str(val)

        ids = add_documents(self.collection, chunks, embeddings.tolist())

        self._chunks = [
            {
                "id": ids[i],
                "document": texts[i],
                "metadata": chunks[i].metadata,
            }
            for i in range(len(chunks))
        ]
        self.bm25.index(texts)

        return {
            "documents_loaded": len(docs),
            "chunks_created": len(chunks),
            "embeddings_stored": len(ids),
        }

    def _load_chunks_from_collection(self) -> None:
        """Load existing chunks from ChromaDB for BM25 indexing."""
        if self._chunks:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        if not data["ids"]:
            return
        self._chunks = [
            {
                "id": data["ids"][i],
                "document": data["documents"][i],
                "metadata": data["metadatas"][i],
            }
            for i in range(len(data["ids"]))
        ]
        self.bm25.index([c["document"] for c in self._chunks])

    def retrieve(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5,
        use_rerank: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Retrieve relevant chunks using the specified strategy.

        Args:
            question: The user's question.
            mode: "semantic", "keyword", or "hybrid".
            top_k: Number of results to return.
            use_rerank: If True, rerank the top results with a cross-encoder.
            rerank_model: Cross-encoder model for reranking.
            where: Optional ChromaDB metadata filter (semantic mode only).

        Returns:
            List of dicts with keys: id, document, metadata, score, method.
        """
        self._load_chunks_from_collection()
        if not self._chunks:
            return []

        candidate_k = top_k * 3 if use_rerank else top_k

        if mode == "semantic":
            results = self._semantic_search(question, candidate_k, where)
        elif mode == "keyword":
            results = self._keyword_search(question, candidate_k)
        elif mode == "hybrid":
            results = self._hybrid_search(question, candidate_k, where)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'semantic', 'keyword', or 'hybrid'.")

        if use_rerank and results:
            results = self._rerank_results(question, results, rerank_model, top_k)
        else:
            results = results[:top_k]

        return results

    def _semantic_search(
        self, question: str, top_k: int, where: Optional[dict] = None
    ) -> list[dict]:
        """Pure embedding-based search via ChromaDB."""
        query_emb = self.model.encode(question).tolist()
        raw = query_collection(self.collection, query_emb, top_k, where)
        return [
            {
                "id": r["id"],
                "document": r["document"],
                "metadata": r["metadata"],
                "score": round(1 - r["distance"], 4),
                "method": "semantic",
            }
            for r in raw
        ]

    def _keyword_search(self, question: str, top_k: int) -> list[dict]:
        """Pure BM25 keyword search."""
        ranked = self.bm25.search(question, top_k)
        return [
            {
                "id": self._chunks[idx]["id"],
                "document": self._chunks[idx]["document"],
                "metadata": self._chunks[idx]["metadata"],
                "score": round(score, 4),
                "method": "keyword",
            }
            for idx, score in ranked
            if score > 0
        ]

    def _hybrid_search(
        self, question: str, top_k: int, where: Optional[dict] = None
    ) -> list[dict]:
        """Fuse semantic + keyword results with Reciprocal Rank Fusion."""
        semantic_ranked = self.bm25.search(question, top_k)
        query_emb = self.model.encode(question).tolist()
        raw_semantic = query_collection(self.collection, query_emb, top_k, where)
        semantic_ranked_list = [
            (self._id_to_index(r["id"]), 1 - r["distance"])
            for r in raw_semantic
            if self._id_to_index(r["id"]) is not None
        ]

        keyword_ranked_list = [
            (idx, score) for idx, score in self.bm25.search(question, top_k)
            if score > 0
        ]

        fused = reciprocal_rank_fusion(semantic_ranked_list, keyword_ranked_list)

        results = []
        for idx, rrf_score in fused[:top_k]:
            results.append(
                {
                    "id": self._chunks[idx]["id"],
                    "document": self._chunks[idx]["document"],
                    "metadata": self._chunks[idx]["metadata"],
                    "score": round(rrf_score, 4),
                    "method": "hybrid",
                }
            )
        return results

    def _rerank_results(
        self,
        question: str,
        results: list[dict],
        model_name: str,
        top_k: int,
    ) -> list[dict]:
        """Rerank a set of candidate results with a cross-encoder."""
        docs = [r["document"] for r in results]
        ranked = rerank(question, docs, model_name, top_k)
        return [
            {
                **results[idx],
                "score": round(score, 4),
                "method": results[idx]["method"] + "+rerank",
            }
            for idx, score in ranked
        ]

    def _id_to_index(self, doc_id: str) -> Optional[int]:
        """Map a ChromaDB document ID back to the chunk index."""
        for i, chunk in enumerate(self._chunks):
            if chunk["id"] == doc_id:
                return i
        return None

    def stats(self) -> dict:
        """Return collection stats plus BM25 index info."""
        base = collection_stats(self.collection)
        base["bm25_indexed"] = self.bm25.doc_count
        base["bm25_vocab_size"] = len(self.bm25.doc_freqs)
        return base
