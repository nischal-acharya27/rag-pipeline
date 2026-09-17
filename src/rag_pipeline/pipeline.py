"""End-to-end RAG pipeline orchestration."""

from pathlib import Path

from .chunking import chunk_documents, load_directory
from .embeddings import embed_texts, load_model
from .generator import generate_answer, preview_prompt
from .vectorstore import (
    add_documents,
    collection_stats,
    get_client,
    get_or_create_collection,
    query_collection,
)


class RAGPipeline:
    """A complete RAG pipeline: index documents, then answer questions.

    Usage:
        pipe = RAGPipeline(persist_dir="./chroma_db")
        pipe.index("data/sample/")
        result = pipe.query("What is the remote work policy?")
        print(result["answer"])
    """

    def __init__(
        self,
        persist_dir: str | Path = "./chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        llm_provider: str = "openai",
        llm_model: str | None = None,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_provider = llm_provider
        self.llm_model = llm_model

        self.model = load_model(embedding_model)
        self.client = get_client(persist_dir)
        self.collection = get_or_create_collection(self.client, collection_name)

    def index(
        self,
        data_path: str | Path,
        clear_existing: bool = True,
    ) -> dict:
        """Index all documents from a directory.

        Returns stats about what was indexed.
        """
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

        embeddings = embed_texts(self.model, [c.page_content for c in chunks])

        for chunk in chunks:
            for key, val in list(chunk.metadata.items()):
                if not isinstance(val, (str, int, float, bool)):
                    chunk.metadata[key] = str(val)

        ids = add_documents(self.collection, chunks, embeddings.tolist())

        return {
            "documents_loaded": len(docs),
            "chunks_created": len(chunks),
            "embeddings_stored": len(ids),
        }

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for a question."""
        query_emb = self.model.encode(question).tolist()
        return query_collection(self.collection, query_emb, top_k, where)

    def query(
        self,
        question: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Full RAG pipeline: retrieve context and generate an answer.

        Returns a dict with: answer, sources, model, question, num_results.
        """
        results = self.retrieve(question, top_k, where)
        response = generate_answer(
            question, results, self.llm_provider, self.llm_model
        )
        return {
            "question": question,
            "answer": response["answer"],
            "sources": response["sources"],
            "model": response["model"],
            "num_results": len(results),
        }

    def preview(self, question: str, top_k: int = 5) -> str:
        """Show the full prompt that would be sent to the LLM, without calling it."""
        results = self.retrieve(question, top_k)
        return preview_prompt(question, results)

    def stats(self) -> dict:
        """Return stats about the current collection."""
        return collection_stats(self.collection)
