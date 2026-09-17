"""LLM-based answer generation with source citations."""

import os

from dotenv import load_dotenv

load_dotenv()

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Rules:\n"
    "1. Answer ONLY based on the provided context. If the context does not contain "
    "enough information, say so clearly.\n"
    "2. Cite your sources by referencing the source document and chunk number "
    "in square brackets, e.g. [handbook.txt, chunk 3].\n"
    "3. Be concise and direct.\n"
    "4. Do not make up information that is not in the context."
)

RAG_USER_TEMPLATE = (
    "Context:\n"
    "---\n"
    "{context}\n"
    "---\n\n"
    "Question: {question}\n\n"
    "Answer the question based only on the context above. "
    "Cite the source for each claim."
)


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    parts = []
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "unknown")
        source_name = source.split("/")[-1] if "/" in source else source
        chunk_idx = r["metadata"].get("chunk_index", "?")
        similarity = 1 - r["distance"]
        parts.append(
            f"[Source {i}: {source_name}, chunk {chunk_idx}] "
            f"(relevance: {similarity:.2f})\n"
            f"{r['document']}"
        )
    return "\n\n".join(parts)


def build_messages(question: str, results: list[dict]) -> list[dict]:
    """Build the chat messages for the LLM from a question and retrieved results."""
    context = format_context(results)
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_USER_TEMPLATE.format(
            context=context, question=question
        )},
    ]


def get_llm(provider: str = "openai", model: str | None = None):
    """Create a LangChain chat model.

    provider: "openai" or "anthropic"
    model: override the default model name
    """
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            temperature=0,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")


def generate_answer(
    question: str,
    results: list[dict],
    provider: str = "openai",
    model: str | None = None,
) -> dict:
    """Generate an answer using an LLM grounded in retrieved context.

    Returns a dict with keys: answer, sources, messages, model.
    """
    messages = build_messages(question, results)
    llm = get_llm(provider, model)

    from langchain_core.messages import HumanMessage, SystemMessage
    lc_messages = [
        SystemMessage(content=messages[0]["content"]),
        HumanMessage(content=messages[1]["content"]),
    ]
    response = llm.invoke(lc_messages)

    sources = []
    for r in results:
        source = r["metadata"].get("source", "unknown")
        source_name = source.split("/")[-1] if "/" in source else source
        sources.append({
            "source": source_name,
            "chunk_index": r["metadata"].get("chunk_index", None),
            "similarity": round(1 - r["distance"], 3),
            "text_preview": r["document"][:150],
        })

    return {
        "answer": response.content,
        "sources": sources,
        "model": response.response_metadata.get("model_name", str(llm.model)),
    }


def preview_prompt(question: str, results: list[dict]) -> str:
    """Return the full prompt that would be sent to the LLM, without calling it.

    Useful for debugging and understanding what the LLM sees.
    """
    messages = build_messages(question, results)
    lines = []
    for msg in messages:
        lines.append(f"=== {msg['role'].upper()} ===")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)
