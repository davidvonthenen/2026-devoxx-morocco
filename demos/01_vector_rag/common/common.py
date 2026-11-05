"""Shared OpenAI-compatible SLM and vector RAG helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from openai import OpenAI
from opensearchpy import OpenSearch
from opensearchpy.exceptions import RequestError

from .config import Settings, load_settings
from .embeddings import EmbeddingModel, to_list
from .opensearch_client import create_client, ensure_index, knn_search


def load_llm(settings: Settings | None = None) -> OpenAI:
    """Create a client for the OpenAI-compatible SLM endpoint."""
    settings = settings or load_settings()
    return OpenAI(
        base_url=settings.llm_server_url,
        api_key=settings.llm_server_api_key or "not-needed",
    )


def _trim_snippet(text: str, max_length: int = 900) -> str:
    """Trim long snippets before sending retrieved context to the LLM."""
    if len(text) <= max_length:
        return text
    trimmed = text[:max_length]
    last_space = trimmed.rfind(" ")
    if last_space == -1:
        return trimmed + "..."
    return trimmed[:last_space] + "..."


def _build_context_block(hits: List[Dict[str, Any]]) -> str:
    """Format raw OpenSearch hits as the context block supplied to the model.

    Only the chunk text and compact source metadata are included. The full
    OpenSearch response stays out of the prompt so the model receives only
    concise, relevant evidence text.
    """
    parts: List[str] = []
    for idx, hit in enumerate(hits, start=1):
        source = hit.get("_source", {})
        snippet = _trim_snippet(source.get("text", ""))
        parts.append(
            "[DOC {idx} | source: {category}/{title} | path: {path}]\n{snippet}".format(
                idx=idx,
                category=source.get("category", "unknown"),
                title=source.get("title", "unknown"),
                path=source.get("path", ""),
                snippet=snippet,
            )
        )
    return "\n\n".join(parts)


def _rag_hits_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert the OpenSearch response into compact metadata for CLI output."""
    hits: List[Dict[str, Any]] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        hits.append(
            {
                "path": source.get("path", ""),
                "title": source.get("title", ""),
                "category": source.get("category", ""),
                "score": float(hit.get("_score", 0.0)),
            }
        )
    return hits


def _compose_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    """Build the chat messages sent to the SLM service.

    The system prompt tells the model to answer only from retrieved snippets and
    to return ``I don't know.`` when the context is insufficient.
    """
    system_prompt = (
        "You are a fact-focused assistant. "
        "If the answer is not grounded in the snippets, respond with 'I don't know.' "
        "Provide concise answers."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block if context_block else 'No context available.'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _search_vector_context(
    question: str,
    settings: Settings,
    embedder: EmbeddingModel,
    client: OpenSearch,
    *,
    top_k: int,
    num_candidates: int,
) -> Dict[str, Any]:
    """Embed the question and retrieve nearest-neighbor chunks from OpenSearch."""
    embedding = embedder.encode([question])[0]
    try:
        return knn_search(
            client,
            settings.opensearch_index,
            to_list(embedding),
            k=top_k,
            num_candidates=num_candidates,
        )
    except RequestError as exc:
        detail = getattr(exc, "info", None) or str(exc)
        raise RuntimeError(f"OpenSearch query failed: {detail}") from exc


def ask(
    llm: OpenAI,
    question: str,
    *,
    settings: Settings | None = None,
    top_k: int | None = None,
    num_candidates: int | None = None,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_tokens: int = 65536,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Run vector retrieval and answer the question with the SLM service."""
    settings = settings or load_settings()
    embedder = EmbeddingModel(settings)
    client = create_client(settings)

    ensure_index(settings, embedder.dimension)

    search_response = _search_vector_context(
        question,
        settings,
        embedder,
        client,
        top_k=int(top_k if top_k is not None else settings.rag_top_k),
        num_candidates=int(
            num_candidates if num_candidates is not None else settings.rag_num_candidates
        ),
    )

    raw_hits = search_response.get("hits", {}).get("hits", [])
    context_block = _build_context_block(raw_hits)
    messages = _compose_messages(question, context_block)

    llm_response = llm.chat.completions.create(
        model=settings.llm_server_model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    answer = (llm_response.choices[0].message.content or "").strip()

    return answer, _rag_hits_from_response(search_response)


__all__ = ["ask", "load_llm"]
