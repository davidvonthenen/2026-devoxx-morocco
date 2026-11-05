#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entry point for the BM25 RAG demo using an external SLM service.

The query path extracts entities from the user's question, searches the
OpenSearch chunk index, optionally re-ranks candidate chunks with the local
``bm25s`` library, and sends the assembled context to an OpenAI-compatible SLM service.

Observability controls default to off:
  --observability           Print query JSON, match summaries, and the LLM prompt
  --save-results PATH       Append compact JSONL records for later inspection

Environment overrides used by the shared helpers:
  OPENSEARCH_HOST=localhost
  OPENSEARCH_PORT=9200
  OPENSEARCH_USER=
  OPENSEARCH_PASS=
  OPENSEARCH_SSL=false
  CHUNK_INDEX_NAME=bbc-bm25-chunks
  SEARCH_SIZE=10
  ALPHA=0.5
  PREFERENCE_TOKEN=governance-audit-v1
  OS_EXPLAIN=false
  OS_PROFILE=false
  NER_URL=http://127.0.0.1:8000/ner
  NER_TIMEOUT_SECS=5
  LLM_SERVER_URL=http://127.0.0.1:8001/v1
  LLM_SERVER_MODEL=local-llm
"""

from __future__ import annotations

import time
import argparse
from typing import List

from common.config import load_settings


##############################################################################
# CLI
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Single-index BM25 RAG query path with optional local BM25 re-ranking.")
    parser.add_argument("--question", help="User question to answer.")
    parser.add_argument("--observability", action="store_true", default=False,
                        help="Print retrieval diagnostics and the prompt sent to the SLM service (default OFF).")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Append compact JSONL result records to this path (default OFF).")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top documents to retrieve and provide to the LLM (default 10).")
    args = parser.parse_args()

    # Load .env before importing common.common, which reads retrieval settings at import.
    settings = load_settings()
    from common.common import ask, load_llm

    llm = load_llm(settings)

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        # Demo questions (safe to replace for your corpus)
        questions = [
            "How much did OpenAI purchase Windsurf for?",
            "How much did Google purchase Windsurf for?",
        ]

    for q in questions:
        t0 = time.time()
        answer, hits = ask(llm, q, observability=args.observability, save_path=args.save_results,
                           top_k=args.top_k, settings=settings)
        dt = time.time() - t0

        print("\n" + "=" * 88)
        print(f"QUESTION: {q}")
        print("=" * 88)
        print("")
        print("=" * 88)
        print(f"ANSWER: {answer}")
        print("=" * 88)
        print(f"\nQuery time: {dt:.2f}s")
        print(f"Docs provided to LLM: {len(hits)}\n\n")


if __name__ == "__main__":
    main()
