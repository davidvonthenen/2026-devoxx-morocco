PYTHON ?= python
DATA_DIR ?= ./bbc
HOST ?= 0.0.0.0
PORT ?= 8000
QUESTION1 ?= How much did OpenAI purchase Windsurf for?
QUESTION2 ?= How much did Google purchase Windsurf for?

.PHONY: help ingest ner-server server slm-service query env

help:
	@echo "Make sure OpenSearch is running and the BM25 chunks are indexed."
	@echo "Start services in separate terminals, then query:"
	@echo "1. make ner-server"
	@echo "2. make slm-service"
	@echo "3. make query"

ingest:
	$(PYTHON) ingest.py

server:
	$(PYTHON) ner_service.py

ner-server: server

slm-service:
	$(PYTHON) slm_service.py

query:
	$(PYTHON) query.py

env:
	@echo "OPENSEARCH_HOST=$${OPENSEARCH_HOST:-localhost}"
	@echo "OPENSEARCH_PORT=$${OPENSEARCH_PORT:-9200}"
	@echo "CHUNK_INDEX_NAME=$${CHUNK_INDEX_NAME:-bbc-bm25-chunks}"
	@echo "NER_URL=$${NER_URL:-http://127.0.0.1:8000/ner}"
	@echo "LLM_SERVER_URL=$${LLM_SERVER_URL:-http://127.0.0.1:8001/v1}"
	@echo "LLM_SERVER_MODEL=$${LLM_SERVER_MODEL:-local-llm}"
	@echo "LLM_RUNTIME=$${LLM_RUNTIME:-gguf}"
	@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-$${MODEL_PATH:-$$HOME/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf}}"
	@echo "LLAMA_CTX=$${LLAMA_CTX:-65536}"
	@echo "LLAMA_N_THREADS=$${LLAMA_N_THREADS:-$$($(PYTHON) -c 'import os; print(max(1, (os.cpu_count() or 4)-1))')}"
	@echo "LLAMA_N_GPU_LAYERS=$${LLAMA_N_GPU_LAYERS:-20}"
	@echo "LLM_SERVER_HOST=$${LLM_SERVER_HOST:-0.0.0.0}"
	@echo "LLM_SERVER_PORT=$${LLM_SERVER_PORT:-8001}"
