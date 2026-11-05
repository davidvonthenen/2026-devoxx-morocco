PYTHON ?= python
INDEX ?= bbc-vector-chunks
DATA_DIR ?= ./bbc

.PHONY: help ingest query env service

help:
	@echo "Start OpenSearch and the SLM service before querying."
	@echo "Run 'make service' in one terminal, then 'make query' in another."
	@echo "'make ingest' requires ingest.py from the full lab (not included in this archive)."

ingest:
	$(PYTHON) ingest.py

query:
	$(PYTHON) query.py

service:
	$(PYTHON) slm_service.py

env:
	@echo "OPENSEARCH_HOST=$${OPENSEARCH_HOST:-127.0.0.1}"
	@echo "OPENSEARCH_PORT=$${OPENSEARCH_PORT:-9200}"
	@echo "OPENSEARCH_INDEX=$${OPENSEARCH_INDEX:-$(INDEX)}"
	@echo "EMBEDDING_MODEL=$${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
	@echo "LLM_SERVER_URL=$${LLM_SERVER_URL:-http://127.0.0.1:8001/v1}"
	@echo "LLM_SERVER_MODEL=$${LLM_SERVER_MODEL:-local-llm}"
	@echo "LLM_RUNTIME=$${LLM_RUNTIME:-gguf}"
	@echo "LLM_SERVER_HOST=$${LLM_SERVER_HOST:-0.0.0.0}"
	@echo "LLM_SERVER_PORT=$${LLM_SERVER_PORT:-8001}"
	@echo "MLX_MODEL_PATH=$${MLX_MODEL_PATH:-$$HOME/models/Qwen2.5-7B-Instruct-1M-4bit}"
	@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-$$HOME/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf}"
	@echo "LLAMA_CTX=$${LLAMA_CTX:-65536}"
	@echo "LLAMA_N_THREADS=$${LLAMA_N_THREADS:-$$($(PYTHON) -c 'import os; print(os.cpu_count() or 1)')}"
	@echo "LLAMA_N_GPU_LAYERS=$${LLAMA_N_GPU_LAYERS:-20}"
	@echo "LLAMA_N_BATCH=$${LLAMA_N_BATCH:-256}"
	@echo "LLAMA_N_UBATCH=$${LLAMA_N_UBATCH:-256}"
	@echo "LLAMA_LOW_VRAM=$${LLAMA_LOW_VRAM:-true}"
	@echo "RAG_TOP_K=$${RAG_TOP_K:-3}"
	@echo "RAG_NUM_CANDIDATES=$${RAG_NUM_CANDIDATES:-50}"

