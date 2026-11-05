PYTHON ?= python
DATA_DIR ?= ./bbc
QUESTION1 ?= How much did OpenAI purchase Windsurf for?
QUESTION2 ?= How much did Google purchase Windsurf for?

.PHONY: help ingest ner-server server slm-service query client env

help:
	@echo "Make sure you have the Neo4j container running."
	@echo "Start the NER and SLM services in separate terminals:"
	@echo "1. make ner-server"
	@echo "2. make slm-service"
	@echo "3. make ingest"
	@echo "4. make query"

ingest:
	$(PYTHON) ingest.py

server:
	$(PYTHON) ner_service.py

ner-server: server

slm-service:
	$(PYTHON) slm_service.py

query:
	$(PYTHON) query.py

client: query

env:
	@echo "NEO4J_URI=$${NEO4J_URI:-bolt://localhost:7687}"
	@echo "NEO4J_USER=$${NEO4J_USER:-neo4j}"
	@echo "NEO4J_PASSWORD=$${NEO4J_PASSWORD:-neo4jneo4j}"
	@echo "DATA_DIR=$${DATA_DIR:-$(DATA_DIR)}"
	@echo "NER_SERVICE_URL=$${NER_SERVICE_URL:-http://127.0.0.1:8000/ner}"
	@echo "NER_SERVICE_TIMEOUT=$${NER_SERVICE_TIMEOUT:-8.0}"
	@echo "LLM_SERVER_URL=$${LLM_SERVER_URL:-http://127.0.0.1:8001/v1}"
	@echo "LLM_SERVER_MODEL=$${LLM_SERVER_MODEL:-local-llm}"
	@echo "LLM_RUNTIME=$${LLM_RUNTIME:-gguf}"
	@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-$${MODEL_PATH:-$$HOME/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf}}"
	@echo "LLM_SERVER_HOST=$${LLM_SERVER_HOST:-0.0.0.0}"
	@echo "LLM_SERVER_PORT=$${LLM_SERVER_PORT:-8001}"
