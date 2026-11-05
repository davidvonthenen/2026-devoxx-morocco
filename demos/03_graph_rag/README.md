# Section 1 Part 3: Local Graph-based RAG Agent

This project provides a compact Graph-based Retrieval Augmented Generation (RAG) workflow. It uses Neo4j as the knowledge graph store, a small Flask-based Named Entity Recognition (NER) service for entity extraction, and a BM25 reranker for retrieved graph candidates. Answer generation is handled by a **separate Small Language Model (SLM) service** that exposes an OpenAI-compatible API.

The retrieval path remains intentionally graph-first: documents are ingested into Neo4j as `Document`, `Paragraph`, and `Entity` nodes. At query time, the user question is sent to the NER service, matching entities are used to retrieve related paragraphs from Neo4j, and BM25 reranks those paragraph candidates. **`query.py` uses the OpenAI Python SDK** to send the final context and question to `slm_service.py` for inference.

## Prerequisites

- Python 3.10 only
- Neo4j 5.26.16 running locally
- A **modified** BBC dataset located at `./bbc` (original version from [derekgreene/bbc-datasets](https://github.com/derekgreene/bbc-datasets))
- A locally available Small Language Model in `~/models`, using one of these runtimes:
  - **GGUF:** [Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf), saved as `~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf`.
  - **MLX (Apple Silicon):** an MLX-compatible model saved in `~/models/Qwen2.5-7B-Instruct-1M-4bit` (or a different directory set with `MLX_MODEL_PATH`).

The GGUF model runs in `slm_service.py`, which uses `llama-cpp-python` for inference by default. An optional MLX runtime is available on Apple Silicon. The dataset and model files are not included in this project ZIP.

## Installation

> **IMPORTANT NOTE:** If you are running this hands-on lab in sequence, you do not need to repeat the installation steps you have already completed. You can reuse the Python environment and NER/SLM services from the previous exercise.

**If** you haven't created a [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [venv](https://docs.python.org/3/library/venv.html), or [uv](https://docs.astral.sh/uv/) virtual environment yet, please do so now. This single environment can be used throughout the entire lab exercise. Call this environment `2026-wearedevs-eu-workshop`.

Install the Python libraries for this section:

```bash
# If you are using miniconda or venv:
pip install -r requirements.txt

# OR, if you are using uv:
uv pip install -r requirements.txt
```

The dependencies include the `openai` Python SDK for the query client and `llama-cpp-python` for the **separate** model-serving process. If using the optional MLX runtime on Apple Silicon, install compatible `mlx` and `mlx-lm` packages as indicated in `requirements.txt`.

**If** you need to start your `podman` VM instance to host containers, run:

```bash
podman machine start
```

> **IMPORTANT NOTE:** Neo4j is new to this section. The previous vector and BM25 exercises used OpenSearch; this exercise uses Neo4j to store document, paragraph, and entity relationships.

If Neo4j is not already running (check with `podman ps`), run the following commands. Create the network only if it does not already exist:

```bash
# Create the network for Neo4j
podman network create graph-net

# Deploy the Neo4j container
podman run -d \
    --name "neo4j-single" \
    --network "graph-net" \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -v "$HOME/neo4j/data:/data" \
    -v "$HOME/neo4j/plugins:/plugins" \
    -v "$HOME/neo4j/import:/var/lib/neo4j/import" \
    neo4j:5.26.16
```

The example container credentials are `neo4j` / `neo4jneo4j`, matching the defaults in `ingest.py` and `query.py`.

## Step 1: Prerequisite Setup

### Start Our Named Entity Recognition Service

**IN A NEW TERMINAL:** Start the NER service from this project directory, with your Python environment activated. If the NER service from the previous exercise is still running on port `8000`, reuse it instead of starting another instance.

```bash
# If you are using the same virtual environment from previous exercises,
# the spaCy model should already be installed. Otherwise, install it:
#
# For miniconda or venv:
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
# OR, if using uv:
# uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# Start the NER service
python ner_service.py
# Equivalent Makefile target: make ner-server
```

The service listens on `http://127.0.0.1:8000/ner`. It extracts named entities with spaCy and returns both a deduplicated entity-name list and detailed `(name, label)` pairs. It does not write to Neo4j; the ingest and query scripts consume the service response and perform graph writes or reads themselves.

Keep the NER service running while ingesting and querying.

### Ingest the British Broadcasting Corporation (BBC) Dataset

> **IMPORTANT NOTE:** This is a new step. We need to create knowledge graphs using the BBC dataset, rather than populate an OpenSearch index.

We need to ingest the modified BBC dataset into Neo4j. This BBC dataset is a collection of technology-related news articles.

**IN ANOTHER TERMINAL:** With Neo4j and the NER service running, execute the following command from this project directory:

```bash
python ingest.py
# Equivalent Makefile target: make ingest
```

The ingest script expects files under `./bbc/<category>/*.txt`. For each file, it treats the first line as the document title and the remaining text as the body. The body is split on blank lines into paragraphs. Each document, paragraph, and detected entity is written to Neo4j, with `MENTIONS` relationships connecting entities to both paragraphs and documents.

**Note:** `ingest.py` clears existing nodes and relationships in the configured Neo4j database before rebuilding the graph. Do not run it against a database containing data you need to keep.

This project does **not** use OpenSearch, vector embeddings, or fixed-size 2048/256 character chunking. Paragraphs are the retrieval units, and BM25 is used only as a reranker over graph-retrieved candidates.

### Start Our Small Language Model (SLM) Service

**IN A SEPARATE TERMINAL:** Start the OpenAI-compatible SLM service from this same project directory, with your Python environment activated. If `slm_service.py` from the previous BM25 exercise is already running on port `8001` using the same configuration, **keep using that instance** instead of starting a second one.

```bash
# Start the external SLM service
python slm_service.py
# Equivalent Makefile target: make slm-service
```

By default, the service loads the GGUF model from:

```text
~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf
```

The model loads when the service starts. When it is ready, the service provides these endpoints:

- `GET http://127.0.0.1:8001/health` for a health check
- `GET http://127.0.0.1:8001/v1/models` to list the configured model
- `POST http://127.0.0.1:8001/v1/chat/completions` for chat inference

You can verify that the service is responding with:

```bash
curl http://127.0.0.1:8001/health
```

`query.py` connects to `http://127.0.0.1:8001/v1` using the OpenAI SDK and sends requests using the model identifier `local-llm`. The local service does not require an OpenAI API key.

**Optional configuration:** Default values are defined in `common/config.py` and can be overridden with environment variables or a project `.env` file.

| Setting | Default | Used by |
| --- | --- | --- |
| `LLM_SERVER_URL` | `http://127.0.0.1:8001/v1` | `query.py` (OpenAI SDK endpoint) |
| `LLM_SERVER_MODEL` | `local-llm` | Query client and SLM service |
| `LLM_RUNTIME` | `gguf` | SLM service (`gguf` or optional `mlx`) |
| `LLAMA_MODEL_PATH` | `~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf` | SLM service (GGUF model) |
| `LLM_SERVER_HOST` | `0.0.0.0` | SLM service bind address |
| `LLM_SERVER_PORT` | `8001` | SLM service listening port |

If your model is stored elsewhere, set `LLAMA_MODEL_PATH` **before** starting `slm_service.py`. To inspect the configured defaults, run `make env`. If you change the service port, also update `LLM_SERVER_URL` for the query client.

> **LOCAL WORKSHOP NOTE:** The SLM service has no API authentication and binds to `0.0.0.0` by default. To accept connections only from your workstation, start it with `LLM_SERVER_HOST=127.0.0.1 python slm_service.py`.

## Step 2: Query the RAG Pipeline

The query script performs this flow:

1. Load the project settings and connect to the external SLM service using the OpenAI Python SDK. The project provides working defaults for the local setup.
2. Send the question to the NER service.
3. Use the extracted `(name, label)` pairs to retrieve candidate paragraphs from Neo4j through graph `MENTIONS` relationships.
4. Rerank those candidates with BM25.
5. Build a compact context block from the highest-ranked paragraphs, including document metadata.
6. Send the context and question to the **external SLM service** with `client.chat.completions.create(...)`.
7. Print the generated answer and query time.

The retrieval logic is unchanged from the original Graph RAG implementation. Neo4j finds paragraphs connected to recognized entities; BM25 orders those paragraphs by lexical relevance. **Only the inference location has changed:** `query.py` no longer loads a GGUF model directly.

Unlike the previous BM25 example, the graph query requires recognized entities to find candidate paragraphs. If no relevant entities are detected or no matching graph relationships are found, it does not fall back to an OpenSearch text search.

**IN A QUERY TERMINAL:** With Neo4j, the NER service, and the SLM service running, execute a question through the graph retrieval and external model pipeline:

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

Note the results of this question.

Try the comparison query:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

Note the results of this question. Compare the evidence retrieved through graph relationships with the results from the vector and BM25 exercises. The answers depend on the article content and which entity-linked paragraphs are selected.

Running `python query.py` without `--question` executes both default demonstration questions.

## Clean Up

> **IMPORTANT NOTE:** Stop both the **NER service** and the **SLM service** with `Ctrl+C` in their respective terminals when you are finished with them. If the services are being reused by another exercise, leave them running instead. Neo4j can remain running for later workshop sections.

## Takeaways

A graph-based RAG pipeline can provide useful retrieval without a vector database when the query contains recognizable entities that map cleanly into the corpus. **HOWEVER**, naively using knowledge graphs can yield the same results as the **vector embedding** and **BM25** examples despite having relationships between data points. Entity-to-paragraph relationships alone do not automatically provide multi-hop reasoning or resolve conflicting facts.

Separating inference into `slm_service.py` changes **how the model is served**, not **which paragraphs are retrieved**. The OpenAI Python SDK now connects the Graph RAG query application to the external SLM, allowing the retrieval logic and model runtime to remain independent.

The big question now is... How do we address the limitations of basic graph retrieval? We will discuss this in Section 2.
