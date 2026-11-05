# Section 1 Part 2: Local BM25 RAG Agent

This project provides an end-to-end Retrieval Augmented Generation (RAG) workflow using **BM25-style lexical retrieval** with OpenSearch and a local Small Language Model (SLM) for generation. OpenSearch stores BBC-style documents and overlapping text chunks, and a small Flask service provides Named Entity Recognition (NER) using spaCy.

Unlike the previous version, **`query.py` does not load a GGUF model directly**. Instead, a separate `slm_service.py` process loads the local model and exposes an OpenAI-compatible API. The BM25 RAG application uses the **OpenAI Python SDK** to send the retrieved context and question to that service for inference. The retrieval approach remains BM25, not vector search.

## Prerequisites

- Python 3.10 only
- OpenSearch 3.5 running locally with security disabled for this workshop
- A **modified** BBC dataset located at `./bbc` (original version from [derekgreene/bbc-datasets](https://github.com/derekgreene/bbc-datasets))
- A locally available Small Language Model in `~/models`, using one of these runtimes:
  - **GGUF:** [Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf), saved as `~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf`.
  - **MLX (Apple Silicon):** an MLX-compatible model saved in `~/models/Qwen2.5-7B-Instruct-1M-4bit` (or a different directory set with `MLX_MODEL_PATH`).
- A populated **BM25 chunk index** named `bbc-bm25-chunks`, created using the BM25 ingestion workflow

The SLM service runs as a separate local process. You do **not** need a hosted OpenAI API key to use the default local endpoint.

## Installation

> **IMPORTANT NOTE:** If you are running this hands-on lab in sequence, you may already have completed the environment and OpenSearch installation steps. Use the existing environment and services where appropriate.

If you haven't created a [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [venv](https://docs.python.org/3/library/venv.html), or [uv](https://docs.astral.sh/uv/) virtual environment yet, create one now. The same environment can be used throughout the lab.

Install the Python libraries for this section:

```bash
# If you are using miniconda or venv:
pip install -r requirements.txt

# OR, if you are using uv:
uv pip install -r requirements.txt
```

The dependencies include `openai` for the RAG client and `llama-cpp-python` for the separate model-serving process. If you choose the optional MLX runtime on Apple Silicon, install the compatible `mlx` and `mlx-lm` packages listed as comments in `requirements.txt`.

If you need to start your `podman` VM instance to host containers, run:

```bash
podman machine start
```

If OpenSearch is not running (check with `podman ps`), run the following commands:

```bash
# Create the network for OpenSearch
podman network create opensearch-net

# Deploy the container images
podman run -d \
    --name "opensearch-single" \
    --network "opensearch-net" \
    -p 9200:9200 -p 9600:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -e "cluster.routing.allocation.disk.threshold_enabled=false" \
    -v "$HOME/opensearch/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch/snapshots:/mnt/snapshots" \
    opensearchproject/opensearch:3.5.0

podman run -d \
    --name "opensearch-single-dashboards" \
    --network "opensearch-net" \
    -p 5601:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-single:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.5.0
```

## Step 1: Prerequisite Setup

### Start Our Named Entity Recognition Service

**IN A NEW TERMINAL:** Start the NER service from this project directory, with your Python environment activated:

```bash
# If you are using miniconda or venv:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# OR, if you are using uv:
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# Start the service
python ner_service.py
# Equivalent Makefile target: make ner-server
```

The NER service exposes:

- `POST /ner`, which accepts text and returns normalized entity names and `{name, label}` pairs
- A default endpoint of `http://127.0.0.1:8000/ner`

Both the BM25 ingestion workflow and `query.py` use the NER service, so keep it running while ingesting and querying.

### Start Our Small Language Model (SLM) Service

**IN A SECOND TERMINAL:** Start the external model service from this same directory, with the same Python environment activated:

```bash
# Start the OpenAI-compatible SLM service
python slm_service.py
# Equivalent Makefile target: make slm-service
```

By default, `slm_service.py` loads the GGUF model from:

```text
~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf
```

The model loads when the service starts. Once ready, it exposes:

- `GET http://127.0.0.1:8001/health` for a health check
- `GET http://127.0.0.1:8001/v1/models` for the configured model
- `POST http://127.0.0.1:8001/v1/chat/completions` for chat inference

`query.py` connects to the API using `LLM_SERVER_URL=http://127.0.0.1:8001/v1` and sends requests with model name `local-llm`. The service listens on port `8001` by default and does not require an OpenAI API key for local use.

**Optional configuration:** The settings are defined in `common/config.py` and can be overridden with environment variables or a project `.env` file. The main settings are:

| Setting | Default | Used by |
| --- | --- | --- |
| `LLM_SERVER_URL` | `http://127.0.0.1:8001/v1` | `query.py` (OpenAI SDK base URL) |
| `LLM_SERVER_MODEL` | `local-llm` | Query client and SLM service |
| `LLM_RUNTIME` | `gguf` | SLM service (`gguf` or optional `mlx`) |
| `LLAMA_MODEL_PATH` | `~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf` | SLM service, GGUF model location |
| `LLM_SERVER_PORT` | `8001` | SLM service listening port |

For example, if your GGUF model is stored at another location, set `LLAMA_MODEL_PATH` **before** starting the SLM service. To inspect the configured defaults, run `make env`. If you change the service port, also update `LLM_SERVER_URL` for the query client.

> **NOTE:** The SLM service binds to `0.0.0.0` by default and does not implement API authentication. For a workstation-only demo, set `LLM_SERVER_HOST=127.0.0.1` before starting the service to limit access to the local machine.

### Ingest the British Broadcasting Corporation (BBC) Dataset

> **IMPORTANT NOTE:** This is a separate ingestion step from the vector retrieval exercise. BM25 uses an OpenSearch full-text index and structured entity fields, not vector embeddings.

We need to ingest the modified BBC dataset into OpenSearch. This dataset contains technology news articles used throughout the workshop.

The BM25 query pipeline searches the index named `bbc-bm25-chunks` by default. If it has not already been populated, run the BM25 ingestion application from the full workshop lab:

```bash
# Requires the BM25 ingest.py and ./bbc dataset from the full workshop lab
python ingest.py
```

The original workshop ingestion workflow divides documents into 2048-character windows with a 256-character overlap. Smaller retrieval chunks preserve relevant nearby context while keeping the material sent to the model manageable.

**Remember:** The supplied replacement ZIP does not include `ingest.py`. You can skip this step when `bbc-bm25-chunks` is already populated.

## Step 2: Introduction to Named Entity Recognition

Named Entity Recognition (NER) is a natural language processing (NLP) technique that identifies and classifies named entities in text.

NER is a form of information extraction. It can use statistical models, rules, or custom-trained systems for specialized domains. In this project, spaCy extracts people, locations, organizations, products, and other named entities from the user question and returns normalized text that can be matched against structured OpenSearch fields.

To understand what NER does for us, run the following examples while the NER service is running:

```bash
# Example 1
python ner_client.py --text "My name is David and I am from Long Beach, California."
```

```bash
# Example 2
python ner_client.py --text "Microsoft has its headquarters in Redmond, Washington."
```

```bash
# Example 3
python ner_client.py --text "The Eiffel Tower is located in Paris."
```

## Step 3: Query the RAG Pipeline

For our retrieval pipeline, `query.py` performs the following flow:

1. Loads the project configuration from `common/config.py`, including any supported `.env` overrides.
2. Creates an OpenAI Python SDK client pointed at the **external SLM service**. The GGUF model is loaded by `slm_service.py`, not by `query.py`.
3. Sends the user question to the NER service to extract named entities.
4. Builds an OpenSearch lexical query using the extracted entities. If no entities are found, it falls back to a `match` query over chunk content.
5. Searches the `bbc-bm25-chunks` index by default.
6. Re-ranks OpenSearch candidate chunks using the local `bm25s` library, which is the default path in `ask()`.
7. Assembles the selected chunks into a context block and submits the context and question through `client.chat.completions.create(...)` to the SLM service.
8. Prints the generated answer, query time, and number of chunks provided to the SLM.

This remains **BM25-style lexical retrieval**, not vector retrieval. The model-serving component is now independent of the retrieval application. NER improves explicit-name matching, but the spaCy recognizer is model-based, and neither NER nor BM25 guarantees that every relevant document will be found.

**IN A THIRD TERMINAL:** With OpenSearch, the NER service, and the SLM service running, execute these example queries from the project directory.

First, ask about OpenAI and Windsurf:

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

Note the results of this question.

Now ask about Google and Windsurf:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

Compare the retrieved evidence and answers with the previous vector RAG exercise. Exact names and terms can make BM25 especially effective when those terms appear in the source documents. The result still depends on the indexed articles and which chunks are returned.

To inspect retrieval details and the context being sent to the SLM, optionally run:

```bash
python query.py --question "How much did Google purchase Windsurf for?" --observability
```

The default query path does **not** print the full retrieval diagnostics. The existing `--save-results PATH` option can append compact JSONL retrieval records, and `--top-k` controls how many selected chunks are provided to the SLM.

## Clean Up

> **IMPORTANT NOTE:** Stop both the **NER service** and the **SLM service** with `Ctrl+C` in their respective terminals. OpenSearch can remain running for subsequent lab exercises.

## Takeaways

Using the same news dataset, BM25 can retrieve articles that use the exact names or terms in a question. However, like the previous **vector retrieval** example, BM25 alone does not reconcile events over time or resolve contradictory information across articles. Those are additional reasoning and data-management challenges.

- **BM25/full-text retrieval** works well when important terms appear directly in the source text.
- **Named Entity Recognition** extracts names, organizations, places, products, events, and similar terms to improve matching against structured OpenSearch fields.
- **Chunk-based retrieval** builds the answer context from `bbc-bm25-chunks`, even though the original workshop ingestion workflow also stores full documents.
- **External SLM inference** separates retrieval from model serving: `query.py` uses the OpenAI SDK, while `slm_service.py` owns model loading and generation.
- **Grounded answers still require scrutiny.** Better lexical matching does not automatically establish which of several evolving news reports reflects the final outcome.
