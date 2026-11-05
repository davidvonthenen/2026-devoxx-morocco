# Section 1 Part 1: Local Vector RAG Agent

This project provides an end-to-end Retrieval-Augmented Generation (RAG) workflow. It stores BBC article chunks as vectors in OpenSearch 3.5, retrieves relevant chunks with pure vector k-NN search, and answers questions through a separate, locally hosted Small Language Model (SLM) service. The RAG application uses the OpenAI Python SDK to call the service's OpenAI-compatible chat API instead of loading a GGUF model directly into `query.py`.

The main scripts are:

- `ingest.py`: reads `./bbc/<category>/*.txt`, chunks articles, embeds each chunk, and writes the chunk plus metadata to OpenSearch.
- `slm_service.py`: loads a local GGUF model with `llama-cpp-python` or an MLX model on Apple Silicon and exposes an OpenAI-compatible chat completions API.
- `query.py`: embeds the question, retrieves vector hits from OpenSearch, builds a grounded prompt, sends it to the SLM service using the OpenAI Python SDK, and prints the answer plus retrieved hit metadata.

## Prerequisites

- Python 3.10 only
- OpenSearch 3.5 running locally with security disabled.
- Podman for running the local OpenSearch containers.
- A **modified** BBC dataset located at `./bbc` (original version from [derekgreene/bbc-datasets](https://github.com/derekgreene/bbc-datasets)). The dataset is not included in the project archive.
- A locally available Small Language Model in `~/models`, using one of these runtimes:
  - **GGUF:** [Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf), saved as `~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf`.
  - **MLX (Apple Silicon):** an MLX-compatible model saved in `~/models/Qwen2.5-7B-Instruct-1M-4bit` (or a different directory set with `MLX_MODEL_PATH`).

The SLM service runs as a separate local process. You do **not** need a hosted OpenAI API key to use the default local endpoint.

## Installation

If you haven't created a [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) or [venv](https://docs.python.org/3/library/venv.html) virtual environment yet, please do so now. This single environment can be used throughout this entire lab exercise. Call this environment `2026-wearedevs-eu-workshop`.

Install the Python libraries used in this section:

```bash
# If you are using miniconda or venv
pip install -r requirements.txt

# OR, if using uv
uv pip install -r requirements.txt
```

The supplied `requirements.txt` includes the OpenAI SDK, Flask, the embedding and OpenSearch dependencies, and `llama-cpp-python` for GGUF serving. If you are using the optional **MLX runtime on Apple Silicon**, install its additional dependencies as well (they are commented out in `requirements.txt`):

```bash
pip install mlx==0.32.0 mlx-lm==0.31.3
```

Start your `podman` VM instance to host containers, if it isn't already running:

```bash
podman machine start
```

If OpenSearch isn't running (check with `podman ps`), you can deploy the containers with the following commands:

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

## Step 1: Ingest the BBC Dataset

### Ingest the British Broadcasting Corporation (BBC) Dataset

The ingest script reads the BBC dataset from `./bbc`, chunks each `.txt` document, embeds each chunk, and stores the chunks in the OpenSearch index named `bbc-vector-chunks` by default.

The default embedding model is [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). The code normalizes embeddings before storing them in a `knn_vector` field configured for cosine similarity.

```bash
python ingest.py
```

Before re-indexing a document, the ingest process deletes existing chunks for that document path so repeated runs replace old chunks instead of duplicating them.

## Step 2: Query the RAG Pipeline

Before running a query, start the external SLM service in a **separate terminal**, using the same Python environment and project directory.

**Terminal 1: Start the SLM service**

```bash
python slm_service.py
```

The service loads the configured model and starts an OpenAI-compatible API at `http://127.0.0.1:8001/v1`. By default, it selects **MLX on Apple Silicon** and **GGUF on other platforms**. You can select the runtime explicitly when starting the service:

```bash
# Serve the GGUF model using llama-cpp-python
LLM_RUNTIME=gguf python slm_service.py

# OR, on Apple Silicon, serve the MLX model
LLM_RUNTIME=mlx python slm_service.py
```

The default model locations are `~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf` for GGUF and `~/models/Qwen2.5-7B-Instruct-1M-4bit` for MLX. Override these paths with `LLAMA_MODEL_PATH` or `MLX_MODEL_PATH` if needed.

After the server starts, verify its health endpoint:

```bash
curl http://127.0.0.1:8001/health
```

**Terminal 2: Query the RAG agent**

`query.py` performs one RAG pass for each question:

1. Connect to the SLM service with the OpenAI Python SDK (default `http://127.0.0.1:8001/v1`, model name `local-llm`).
2. Embed the user question with the configured sentence-transformers model.
3. Retrieve the top-k chunks from OpenSearch using vector k-NN against the `embedding` field (three chunks by default).
4. Build a prompt containing the retrieved context, source metadata, and question.
5. Ask the SLM to answer using the retrieved snippets, correcting an inaccurate premise when the context supports the correction. The client requests a concise response with `temperature=0` and up to 256 output tokens.
6. Print the answer, retrieved hit metadata, and query timing.

The defaults come from `common/config.py`. To connect the client to a different OpenAI-compatible service, set `LLM_SERVER_URL`, `LLM_SERVER_MODEL`, and, if required, `LLM_SERVER_API_KEY`. No additional query flags are needed.

Let's run our first question on the RAG agent:

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

Note the results of this question. The older article in the dataset reports a proposed **$3 billion acquisition** by OpenAI; it does not establish that the acquisition was completed.

Now, run the next query below:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

Note the results of this question. When the relevant article is retrieved, the answer should identify the reported **$2.4 billion licensing and talent arrangement** and clarify that Google did **not** purchase Windsurf outright.

You can also run `python query.py` with no arguments to execute both demonstration questions.

## Takeaways

Using a single news dataset, pure vector semantic similarity search can surface articles that describe different transactions, reported proposals, and events at different points in time. The retrieved text may be relevant without representing the most recent or final outcome.

The dataset includes news articles as they happened over a period of time, without reconciling the information contained within.

The workshop's example timeline illustrates the distinction:

- **May 6, 2025:** Reports described a proposed acquisition of Windsurf by OpenAI for approximately $3 billion.
- **July 12, 2025:** Reports described Google's agreement to pay approximately $2.4 billion to license Windsurf technology and hire key employees, not to acquire the company outright.
- **July 14, 2025:** Reports described Cognition's agreement to acquire Windsurf following the Google licensing and talent arrangement.

> **NOTE:** An answer of "I don't know" can be appropriate when the relevant information is missing from the retrieved snippets. If the right article **is** present in the context but the model still says "I don't know," the problem may be in prompting or answer generation rather than retrieval. In this case, there is conflicting information and the prompt doesn't instruct the LLM how to resolve the conflict.

Both the **$3 billion OpenAI report** and the **$2.4 billion Google report** can appear in a vector search, but they describe different proposed or negotiated arrangements. The model must preserve those distinctions rather than presenting them as contradictory prices for the same completed acquisition.

This is the limitation we want to explore: a RAG system should not merely retrieve semantically similar documents. It must also produce answers that are faithful to the retrieved evidence and sensitive to how facts and reported events change over time.
