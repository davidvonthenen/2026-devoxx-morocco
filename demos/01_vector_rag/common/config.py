"""Configuration loading utilities for vector RAG and its SLM service."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import platform
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    """Runtime configuration for retrieval, the SLM client, and the SLM server."""

    # OpenSearch
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_index: str = "bbc-vector-chunks"

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # SLM server connection (OpenAI-compatible chat API)
    llm_server_url: str = "http://127.0.0.1:8001/v1"
    llm_server_api_key: str = ""
    llm_server_model: str = "local-llm"

    # SLM server runtime: GGUF by default, MLX optional on Apple Silicon
    llm_runtime: str = "gguf"

    # llama.cpp settings used by slm_service.py only
    llama_model_path: str = "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"
    # Context window requested from llama.cpp. The Qwen model may support larger
    # contexts, but this project defaults to 65,536 tokens to keep memory use sane.
    llama_ctx: int = 65536
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    # Number of layers to offload to the GPU. Set to 0 for CPU-only execution.
    llama_n_gpu_layers: int = 20
    llama_n_batch: int = 256                 # llama.cpp prompt-processing batch size
    llama_n_ubatch: Optional[int] = 256      # physical micro-batch; None lets llama.cpp choose
    llama_low_vram: bool = True              # trade speed for lower VRAM usage on constrained GPUs

    # MLX model path used by slm_service.py when LLM_RUNTIME=mlx
    mlx_model_path: str = "Qwen2.5-7B-Instruct-1M-4bit"

    # Server bind address
    server_host: str = "0.0.0.0"
    server_port: int = 8001

    # RAG
    rag_top_k: int = 3
    rag_num_candidates: int = 50




def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _default_llama_n_gpu_layers() -> int:
    """Return a reasonable GPU offload default for the local LLM runtime."""

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return -1
    return 20


def _is_apple_silicon() -> bool:
    """Return True when running on Apple Silicon hardware."""

    return platform.system() == "Darwin" and platform.machine() == "arm64"

def load_settings(env_file: str | None = None) -> Settings:
    """Load runtime settings from environment variables and an optional .env file."""
    # Load an explicit .env first; otherwise check the current working directory
    # and then the project root inferred from this file location.
    if env_file:
        load_dotenv(env_file)
    else:
        for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
            if candidate.exists():
                load_dotenv(str(candidate))
                break

    opensearch_port = _get_int("OPENSEARCH_PORT", Settings.opensearch_port)
    if opensearch_port != 9200:
        print("\n-----------------------------------------------------")
        print(f"WARNING: Using NON STANDARD OpenSearch Port: {opensearch_port}")
        print("Default OpenSearch Port is 9200")
        print("-----------------------------------------------------\n")

    external_ai = os.getenv("USE_EXTERNAL_AI", "false").lower() in ("1", "true", "yes", "on")

    llm_runtime = _get_str("LLM_RUNTIME", "").strip().lower()
    if llm_runtime in {"gguf", "mlx"}:
        llm_runtime = Settings.llm_runtime
    else:
        if _is_apple_silicon():
            llm_runtime = "mlx"

    llm_server_url = os.getenv("LLM_SERVER_URL", Settings.llm_server_url)
    if external_ai:
        llm_server_url = _get_str("EXTERNAL_LLM_URL", Settings.external_base_url)

    llm_server_api_key = os.getenv("LLM_SERVER_API_KEY", Settings.llm_server_api_key)
    if external_ai:
        llm_server_api_key = _get_str("EXTERNAL_LLM_API_KEY", "")

    llm_server_model = os.getenv("LLM_SERVER_MODEL", Settings.llm_server_model)
    if external_ai:
        llm_server_model = os.getenv("EXTERNAL_LLM_MODEL", Settings.external_model)

    llama_ctx = _get_int("LLAMA_CTX", Settings.llama_ctx)
    if external_ai:
        llama_ctx = _get_int("EXTERNAL_LLM_MAX_TOKENS", 127000)

    return Settings(
        opensearch_host=os.getenv("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", Settings.opensearch_index),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),

        llm_runtime=llm_runtime,

        # LLaMA
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=llama_ctx,
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", _default_llama_n_gpu_layers()),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        llama_low_vram=_get_bool("LLAMA_LOW_VRAM", Settings.llama_low_vram),

        # MLX
        mlx_model_path=os.getenv(
            "MLX_MODEL_PATH",
            str(Path.home() / "models" / Settings.mlx_model_path),
        ),

        # External LLM (OpenAI-compatible endpoint)
        llm_server_url=llm_server_url,
        llm_server_api_key=llm_server_api_key,
        llm_server_model=llm_server_model,

        # Server
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),

        rag_top_k=_get_int("RAG_TOP_K", Settings.rag_top_k),
        rag_num_candidates=_get_int("RAG_NUM_CANDIDATES", Settings.rag_num_candidates),
    )


__all__ = ["Settings", "load_settings"]
