"""Configuration for the OpenAI-compatible SLM client and model service."""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Settings:
    """Connection details for the query client and runtime settings for the SLM service."""

    # OpenAI-compatible service used by query.py
    llm_server_url: str = "http://127.0.0.1:8001/v1"
    llm_server_api_key: str = ""
    llm_server_model: str = "local-llm"

    # Model runtime used only by slm_service.py (MLX is opt-in)
    llm_runtime: str = "gguf"

    # llama.cpp / GGUF model settings used only by slm_service.py
    llama_model_path: str = "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"
    llama_ctx: int = 65536
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llama_n_gpu_layers: int = 20
    llama_n_batch: int = 256
    llama_n_ubatch: Optional[int] = 256

    # Optional Apple Silicon MLX runtime
    mlx_model_path: str = "Qwen2.5-7B-Instruct-1M-4bit"

    # SLM service bind address
    server_host: str = "0.0.0.0"
    server_port: int = 8001


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None and value.strip() else default


def _default_gpu_layers() -> int:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return -1
    return Settings.llama_n_gpu_layers


def load_settings(env_file: str | None = None) -> Settings:
    """Read explicit environment variables, optionally loading a project .env."""
    if env_file:
        load_dotenv(env_file)
    else:
        for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate)
                break

    runtime = os.getenv("LLM_RUNTIME", Settings.llm_runtime).strip().lower()
    if runtime not in ("gguf", "mlx"):
        raise ValueError("LLM_RUNTIME must be 'gguf' or 'mlx'.")

    return Settings(
        llm_server_url=os.getenv("LLM_SERVER_URL", Settings.llm_server_url),
        llm_server_api_key=os.getenv("LLM_SERVER_API_KEY", Settings.llm_server_api_key),
        llm_server_model=os.getenv("LLM_SERVER_MODEL", Settings.llm_server_model),
        llm_runtime=runtime,
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            os.getenv("MODEL_PATH", str(Path.home() / "models" / Settings.llama_model_path)),
        ),
        llama_ctx=_get_int("LLAMA_CTX", Settings.llama_ctx),
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", _default_gpu_layers()),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        mlx_model_path=os.getenv(
            "MLX_MODEL_PATH", str(Path.home() / "models" / Settings.mlx_model_path)
        ),
        server_host=os.getenv("LLM_SERVER_HOST", os.getenv("SERVER_HOST", Settings.server_host)),
        server_port=_get_int("LLM_SERVER_PORT", _get_int("SERVER_PORT", Settings.server_port)),
    )


__all__ = ["Settings", "load_settings"]
