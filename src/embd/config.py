"""Configuration loading from config.toml + .env secrets."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class PathsConfig:
    documents_dir: Path
    db_dir: Path


_DEFAULT_TYPES = ["pdf", "epub", "txt", "md", "docx", "xlsx"]


@dataclass
class IngestionConfig:
    chunk_size: int = 2000
    chunk_overlap: int = 200
    enabled_types: list[str] = field(default_factory=lambda: list(_DEFAULT_TYPES))
    # When to OCR embedded images in PDF / DOCX (major speed impact on image-heavy PDFs).
    # "when_no_text" — only if the page/body has no extractable text (typical scanned pages).
    # "always" — OCR every embedded image even when text exists (slow; charts in images).
    # "never" — skip image OCR entirely.
    ocr_embedded_images: str = "when_no_text"
    # "auto" — Surya on MPS/CUDA if available (best quality), else Tesseract (CPU).
    # "surya" — Surya document OCR (requires MPS or CUDA; falls back to Tesseract otherwise).
    # "tesseract" — CPU Tesseract only.
    ocr_backend: str = "auto"
    # Tesseract only (ignored by Surya). Examples: "eng", "deu", "eng+deu". Install packs: apt/brew.
    ocr_tesseract_lang: str = "eng"
    # 0 = no limit. N>0 = OCR at most the N largest embedded images per page (by raw bytes;
    # skips tiny icons/logos after the cap). Cuts runtime when ocr_embedded_images = "always".
    ocr_max_images_per_page: int = 0
    # Glob patterns for directories/files to skip during recursive scanning.
    # Matched against relative paths from documents_dir. Useful for cloned repos.
    ignore_patterns: list[str] = field(default_factory=lambda: [
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        "*.pyc", "*.pyo", "*.o", "*.so", "*.dylib",
    ])


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "auto"    # auto-detects: mps (Apple Silicon), cuda, or cpu


@dataclass
class RetrievalConfig:
    top_k: int = 5
    collection_name: str = "documents"


@dataclass
class LLMConfig:
    backend: str = "mlx"           # "mlx" (Apple-native), "ollama", or "claude"
    mlx_model: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    temperature: float = 0.1
    max_tokens: int = 512
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    claude_model: str = "claude-sonnet-4-6"
    claude_api_key: str | None = None
    # system_prompt_preset: "grounded" = strict doc-only Q&A; "research" = docs + reasoning
    # (suited to Claude + /search). If system_prompt is non-empty, it overrides the preset.
    system_prompt_preset: str = "grounded"
    system_prompt: str | None = None


@dataclass
class SearchConfig:
    searxng_url: str = "http://localhost:8080"
    max_results: int = 5


@dataclass
class ServerConfig:
    """HTTP retrieval API (``embd serve``) for ChatGPT Actions and similar clients."""

    host: str = "0.0.0.0"
    port: int = 8000
    # Prefer EMBD_API_KEY in the environment instead of storing a secret in TOML.
    api_key: str | None = None
    # Public base URL for OpenAPI `servers` (ChatGPT Actions import). Env overrides TOML.
    openapi_base_url: str | None = None


@dataclass
class Config:
    paths: PathsConfig
    ingestion: IngestionConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    search: SearchConfig
    server: ServerConfig


def _env_secret(env_key: str, toml_value: str | None = None) -> str | None:
    """Return env var if set, else TOML fallback, else None."""
    val = os.environ.get(env_key, "").strip()
    if val:
        return val
    if toml_value and str(toml_value).strip():
        return str(toml_value).strip()
    return None


def load_config(config_path: Path = Path("config.toml")) -> Config:
    """Load configuration from TOML + secrets from .env (if present)."""
    load_dotenv(override=False)

    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    p = raw.get("paths", {})
    llm_raw = dict(raw.get("llm", {}))
    llm_raw["claude_api_key"] = _env_secret(
        "ANTHROPIC_API_KEY", llm_raw.get("claude_api_key"),
    )

    server_raw = dict(raw.get("server", {}))
    server_raw["api_key"] = _env_secret(
        "EMBD_API_KEY", server_raw.get("api_key"),
    )
    obu = os.environ.get("EMBD_OPENAPI_BASE_URL", "").strip()
    if obu:
        server_raw["openapi_base_url"] = obu

    return Config(
        paths=PathsConfig(
            documents_dir=Path(p.get("documents_dir", "./documents")),
            db_dir=Path(p.get("db_dir", "./chroma_db")),
        ),
        ingestion=IngestionConfig(**raw.get("ingestion", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        llm=LLMConfig(**llm_raw),
        search=SearchConfig(**raw.get("search", {})),
        server=ServerConfig(**server_raw),
    )
