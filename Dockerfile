# embd — retrieval API container (CPU or CUDA)
# MLX is Apple-only and excluded; use Ollama or Claude as the LLM backend.
#
# Build:
#   docker build -t embd .
#
# Run (mount your documents and chroma_db, pass secrets via env):
#   docker run -d --name embd \
#     -p 8000:8000 \
#     -v ./documents:/app/documents \
#     -v ./chroma_db:/app/chroma_db \
#     -v ./config.toml:/app/config.toml:ro \
#     -e EMBD_API_KEY="$(cat .env | grep EMBD_API_KEY | cut -d= -f2)" \
#     -e ANTHROPIC_API_KEY="$(cat .env | grep ANTHROPIC_API_KEY | cut -d= -f2)" \
#     embd

FROM python:3.12-slim AS base

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir '.[docx,xlsx,ocr-cpu]'

COPY config.toml ./

RUN mkdir -p documents chroma_db

EXPOSE 8000

ENTRYPOINT ["embd"]
CMD ["serve"]
