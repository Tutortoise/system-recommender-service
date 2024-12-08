FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# Install core dependencies first
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system numpy scipy wget

# Copy application code
COPY recommender/ recommender/
COPY script/ script/
COPY config.py ./

# Set up embeddings
RUN python3 script/convert_embeddings.py

# Create setup.py for project installation
RUN echo 'from setuptools import setup, find_packages\n\
setup(\n\
    name="system-recommendation",\n\
    version="0.1.0",\n\
    packages=find_packages(),\n\
    install_requires=[],\n\
)\n' > setup.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m app

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /build/recommender/embedding_model/ /app/recommender/embedding_model/

WORKDIR /app
RUN chown -R app:app /app

USER app

COPY --chown=app:app config.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/app/.local/bin:$PATH"

EXPOSE 8000

CMD ["python", "-m", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8000", "recommender.main:app"]