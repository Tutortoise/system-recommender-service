# Build stage
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system numpy scipy wget

COPY recommender/ recommender/
COPY script/convert_embeddings.py script/
COPY config.py ./

RUN python3 script/convert_embeddings.py && \
    rm -rf downloads extracted

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r pyproject.toml

# Runtime stage
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN useradd -m app

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /build/recommender/ /app/recommender/
COPY --from=builder /build/config.py /app/
COPY --from=builder /usr/local/bin/granian /usr/local/bin/

ENV GRANIAN_HOST="0.0.0.0" \
    GRANIAN_PORT="8000" \
    GRANIAN_INTERFACE="asgi" \
    GRANIAN_WORKERS_PER_CORE="2" \
    GRANIAN_MAX_WORKERS="4" \
    GRANIAN_MIN_WORKERS="2" \
    GRANIAN_HTTP="auto" \
    GRANIAN_BACKLOG="1024" \
    GRANIAN_LOG_LEVEL="info" \
    GRANIAN_LOG_ACCESS_ENABLED="true" \
    GRANIAN_THREADING_MODE="workers" \
    GRANIAN_LOOP="auto" \
    GRANIAN_HTTP1_KEEP_ALIVE="true" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    PATH="/home/app/.local/bin:$PATH" \
    PYTHONPATH="/app"

RUN echo '#!/usr/bin/env python3\n\
import multiprocessing\n\
import os\n\
\n\
workers_per_core = int(os.getenv("GRANIAN_WORKERS_PER_CORE", 2))\n\
max_workers = int(os.getenv("GRANIAN_MAX_WORKERS", 4))\n\
min_workers = int(os.getenv("GRANIAN_MIN_WORKERS", 2))\n\
\n\
workers = multiprocessing.cpu_count() * workers_per_core\n\
workers = min(max(workers, min_workers), max_workers)\n\
\n\
print(workers)' > /app/calculate_workers.py && chmod +x /app/calculate_workers.py

RUN echo '#!/bin/bash\n\
\n\
WORKERS=$(python /app/calculate_workers.py)\n\
\n\
echo "Starting Granian with configuration:"\n\
echo "Host: $GRANIAN_HOST"\n\
echo "Port: $GRANIAN_PORT"\n\
echo "Workers: $WORKERS"\n\
echo "Interface: $GRANIAN_INTERFACE"\n\
echo "HTTP Version: $GRANIAN_HTTP"\n\
echo "Threading Mode: $GRANIAN_THREADING_MODE"\n\
echo "Event Loop: $GRANIAN_LOOP"\n\
echo "Log Level: $GRANIAN_LOG_LEVEL"\n\
\n\
cd /app && exec granian "recommender.main:app" \\\n\
--host "$GRANIAN_HOST" \\\n\
--port "$GRANIAN_PORT" \\\n\
--interface "$GRANIAN_INTERFACE" \\\n\
--workers "$WORKERS" \\\n\
--http "$GRANIAN_HTTP" \\\n\
--threading-mode "$GRANIAN_THREADING_MODE" \\\n\
--loop "$GRANIAN_LOOP" \\\n\
--backlog "$GRANIAN_BACKLOG" \\\n\
--log-level "$GRANIAN_LOG_LEVEL" \\\n\
$([ "$GRANIAN_LOG_ACCESS_ENABLED" = "true" ] && echo "--access-log") \\\n\
$([ "$GRANIAN_HTTP1_KEEP_ALIVE" = "true" ] && echo "--http1-keep-alive") \\\n\
"$@"' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

WORKDIR /app
RUN chown -R app:app /app

USER app

EXPOSE $GRANIAN_PORT

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${GRANIAN_PORT}/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]