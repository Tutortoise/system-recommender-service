FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cmake \
    libboost-all-dev \
    libfmt-dev \
    libc6-dev \
    libspdlog-dev \
    libeigen3-dev \
    zlib1g-dev \
    cython3 \
    gcc \
    g++ \
    python3-numpy \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    CFLAGS="-fPIC" \
    CXXFLAGS="-fPIC" \
    PYTHONPATH=/usr/local/lib/python3.10/site-packages

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system numpy scipy cython

# Build gensim from source
WORKDIR /build/gensim
RUN git clone --depth 1 --branch 4.3.3 https://github.com/RaRe-Technologies/gensim.git . && \
    python3 setup.py build_ext --inplace && \
    python3 setup.py install

# Test gensim installation
RUN python3 -c "from gensim.models import Word2Vec; print('gensim installation successful')"

# Install remaining dependencies
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i '/gensim/d' pyproject.toml && \
    uv pip install --system -r pyproject.toml

# Create separate directory for project installation
WORKDIR /build/project
COPY recommender/ recommender/
COPY config.py ./

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
    libboost-python-dev \
    libgomp1 \
    libblas3 \
    liblapack3 \
    libatlas3-base \
    libgfortran5 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m app

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/
COPY --from=builder /build/gensim/build/ /build/gensim/build/
COPY --from=builder /build/gensim/gensim/ /build/gensim/gensim/

WORKDIR /app
RUN chown -R app:app /app

USER app

COPY --chown=app:app config.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/app/.local/bin:$PATH" \
    PYTHONPATH="/build/gensim:/usr/local/lib/python3.10/site-packages" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu"

EXPOSE 8000

CMD ["python", "-m", "granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8000", "recommender.main:app"]
