FROM python:3.11-slim AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
COPY external/ external/
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .
FROM python:3.11-slim
COPY --from=base /usr/local /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENTRYPOINT ["sabr"]
CMD ["--help"]