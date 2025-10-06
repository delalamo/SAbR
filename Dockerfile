FROM python:3.11-slim AS build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel && \
    pip install sabr-kit

# -----------------------
# Stage 2: Runtime image
# -----------------------
FROM python:3.11-slim
COPY --from=build /usr/local /usr/local
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
ENTRYPOINT ["sabr"]
CMD ["--help"]