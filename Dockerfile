# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.6.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Python and system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        python3.10-venv \
        curl \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

WORKDIR /app

# Copy all necessary files
COPY pyproject.toml poetry.lock README.md ./

# Copy source code
COPY inception/ inception/

# Install dependencies
RUN poetry install --no-interaction --no-ansi

EXPOSE 8005

ENV EMBEDDING_WORKERS=4

# Use shell form to allow environment variable expansion
CMD poetry run gunicorn \
    --workers=$EMBEDDING_WORKERS \
    --worker-class=uvicorn.workers.UvicornWorker \
    --bind=0.0.0.0:8005 \
    --timeout=300 \
    --access-logfile=- \
    --error-logfile=- \
    inception.embed_endpoint:app

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1