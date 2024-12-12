# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.5 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    # For installing poetry and git-based deps
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && poetry config virtualenvs.create false

WORKDIR /app

# Copy all necessary files
COPY pyproject.toml poetry.lock README.md docker-entrypoint.sh ./
# Copy source code
COPY inception/ inception/

# Use an ARG to control build environment
ARG TARGET_ENV=dev
ENV TARGET_ENV=${TARGET_ENV}

RUN if [ "$TARGET_ENV" = "prod" ]; then \
      poetry install --with gpu --no-interaction --no-ansi; \
    else \
      poetry install --no-interaction --no-ansi; \
    fi

RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/bin/sh","/app/docker-entrypoint.sh"]
