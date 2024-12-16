# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base

# Set UV environment variables
# https://hynek.me/articles/docker-uv/
# - Silence uv complaining about not being able to use hard links,
# - tell uv to byte-compile packages for faster application startups
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


# The installer requires curl (and certificates) to download the latest uv release
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Download the latest uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Install uv then remove the installer
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# Copy all necessary files
COPY pyproject.toml uv.lock README.md docker-entrypoint.sh ./
# Copy source code
COPY inception/ inception/

RUN uv python install 3.12

RUN uv venv
ENV PATH="/app/.venv/bin:$PATH"

# Use an ARG to control build environment
ARG TARGET_ENV=dev
ENV TARGET_ENV=${TARGET_ENV}

RUN if [ "$TARGET_ENV" = "prod" ]; then \
      uv sync --extra gpu --no-group dev --frozen; \
    else \
      uv sync --extra cpu --frozen; \
    fi

RUN chmod +x /app/docker-entrypoint.sh
ENTRYPOINT ["/bin/sh","/app/docker-entrypoint.sh"]
