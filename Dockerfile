# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04 as base

# Set UV environment variables
# https://hynek.me/articles/docker-uv/
# - Silence uv complaining about not being able to use hard links,
# - tell uv to byte-compile packages for faster application startups
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1


# The installer requires curl (and certificates) to download the latest uv release
RUN apt-get update && apt-get install -y --no-install-recommends && apt-get install build-essential -y \
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

# UV_PROJECT_ENVIRONMENT is required to specify the path for the venv to use for project operations.
# Otherwise .venv is created within the project path.
# https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
ENV UV_PROJECT_ENVIRONMENT=/home/venv
# Set the active virtual environment to the same path as UV_PROJECT_ENVIRONMENT
ENV VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT

RUN uv python install 3.12

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

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
