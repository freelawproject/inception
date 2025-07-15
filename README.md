<img align="right" src="https://github.com/user-attachments/assets/7ec21ced-43a3-4c72-995c-70c3832d594e" alt="A logo for the project showing a document and some AI sparkles.">

# Inception

Inception is our microservice for generating embeddings from blocks of text.

It is a high-performance FastAPI service that generates text embeddings using SentenceTransformers, specifically designed for processing legal documents and search queries. The service efficiently handles both short search queries and lengthy court opinions, generating semantic embeddings that can be used for document similarity matching and semantic search applications. It includes support for GPU acceleration when available.

The service is optimized to handle two main use cases:
- Embedding search queries: Quick, CPU-based processing for short search queries
- Embedding court opinions: NVIDIA CUDA GPU-accelerated processing for longer legal documents, with intelligent text chunking to maintain context

## Features

- Specialized text embedding generation for legal documents using `freelawproject/modernbert-embed-base_finetune_512`, a `sentence_transformer` model finetuned on top of `nomic-ai/modernbert-embed-base`
- Intelligent text chunking optimized for court opinions, based on sentence boundaries
- Dedicated CPU-based processing for search queries, ensuring fast response times
- GPU acceleration support for processing lengthy court opinions
- Batch processing capabilities for multiple documents
- Comprehensive text preprocessing and cleaning tailored for legal text
- Health check endpoint


## Configuration

The service can be configured through environment variables or a `.env` file. Copy `.env.example` to `.env` to get started:
```bash
cp .env.example .env
```

### Environment Variables

Model Settings:
- `TRANSFORMER_MODEL_NAME`

    Default: `freelawproject/modernbert-embed-base_finetune_512`

    The name or path of the SentenceTransformer model to use for generating embeddings.

- `TRANSFORMER_MODEL_VERSION`

    Default: `main`

    The version of the SentenceTransformer model to use for generating embeddings. Use `main` for the latest version.

- `MAX_TOKENS`

    Default: `512` (Range: 256–10000)

    Maximum number of tokens per chunk when splitting text. If the text exceeds this limit, it is split into multiple chunks based on sentence boundaries. If a sentence exceeds this limit, it is truncated. Sentences are defined by `nltk.tokenize.sent_tokenize`, which follows English heuristics to detect sentence boundaries.

- `OVERLAP_RATIO`

    Default: `0.004` (Range: 0-0.01)

    The ratio to calculate the number of sentences to overlap between chunks when splitting text. Sentences are defined by `nltk.tokenize.sent_tokenize`, which follows English heuristics to detect sentence boundaries. `num_overlap_sentences = int(MAX_TOKENS * OVERLAP_RATIO)`.

- `MIN_TEXT_LENGTH`

    Default: `1`

    The minimum length (in characters) of text required before attempting to process.

- `MAX_QUERY_LENGTH`

    Default: `1000`

    The maximum allowable length (in characters) for a query text.

- `MAX_TEXT_LENGTH`

    Default: `10000000` (characters)

    The maximum allowable length (in characters) for any single text input.

- `MAX_BATCH_SIZE`

    Default: `100`

    The maximum number of items you can process in a single batch.

- `PROCESSING_BATCH_SIZE`

    Default: `8`

    The batch size used internally by the model encoder. This helps to control memory usage and speed when processing multiple chunks or texts.

- `POOL_TIMEOUT`

    Default: `3600` (seconds)

    Timeout for multi-process pool operations. Determines how long worker processes will wait before timing out.

Server Settings:
- `HOST`

    Default: `0.0.0.0`

    The host interface on which the server listens.

- `PORT`

    Default: `8005`

    The port on which the server listens for incoming requests.

- `EMBEDDING_WORKERS`

    Default: `4`

    Number of Gunicorn worker processes for serving the embedding service. Increase if you need higher concurrency.

- `MAX_WORKERS`

    Default: `4`

    Number of Threadpool worker processes for chunking. Increase if you need higher concurrency.

GPU Settings:
- `FORCE_CPU`

    Default: `false`

    Forces the service to run on CPU even if a GPU is available. Useful for debugging or ensuring CPU is selected on query embedding service instances.

Monitoring:
- `ENABLE_METRICS`

    Default: `true`

    Enables Prometheus metrics collection for performance and usage monitoring.

- `SENTRY_DSN`

    Optional

    Sentry DSN for error tracking.

CORS Settings:
- `ALLOWED_ORIGINS`

    A comma-separated list of allowed origins for cross-origin requests.

    Example: `ALLOWED_ORIGINS=https://example.com,https://example2.com`

- `ALLOWED_METHODS`

    A comma-separated list of allowed HTTP methods for cross-origin requests.

    Example: `ALLOWED_METHODS=GET,POST,OPTIONS`

- `ALLOWED_HEADERS`

    A comma-separated list of allowed HTTP headers for cross-origin requests.

    Example: `ALLOWED_HEADERS=Authorization,Content-Type`

See `.env.example` for a complete list of configuration options.


## Installation

This project uses UV for dependency management. To get started:

1. Install [UV](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/freelawproject/inception
   cd inception
   uv sync --extra cpu
   ```
   Use `--extra gpu`  for CUDA GPU support.

## Quick Start

### Running the Service

The easiest way to run the embedding service is using Docker.

Build:
```bash
docker build -t inception:latest --build-arg TARGET_ENV=prod .
```

Run with CPU:
```bash
docker run -d -p 8005:8005 inception
```

Run with GPU:
```bash
docker run -d \
  --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute \
  --shm-size=16g \
  -p 8005:8005 \
  inception
```

Run from hosted image:

with CPU:
```bash
docker run -d -p 8005:8005 freelawproject/inception:v2
```

with GPU:
```bash
docker run -d \
  --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute \
  --shm-size=16g \
  -p 8005:8005 \
  freelawproject/inception:v2
```

For development, run it with docker compose:

with CPU:
```bash
docker compose -f docker-compose.dev.yml up
```

with GPU:
```bash
docker compose -f docker-compose.dev-gpu.yml up
```

To handle more concurrent tasks, increase the number of workers:
```bash
docker run -d -p 8005:8005 -e EMBEDDING_WORKERS=4 freelawproject/inception:v2
```

Check that the service is running:
```bash
curl http://localhost:8005
```
Should return: "Heartbeat detected."

### Running the Service Outside of Docker

1. Activate the venv
```bash
source .venv/bin/activate
```

2. Start the microservice
```bash
uvicorn inception.main:app \
        --host 0.0.0.0 \
        --port 8005 \
        --log-level debug \
        --reload
```

## Running tests

Run all the tests:
```bash
docker exec -it inception-embedding-service pytest tests -v
```

Run tests from a marker:
```bash
docker exec -it inception-embedding-service pytest -m embedding_generation -v
```
See all available markers in [pytest.ini](pytest.ini)

## API Endpoints

### Query Embeddings
Generate embeddings for search queries (CPU-optimized):
```bash
curl 'http://localhost:8005/api/v1/embed/query' \
  -X 'POST' \
  -H 'Content-Type: application/json' \
  -d '{"text": "What are the requirements for copyright infringement?"}'
```

### Document Embeddings
Generate embeddings for court opinions or legal documents (GPU-accelerated when available):
```bash
curl 'http://localhost:8005/api/v1/embed/text' \
  -X 'POST' \
  -H 'Content-Type: text/plain' \
  -d 'The court finds that the defendant...'
```

### Batch Processing
Process multiple documents in one request:
```bash
curl 'http://localhost:8005/api/v1/embed/batch' \
  -X 'POST' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"id": 1, "text": "First court opinion..."},
      {"id": 2, "text": "Second court opinion..."}
    ]
  }'
```

You can interact with the service using any HTTP client. Here's a Python example using the `requests` library:

```python
import requests

# Initialize base URL
base_url = "http://localhost:8005"

# Get embedding for a query.
response = requests.post(
    f"{base_url}/api/v1/embed/query",
    json={"text": "What is copyright infringement?"},
    timeout=1,
)
query_embedding = response.json()["embedding"]

# Get embeddings for a single document.
response = requests.post(
    f"{base_url}/api/v1/embed/text",
    data="The court finds that...",
    timeout=10,
)
doc_embeddings = response.json()["embeddings"]

# Get embeddings for a batch of documents.
response = requests.post(
    f"{base_url}/api/v1/embed/batch",
    json={
            "documents": [
                {"id": 1, "text": "First test document"},
                {"id": 2, "text": "Second test document"},
            ]
        },
    timeout=20,
)
document_1_embeddings = response.json()[0]["embeddings"]
document_2_embeddings = response.json()[1]["embeddings"]
```

## Contributing

We welcome contributions to improve the embedding service!

Please ensure you:
- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Test thoroughly using testing [provided tools](#running-tests).
- Lint tools pass:
```bash
docker exec -it inception-embedding-service pre-commit run --all-files
```
- Use type hints and make sure mypy passes:
```bash
docker exec -it inception-embedding-service mypy inception
```

### Updating Dependencies

1. The best way is to update the dependencies in Docker

Step 1: Build the docker image

with CPU:
```bash
docker compose -f docker-compose.dev.yml build --no-cache
```

with GPU:
```bash
docker compose -f docker-compose.dev-gpu.yml build --no-cache
```

Step 2: Spin-up the docker image

with CPU:
```bash
docker compose -f docker-compose.dev.yml up
```

with GPU:
```bash
docker compose -f docker-compose.dev-gpu.yml up
```

Step 3: Add the new dependencies
```bash
docker exec -it inception-embedding-service uv add "transformers>=4.48.0,<5.0.0"
```

2. Alternatively, add the dependency to your list of dependencies in `pyproject.toml`:
`"transformers>=4.48.0,<5.0.0"`
Then run:
```bash
docker exec -it inception-embedding-service uv lock
```
This will rebuild the `uv.lock` file, adding the new dependency and updating other dependencies if available, while ensuring they match the specified version constraints.

3. Alternatively, use uv to add or update dependencies outside of Docker if dependency conflicts arise when resolving them within Docker. This can happen when the development container platform mismatches the production or testing target platform.

```bash
uv add "transformers>=4.48.0,<5.0.0"
```

```bash
uv lock
```

## Monitoring

The service includes several monitoring endpoints:

- `/health`: Health check endpoint providing service status and GPU information
- `/metrics`: Prometheus metrics endpoint for monitoring request counts and processing times

Example health check:
```bash
curl http://localhost:8005/health
```

Example metrics:
```bash
curl http://localhost:8005/metrics
```

## Requirements

- Python 3.12+
- CUDA-compatible GPU (highly recommended, for long texts embedding)
