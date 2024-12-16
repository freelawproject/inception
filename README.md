# inception
Our microservice for generating embeddings from blocks of text
# Inception v2 - Text Embedding Service

A high-performance FastAPI service for generating text embeddings using SentenceTransformers, specifically designed for processing legal documents and search queries. The service efficiently handles both short search queries and lengthy court opinions, generating semantic embeddings that can be used for document similarity matching and semantic search applications. It includes support for GPU acceleration when available.

The service is optimized to handle two main use cases:
- Embedding search queries: Quick, CPU-based processing for short search queries
- Embedding court opinions: GPU-accelerated processing for longer legal documents, with intelligent text chunking to maintain context

## Features

- Specialized text embedding generation for legal documents using the `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- Intelligent text chunking optimized for court opinions, based on sentence boundaries
- Dedicated CPU-based processing for search queries, ensuring fast response times
- GPU acceleration support for processing lengthy court opinions
- Batch processing capabilities for multiple documents
- Comprehensive text preprocessing and cleaning tailored for legal text
- Health check endpoint

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

Run:
```bash
docker run -d -p 8005:8005 inception
```

Run from hosted image:
```bash
docker run -d -p 8005:8005 freelawproject/inception:v2
```

For development run it with docker compose:
```bash
docker compose -f docker-compose.dev.yml up
```

To handle more concurrent tasks, increase the number of workers:
```bash
docker run -d -p 8005:8005 -e EMBEDDING_WORKERS=4 freelawproject/inception:v2
```

Test that the service is running:
```bash
curl http://localhost:8005
# Should return: "Heartbeat detected."
```

### Using the Service

You can interact with the service using any HTTP client. Here's a Python example using the `requests` library:

```python
import requests

# Initialize base URL
base_url = "http://localhost:8005"

# Get embedding for a query
response = requests.post(
    f"{base_url}/api/v1/embed/query",
    json={"text": "What is copyright infringement?"}
)
query_embedding = response.json()["embedding"]

# Get embeddings for a document
response = requests.post(
    f"{base_url}/api/v1/embed/text",
    data="The court finds that..."
)
doc_embeddings = response.json()["embeddings"]
```

For more examples and detailed API documentation, check the API endpoints section below.

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

## Configuration

The service can be configured through environment variables or a `.env` file. Copy `.env.example` to `.env` to get started:
```bash
cp .env.example .env
```

### Environment Variables

Model Settings:
- `TRANSFORMER_MODEL_NAME`: Model to use (default: "sentence-transformers/multi-qa-mpnet-base-dot-v1")
- `MAX_WORDS`: Maximum words per chunk (default: 350)

Server Settings:
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8005)
- `EMBEDDING_WORKERS`: Number of Gunicorn workers (default: 4)

GPU Settings:
- `FORCE_CPU`: Force CPU usage even if GPU is available (default: false)

Monitoring:
- `SENTRY_DSN`: Sentry DSN for error tracking (optional)
- `ENABLE_METRICS`: Enable Prometheus metrics (default: true)

CORS Settings:
- `ALLOWED_ORIGINS`: Comma-separated list of allowed origins
- `ALLOWED_METHODS`: Comma-separated list of allowed methods
- `ALLOWED_HEADERS`: Comma-separated list of allowed headers

See `.env.example` for a complete list of configuration options.

## Development and Testing

For development setup and testing instructions, see [DEVELOPING.md](DEVELOPING.md).

## Contributing

We welcome contributions to improve the embedding service!

1. For development setup, see [DEVELOPING.md](DEVELOPING.md)
2. For submitting changes, see [SUBMITTING.md](SUBMITTING.md)

Please ensure you:
- Follow the existing code style
- Add tests for new features
- Update documentation as needed
- Test thoroughly using provided tools:
  ```bash
  # Run tests
  docker exec -it inception-embedding-service pytest tests -v
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

- Python 3.8+
- CUDA-compatible GPU (highly recommended, for long texts embedding)