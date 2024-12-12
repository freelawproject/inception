"""
Tests for the embedding service API endpoints and functionality.
Covers health checks, embedding generation, input validation, batch processing,
GPU memory management, and text processing.
"""

import pytest
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer

from inception.config import settings
from inception.embedding_service import EmbeddingService
from inception.main import app

# Mark all tests with appropriate categories
pytestmark = [pytest.mark.asyncio, pytest.mark.embedding]


# --- Fixtures ---
@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def test_service() -> EmbeddingService:
    """Create an instance of EmbeddingService for testing."""
    model = SentenceTransformer(settings.transformer_model_name)
    max_words = settings.max_words
    return EmbeddingService(model=model, max_words=max_words)


@pytest.fixture
def sample_text() -> str:
    """Load sample text data for testing."""
    with open("tests/test_data/sample_opinion.txt") as f:
        return f.read()


@pytest.fixture
def mock_gpu_cleanup(monkeypatch):
    """Mock GPU memory cleanup for testing."""
    cleanup_called = False

    def mock_cleanup(self):
        nonlocal cleanup_called
        cleanup_called = True

    monkeypatch.setattr(
        "inception.embedding_service.EmbeddingService.cleanup_gpu_memory", mock_cleanup
    )
    return lambda: cleanup_called


# --- Health Check Tests ---
class TestHealthEndpoints:
    """Tests for health check and monitoring endpoints."""

    @pytest.mark.health
    def test_health_check(self, client):
        """Verify health check endpoint returns correct status and information."""
        with TestClient(app) as client:
            response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "gpu_available" in data

    @pytest.mark.health
    def test_heartbeat(self, client):
        """Verify heartbeat endpoint is responding."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == '"Heartbeat detected."'

    @pytest.mark.health
    def test_service_unavailable(self, client, monkeypatch):
        """Verify appropriate response when service is not initialized."""
        monkeypatch.setattr("inception.main.embedding_service", None)
        response = client.post("/api/v1/embed/query", json={"text": "test"})
        assert response.status_code == 503
        assert "service not initialized" in response.json()["detail"].lower()


# --- Embedding Generation Tests ---
class TestEmbeddingGeneration:
    """Tests for embedding generation endpoints."""

    @pytest.mark.embedding_generation
    def test_query_embedding(self, client):
        """Test query embedding generation."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/embed/query",
                json={"text": "What constitutes copyright infringement?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert isinstance(data["embedding"], list)

    @pytest.mark.embedding_generation
    def test_text_embedding(self, client, sample_text):
        """Test document embedding generation."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/embed/text",
                data=sample_text,
                headers={"Content-Type": "text/plain"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert isinstance(data["embeddings"], list)

    @pytest.mark.embedding_generation
    def test_batch_processing(self, client):
        """Test batch processing of multiple documents."""
        batch_request = {
            "documents": [
                {"id": 1, "text": "First test document"},
                {"id": 2, "text": "Second test document"},
            ]
        }
        with TestClient(app) as client:
            response = client.post("/api/v1/embed/batch", json=batch_request)
        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 2
        assert all(isinstance(doc["embeddings"], list) for doc in data)
        assert all(doc["id"] in [1, 2] for doc in data)


# --- Input Validation Tests ---
class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.validation
    def test_query_embedding_validation(self, client):
        """Test query endpoint input validation."""
        test_cases = [
            {
                "name": "empty text",
                "input": {"text": ""},
                "expected_status": 422,
                "expected_error": "text is empty after cleaning.",
            }
        ]

        for case in test_cases:
            with TestClient(app) as client:
                response = client.post("/api/v1/embed/query", json=case["input"])
            assert (
                response.status_code == case["expected_status"]
            ), f"Failed on: {case['name']}"
            assert case["expected_error"] in response.json()["detail"].lower()

    @pytest.mark.validation
    def test_text_embedding_validation(self, client):
        """Test text endpoint input validation."""
        test_cases = [
            {
                "name": "empty text",
                "input": "",
                "expected_status": 422,
                "expected_error": "text length (0) below minimum (1)",
            },
            {
                "name": "invalid UTF-8",
                "input": bytes([0xFF, 0xFE, 0xFD]),
                "expected_status": 422,
                "expected_error": "invalid utf-8",
            },
        ]

        for case in test_cases:
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/embed/text",
                    data=case["input"],
                    headers={"Content-Type": "text/plain"},
                )
            assert (
                response.status_code == case["expected_status"]
            ), f"Failed on: {case['name']}"
            assert case["expected_error"] in response.json()["detail"].lower()

    @pytest.mark.validation
    def test_batch_validation(self, client):
        """Test batch processing validation."""
        test_cases = [
            {
                "name": "batch size limit",
                "input": {
                    "documents": [
                        {"id": i, "text": f"Document {i}"}
                        for i in range(settings.max_batch_size + 1)
                    ]
                },
                "expected_status": 422,
                "expected_error": "batch size exceeds maximum of 100 documents",
            },
            {
                "name": "empty batch",
                "input": {"documents": []},
                "expected_status": 422,
                "expected_error": "empty text list",
            },
            {
                "name": "invalid document",
                "input": {
                    "documents": [
                        {"id": 1, "text": ""},  # Empty text
                        {"id": 2, "text": "Valid document"},
                    ]
                },
                "expected_status": 422,
                "expected_error": "document 1",
            },
        ]

        for case in test_cases:
            with TestClient(app) as client:
                response = client.post("/api/v1/embed/batch", json=case["input"])
            assert (
                response.status_code == case["expected_status"]
            ), f"Failed on: {case['name']}"
            assert case["expected_error"] in response.json()["detail"].lower()


# --- GPU Memory Management Tests ---
class TestGPUMemoryManagement:
    """Tests for GPU memory management."""

    @pytest.mark.gpu
    def test_gpu_cleanup(self, client, mock_gpu_cleanup, sample_text):
        """Test GPU memory cleanup after processing large texts."""
        long_text = sample_text * 100  # Make text long enough to trigger cleanup
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/embed/text",
                data=long_text,
                headers={"Content-Type": "text/plain"},
            )
        assert response.status_code == 200
        assert mock_gpu_cleanup(), "GPU memory cleanup was not called"


# --- Text Processing Tests ---
class TestTextProcessing:
    """Tests for text processing functionality."""

    @pytest.mark.text_processing
    def test_text_chunking(self, test_service, sample_text):
        """Test text chunking functionality."""
        chunks = test_service.split_text_into_chunks(sample_text)

        # Basic properties
        assert len(chunks) > 0, "No chunks were generated"
        assert all(isinstance(chunk, str) for chunk in chunks), "Non-string chunk found"
        assert all(
            len(chunk.split()) <= test_service.max_words for chunk in chunks
        ), "Chunk exceeds maximum word limit"

        # Sentence boundary verification
        for i, chunk in enumerate(chunks):
            assert chunk.strip()[-1] in {
                ".",
                "?",
                "!",
                '"',
            }, f"Chunk {i} does not end with proper punctuation: {chunk[-10:]}"
            assert all(
                sent.strip() for sent in chunk.split(".") if sent.strip()
            ), f"Chunk {i} contains incomplete sentences"

        # Content preservation
        original_content = "".join(sample_text.split())
        chunked_content = "".join("".join(chunks).split())
        assert (
            original_content == chunked_content
        ), "Content was lost or altered during chunking"

        # Chunk transitions
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].strip()
            next_chunk = chunks[i + 1].strip()
            assert current_chunk[-1] in {
                ".",
                "?",
                "!",
                '"',
            }, f"Chunk {i} does not end with proper punctuation"
            assert next_chunk[
                0
            ].isupper(), f"Chunk {i + 1} does not start with uppercase letter"
