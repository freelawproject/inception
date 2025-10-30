"""
Tests for the embedding service API endpoints and functionality.
Covers health checks, embedding generation, input validation, batch processing,
GPU memory management, and text processing.
"""

from http import HTTPStatus
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from inception.config import settings
from inception.embedding_service import EmbeddingService
from inception.main import app, handle_nltk_download

# Mark all tests with appropriate categories
pytestmark = [pytest.mark.embedding]


@pytest.fixture
def test_embedding_service():
    def _create_service(
        max_tokens: int = None, overlap_ratio: float = None
    ) -> EmbeddingService:
        """Create an instance of EmbeddingService with dynamic parameters for testing."""
        model = SentenceTransformer(
            settings.transformer_model_name,
            revision=settings.transformer_model_version,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            settings.transformer_model_name
        )

        return EmbeddingService(
            model=model,
            tokenizer=tokenizer,
            max_tokens=(
                max_tokens if max_tokens is not None else settings.max_tokens
            ),
            overlap_ratio=(
                overlap_ratio
                if overlap_ratio is not None
                else settings.overlap_ratio
            ),
            processing_batch_size=settings.processing_batch_size,
            max_workers=settings.max_workers,
        )

    return _create_service


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
        "inception.embedding_service.EmbeddingService.cleanup_gpu_memory",
        mock_cleanup,
    )
    return lambda: cleanup_called


class TestNLTKDownload:
    """Tests for NLTK punkt tokenizer download and handling."""

    @pytest.mark.nltk
    def test_resource_found_and_read_success(self):
        """Test when the resource is found and can be read successfully."""
        with mock.patch("nltk.data.find") as mock_find:
            mock_find.return_value.open.return_value.read.return_value = "test"
            handle_nltk_download("punkt")
            mock_find.assert_called_once_with("punkt")

    @pytest.mark.nltk
    def test_resource_not_found_download(self):
        """Test when the resource is not found and needs to be downloaded."""
        with (
            mock.patch("nltk.data.find", side_effect=LookupError),
            mock.patch("nltk.download") as mock_download,
        ):
            handle_nltk_download("punkt")
            mock_download.assert_called_once_with("punkt", quiet=True)

    @pytest.mark.nltk
    def test_invalid_resource_removal_and_redownload(self):
        """Test when the resource is invalid and needs to be removed and redownloaded."""
        with (
            mock.patch("nltk.data.find", side_effect=OSError),
            mock.patch("shutil.rmtree") as mock_remove,
            mock.patch("nltk.download") as mock_download,
        ):
            handle_nltk_download("punkt")
            mock_remove.assert_called_once_with("punkt")
            mock_download.assert_called_once_with("punkt", quiet=True)


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

        assert response.status_code == HTTPStatus.OK
        data = response.json()
        assert "embedding" in data
        assert isinstance(data["embedding"], list)

    @pytest.mark.embedding_generation
    def test_text_embedding(self, client, sample_text):
        """Test document embedding generation."""
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/embed/text",
                content=sample_text,
                headers={"Content-Type": "text/plain"},
            )
        assert response.status_code == HTTPStatus.OK
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
        assert response.status_code == HTTPStatus.OK
        data = response.json()

        assert isinstance(data, list)
        assert len(data) == 2
        assert all(isinstance(doc["embeddings"], list) for doc in data)
        assert all(doc["id"] in [1, 2] for doc in data)


class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.validation
    def test_query_embedding_validation(self, client):
        """Test query endpoint input validation."""

        long_query = (
            "Pellentesque tellus felis cursus id velit ac feugiat rutrum massa Mauris dapibus fermentum sagittis Donec viverra mauris a velit ac quam consectetur, a facilisis enim eleifend."
            * 10
        )
        test_cases = [
            {
                "name": "short text",
                "input": {"text": ""},
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": "text length (0) below minimum (1)",
            },
            {
                "name": "empty text",
                "input": {"text": "ñ😊"},
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": "text is empty after cleaning",
            },
            {
                "name": "query too long",
                "input": {"text": long_query},
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": f"query length ({len(long_query)}) exceeds maximum ({settings.max_query_length})",
            },
        ]

        for case in test_cases:
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/embed/query", json=case["input"]
                )
            assert response.status_code == case["expected_status"], (
                f"Failed on: {case['name']}"
            )
            assert case["expected_error"] in response.json()["detail"].lower()

    @pytest.mark.validation
    def test_text_embedding_validation(self, client):
        """Test text endpoint input validation."""
        test_cases = [
            {
                "name": "empty text",
                "input": "",
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": "text length (0) below minimum (1)",
            },
            {
                "name": "invalid UTF-8",
                "input": bytes([0xFF, 0xFE, 0xFD]),
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": "invalid utf-8",
            },
        ]

        for case in test_cases:
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/embed/text",
                    content=case["input"],
                    headers={"Content-Type": "text/plain"},
                )
            assert response.status_code == case["expected_status"], (
                f"Failed on: {case['name']}"
            )
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
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": "batch size exceeds maximum of 100 documents",
            },
            {
                "name": "empty batch",
                "input": {"documents": []},
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": "empty text dict",
            },
            {
                "name": "invalid document",
                "input": {
                    "documents": [
                        {"id": 1, "text": ""},  # Empty text
                        {"id": 2, "text": "Valid document"},
                    ]
                },
                "expected_status": HTTPStatus.UNPROCESSABLE_ENTITY,
                "expected_error": "document 1",
            },
        ]

        for case in test_cases:
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/embed/batch", json=case["input"]
                )
            assert response.status_code == case["expected_status"], (
                f"Failed on: {case['name']}"
            )
            assert case["expected_error"] in response.json()["detail"].lower()


class TestGPUMemoryManagement:
    """Tests for GPU memory management."""

    @pytest.mark.gpu
    def test_gpu_cleanup(self, client, mock_gpu_cleanup, sample_text):
        """Test GPU memory cleanup after processing large texts."""
        long_text = (
            sample_text * 100
        )  # Make text long enough to trigger cleanup
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/embed/text",
                content=long_text,
                headers={"Content-Type": "text/plain"},
            )
        assert response.status_code == HTTPStatus.OK
        assert mock_gpu_cleanup(), "GPU memory cleanup was not called"


class TestTextProcessing:
    """Tests for text processing functionality."""

    @pytest.mark.text_processing
    def test_text_chunking(self, test_embedding_service, sample_text):
        """Test text chunking functionality."""
        test_service = test_embedding_service()
        chunks = test_service.split_text_into_chunks(sample_text)
        tokenizer = test_service.tokenizer

        # Basic properties
        assert test_service.max_tokens == settings.max_tokens, (
            "Incorrect max_tokens in test_service"
        )
        assert test_service.num_overlap_sentences == int(
            test_service.max_tokens * settings.overlap_ratio
        ), "Incorrect overlap_ratio in test_service"
        assert len(chunks) > 0, "No chunks were generated"
        assert all(isinstance(chunk, str) for chunk in chunks), (
            "Non-string chunk found"
        )
        assert all(
            len(tokenizer.encode(chunk)) <= test_service.max_tokens
            for chunk in chunks
        ), "Chunk exceeds maximum token limit"

        # Lead text verification
        for i, chunk in enumerate(chunks):
            assert chunk[:17] == "search_document: ", (
                f"Chunk {i} does not begin with proper lead text: {chunk[:10]}"
            )

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
        chunks = [s.replace("search_document: ", "").strip() for s in chunks]
        original_content = "".join(sample_text.split())
        chunked_content = "".join("".join(chunks).split())
        assert original_content == chunked_content, (
            "Content was lost or altered during chunking"
        )

        # Chunk transitions
        for i, chunk in enumerate(chunks):
            assert chunk[-1] in {
                ".",
                "?",
                "!",
                '"',
            }, f"Chunk {i} does not end with proper punctuation: {chunk[-10:]}"
            assert chunk[0].isupper(), (
                f"Chunk {i} does not start with uppercase letter: {chunks[:10]}"
            )


class TestTextTruncation:
    """Tests for opinion chunking functionality with truncation and no overlap."""

    @pytest.mark.text_truncation
    def test_text_truncation(
        self,
        test_embedding_service,
        sample_text,
        max_tokens=15,
        overlap_ratio=0,
    ):
        """Test text truncation functionality."""
        test_service = test_embedding_service(
            max_tokens=max_tokens, overlap_ratio=overlap_ratio
        )
        chunks = test_service.split_text_into_chunks(sample_text)
        tokenizer = test_service.tokenizer

        # Basic properties
        assert test_service.max_tokens == max_tokens, (
            "Incorrect max_tokens in test_service"
        )
        assert test_service.num_overlap_sentences == int(
            max_tokens * overlap_ratio
        ), "Incorrect overlap_ratio in test_service"
        assert len(chunks) > 0, "No chunks were generated"
        assert all(isinstance(chunk, str) for chunk in chunks), (
            "Non-string chunk found"
        )
        assert all(
            len(tokenizer.encode(chunk)) <= test_service.max_tokens
            for chunk in chunks
        ), "Chunk exceeds maximum token limit"

        # Lead text verification
        for i, chunk in enumerate(chunks):
            assert chunk[:17] == "search_document: ", (
                f"Chunk {i} does not begin with proper lead text: {chunk[:10]}"
            )

        original_content = sent_tokenize(sample_text)
        chunks = [s.replace("search_document: ", "").strip() for s in chunks]

        # Sentence truncation verification
        for i, chunk in enumerate(chunks):
            if i == 5:
                assert chunk.strip()[-1] == ".", (
                    f"Full sentence chunk does not end with proper punctuation: {chunk[-10:]}"
                )
            else:
                assert chunk.strip()[-1] not in {
                    ".",
                    "?",
                    "!",
                    '"',
                }, (
                    f"Truncated sentence chunk should not end with punctuation: {chunk[-10:]}"
                )

        # Content preservation & transition
        assert len(original_content) == len(chunks), (
            "Sentence was lost during chunking"
        )

        for i in range(len(chunks) - 1):
            assert (
                original_content[i][:10].strip() == chunks[i][:10].strip()
            ), "Content was altered during chunking"
            next_chunk = chunks[i + 1].strip()
            assert next_chunk[0].isupper(), (
                f"Chunk {i + 1} does not start with uppercase letter"
            )


class TestSentenceOverlap:
    """Tests for opinion chunking functionality with overlap and no truncation."""

    @pytest.mark.sentence_overlap
    def test_sentence_overlap(
        self,
        test_embedding_service,
        sample_text,
        max_tokens=200,
        overlap_ratio=0.005,
    ):
        """Test text chunking functionality."""
        test_service = test_embedding_service(
            max_tokens=max_tokens, overlap_ratio=overlap_ratio
        )
        chunks = test_service.split_text_into_chunks(sample_text)
        tokenizer = test_service.tokenizer

        # Basic properties
        assert test_service.max_tokens == max_tokens, (
            "Incorrect max_tokens in test_service"
        )
        assert test_service.num_overlap_sentences == int(
            max_tokens * overlap_ratio
        ), "Incorrect overlap_ratio in test_service"
        assert len(chunks) > 0, "No chunks were generated"
        assert all(isinstance(chunk, str) for chunk in chunks), (
            "Non-string chunk found"
        )
        assert all(
            len(tokenizer.encode(chunk)) <= test_service.max_tokens
            for chunk in chunks
        ), "Chunk exceeds maximum token limit"

        # Lead text verification
        for i, chunk in enumerate(chunks):
            assert chunk[:17] == "search_document: ", (
                f"Chunk {i} does not begin with proper lead text: {chunk[:10]}"
            )

        original_content = sent_tokenize(sample_text)
        chunks = [s.replace("search_document: ", "").strip() for s in chunks]

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
            assert chunk[0].isupper(), (
                f"Chunk {i} does not start with uppercase letter: {chunk[:10]}"
            )

        # Content preservation
        assert original_content[0][:10].strip() == chunks[0][:10].strip(), (
            "Beginning of the content was altered during chunking"
        )

        assert (
            original_content[-1][-10:].strip() == chunks[-1][-10:].strip()
        ), "Ending of the content was altered during chunking"

        # Sentence overlap
        for i in range(len(chunks) - 1):
            assert (
                sent_tokenize(chunks[i])[-1].strip()
                == sent_tokenize(chunks[i + 1])[0].strip()
            ), "Sentence overlap failed during chunking"
