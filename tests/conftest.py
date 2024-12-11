import pytest
from fastapi.testclient import TestClient
from inception.main import app
from inception.embedding_service import EmbeddingService
from inception.config import Settings
from sentence_transformers import SentenceTransformer
import os
import shutil

@pytest.fixture
def test_settings():
    return Settings(
        transformer_model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        max_words=350,
        min_text_length=1,
        max_batch_size=100,
        pool_timeout=3600,
        force_cpu=True,  # Force CPU for tests
        enable_metrics=True
    )

@pytest.fixture
def test_model():
    return SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

@pytest.fixture
def test_service(test_model, test_settings):
    return EmbeddingService(model=test_model, max_words=test_settings.max_words)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Clean up any test artifacts
    if os.path.exists("tests/test_data/temp"):
        shutil.rmtree("tests/test_data/temp")
