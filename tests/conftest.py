import os
import shutil

import pytest
from fastapi.testclient import TestClient

from inception.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    # Clean up any test artifacts
    if os.path.exists("tests/test_data/temp"):
        shutil.rmtree("tests/test_data/temp")
