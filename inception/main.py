import asyncio
import os
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import nltk
import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sentence_transformers import SentenceTransformer
from sentry_sdk.integrations.fastapi import FastApiIntegration
from transformers import AutoTokenizer

from inception import routes
from inception.config import settings
from inception.embedding_service import EmbeddingService
from inception.utils import logger

try:
    nltk.data.find("tokenizers/punkt_tab")
except (LookupError, zipfile.BadZipFile):
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
except FileExistsError:
    nltk.data.find("tokenizers/punkt_tab")


# Initialize Sentry
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)

embedding_service = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle application startup and shutdown.

    :param app: The FastAPI application instance.
    :return: None
    """
    global embedding_service
    max_retries = 3
    retry_delay = 5  # seconds

    # Startup
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Attempting to initialize embedding service (attempt {attempt + 1}/{max_retries})"
            )
            model = SentenceTransformer(
                settings.transformer_model_name,
                revision=settings.transformer_model_version,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                settings.transformer_model_name
            )
            embedding_service = EmbeddingService(
                model=model,
                tokenizer=tokenizer,
                max_tokens=settings.max_tokens,
                overlap_ratio=settings.overlap_ratio,
                processing_batch_size=settings.processing_batch_size,
                max_workers=settings.max_workers,
            )
            logger.info("Embedding service initialized successfully")
            break
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
            )
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                sentry_sdk.capture_exception(e)
                raise RuntimeError("Failed to initialize embedding service")

    try:
        # Allows the application to run between startup and shutdown.
        yield
    finally:
        # Shutdown
        try:
            if embedding_service:
                embedding_service = None
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(f"Error during shutdown: {str(e)}")


app = FastAPI(
    title="Inception v0",
    description="Service for generating embeddings from queries and opinions",
    version="0.0.1",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=os.getenv("ALLOWED_METHODS", "*").split(","),
    allow_headers=os.getenv("ALLOWED_HEADERS", "*").split(","),
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Inception v2",
        version="2.0.0",
        description="Service for generating embeddings from queries and opinions",
        routes=app.routes,
    )

    if "/api/v1/embed/text" in openapi_schema["paths"]:
        openapi_schema["paths"]["/api/v1/embed/text"]["post"][
            "requestBody"
        ] = {
            "content": {
                "text/plain": {
                    "example": "A very long opinion goes here.\nIt can span multiple lines.\nEach line will be preserved."
                }
            },
            "required": True,
        }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore
app.include_router(routes.api_router)  # type: ignore


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005)
