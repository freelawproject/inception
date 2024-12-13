import time

from fastapi import APIRouter, HTTPException, Request

from inception import main
from inception.config import settings
from inception.embedding_service import EmbeddingService
from inception.metrics import ERROR_COUNT, PROCESSING_TIME, REQUEST_COUNT
from inception.schemas import (
    BatchTextRequest,
    QueryRequest,
    QueryResponse,
    TextRequest,
    TextResponse,
)
from inception.utils import (
    handle_exception,
    preprocess_text,
    validate_text_length,
)

router = APIRouter()


def check_embedding_service(
    embedding_service: EmbeddingService | None, endpoint: str
) -> EmbeddingService:
    """Check if the embedding service is initialized.

    :param embedding_service: The embedding service instance.
    :param endpoint: The name of the endpoint.
    :return: None it raises HTTPException if service is not initialized.
    """
    if embedding_service is None:
        ERROR_COUNT.labels(
            endpoint=endpoint, error_type="service_unavailable"
        ).inc()
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized",
        )
    return embedding_service


@router.post("/api/v1/embed/query", response_model=QueryResponse)
async def create_query_embedding(request: QueryRequest):
    """Generate embedding for a query"""

    REQUEST_COUNT.labels(endpoint="query").inc()
    start_time = time.time()
    embedding_service = check_embedding_service(
        main.embedding_service, "query"
    )
    try:
        validate_text_length(request.text, "query")
        embedding = await embedding_service.generate_query_embedding(
            request.text
        )
        PROCESSING_TIME.labels(endpoint="query").observe(
            time.time() - start_time
        )
        return QueryResponse(embedding=embedding)
    except Exception as e:
        handle_exception(e, "query")


@router.post("/api/v1/embed/text", response_model=TextResponse)
async def create_text_embedding(request: Request):
    """Generate embeddings for opinion text"""
    REQUEST_COUNT.labels(endpoint="text").inc()
    start_time = time.time()
    embedding_service = check_embedding_service(main.embedding_service, "text")
    try:
        raw_text = await request.body()
        text = raw_text.decode("utf-8")
        validate_text_length(text, "text")
        result = await embedding_service.generate_text_embeddings([text])

        # Clean up GPU memory after processing large texts
        text_length = len(text.strip())
        if (
            text_length > settings.max_words * 10
        ):  # Arbitrary threshold for "large" texts
            embedding_service.cleanup_gpu_memory()

        PROCESSING_TIME.labels(endpoint="text").observe(
            time.time() - start_time
        )
        return TextResponse(embeddings=result[0])
    except Exception as e:
        handle_exception(e, "text")


@router.post("/api/v1/embed/batch", response_model=list[TextResponse])
async def create_batch_text_embeddings(request: BatchTextRequest):
    """Generate embeddings for multiple documents"""
    REQUEST_COUNT.labels(endpoint="batch").inc()
    start_time = time.time()
    embedding_service = check_embedding_service(
        main.embedding_service, "batch"
    )
    if len(request.documents) > settings.max_batch_size:
        ERROR_COUNT.labels(
            endpoint="batch", error_type="batch_too_large"
        ).inc()
        raise HTTPException(
            status_code=422,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size} documents",
        )

    try:
        # Validate all texts before processing
        for doc in request.documents:
            validate_text_length(doc.text, "batch", doc.id)

        texts = [doc.text for doc in request.documents]
        embeddings_list = await embedding_service.generate_text_embeddings(
            texts
        )
        results = [
            TextResponse(id=doc.id, embeddings=embeddings)
            for doc, embeddings in zip(request.documents, embeddings_list)
        ]
        # Clean up GPU memory after batch processing
        embedding_service.cleanup_gpu_memory()
        PROCESSING_TIME.labels(endpoint="batch").observe(
            time.time() - start_time
        )
        return results
    except Exception as e:
        handle_exception(e, "batch")


# this is a temporary validation endpoint to test text preprocessing
@router.post("/api/v1/validate/text")
async def validate_text(request: TextRequest):
    """
    Validate and clean text without generating embeddings.
    Useful for testing text preprocessing.
    """
    try:
        processed_text = preprocess_text(request.text)
        return {
            "id": request.id,
            "original_text": request.text,
            "processed_text": processed_text,
            "is_valid": True,
        }
    except Exception as e:
        return {
            "id": request.id,
            "original_text": request.text,
            "error": str(e),
            "is_valid": False,
        }
