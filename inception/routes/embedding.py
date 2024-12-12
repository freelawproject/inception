import time

import torch
from fastapi import APIRouter, HTTPException, Request
from sentry_sdk import capture_exception

from inception import main
from inception.config import settings
from inception.metrics import ERROR_COUNT, PROCESSING_TIME, REQUEST_COUNT
from inception.schemas import (BatchTextRequest, QueryRequest, QueryResponse,
                               TextRequest, TextResponse)
from inception.utils import preprocess_text

router = APIRouter()


@router.post("/api/v1/embed/query", response_model=QueryResponse)
async def create_query_embedding(request: QueryRequest):
    """Generate embedding for a query"""

    REQUEST_COUNT.labels(endpoint="query").inc()
    start_time = time.time()
    if not main.embedding_service:
        ERROR_COUNT.labels(endpoint="query", error_type="service_unavailable").inc()
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        embedding = await main.embedding_service.generate_query_embedding(request.text)
        PROCESSING_TIME.labels(endpoint="query").observe(time.time() - start_time)
        return QueryResponse(embedding=embedding)
    except ValueError as e:
        ERROR_COUNT.labels(endpoint="query", error_type="validation_error").inc()
        raise HTTPException(status_code=422, detail=str(e))
    except torch.cuda.OutOfMemoryError as e:
        ERROR_COUNT.labels(endpoint="query", error_type="gpu_error").inc()
        raise HTTPException(status_code=503, detail="GPU memory exhausted")
    except Exception as e:
        ERROR_COUNT.labels(endpoint="query", error_type="processing_error").inc()
        capture_exception(e)
        raise e


@router.post("/api/v1/embed/text", response_model=TextResponse)
async def create_text_embedding(request: Request):
    """Generate embeddings for opinion text"""
    REQUEST_COUNT.labels(endpoint="text").inc()
    start_time = time.time()

    if not main.embedding_service:
        ERROR_COUNT.labels(endpoint="text", error_type="service_unavailable").inc()
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    try:
        raw_text = await request.body()
        try:
            text = raw_text.decode("utf-8")
        except UnicodeDecodeError:
            ERROR_COUNT.labels(endpoint="text", error_type="decode_error").inc()
            raise HTTPException(
                status_code=422, detail="Invalid UTF-8 encoding in text"
            )

        text_length = len(text.strip())

        if text_length < settings.min_text_length:
            ERROR_COUNT.labels(endpoint="text", error_type="text_too_short").inc()
            raise HTTPException(
                status_code=422,
                detail=f"Text length ({text_length}) below minimum ({settings.min_text_length})",
            )

        if text_length > settings.max_text_length:
            ERROR_COUNT.labels(endpoint="text", error_type="text_too_long").inc()
            raise HTTPException(
                status_code=422,
                detail=f"Text length ({text_length}) exceeds maximum ({settings.max_text_length})",
            )

        result = await main.embedding_service.generate_text_embeddings([text])

        # Clean up GPU memory after processing large texts
        if (
            text_length > settings.max_words * 10
        ):  # Arbitrary threshold for "large" texts
            main.embedding_service.cleanup_gpu_memory()

        PROCESSING_TIME.labels(endpoint="text").observe(time.time() - start_time)
        return TextResponse(embeddings=result[0])
    except Exception as e:
        ERROR_COUNT.labels(endpoint="text", error_type="processing_error").inc()
        capture_exception(e)
        raise e


@router.post("/api/v1/embed/batch", response_model=list[TextResponse])
async def create_batch_text_embeddings(request: BatchTextRequest):
    """Generate embeddings for multiple documents"""
    REQUEST_COUNT.labels(endpoint="batch").inc()
    start_time = time.time()

    if not main.embedding_service:
        ERROR_COUNT.labels(endpoint="batch", error_type="service_unavailable").inc()
        raise HTTPException(status_code=503, detail="Embedding service not initialized")

    if len(request.documents) > settings.max_batch_size:
        ERROR_COUNT.labels(endpoint="batch", error_type="batch_too_large").inc()
        raise HTTPException(
            status_code=422,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size} documents",
        )

    try:
        # Validate all texts before processing
        for doc in request.documents:
            text_length = len(doc.text)
            if text_length < settings.min_text_length:
                raise ValueError(
                    f"Document {doc.id}: Text length ({text_length}) below minimum ({settings.min_text_length})."
                )

        texts = [doc.text for doc in request.documents]
        embeddings_list = await main.embedding_service.generate_text_embeddings(texts)

        results = [
            TextResponse(id=doc.id, embeddings=embeddings)
            for doc, embeddings in zip(request.documents, embeddings_list)
        ]

        # Clean up GPU memory after batch processing
        main.embedding_service.cleanup_gpu_memory()

        PROCESSING_TIME.labels(endpoint="batch").observe(time.time() - start_time)
        return results
    except Exception as e:
        ERROR_COUNT.labels(endpoint="batch", error_type="processing_error").inc()
        capture_exception(e)
        raise HTTPException(status_code=422, detail=f"{str(e)}")


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
