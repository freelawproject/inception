import torch
from fastapi import APIRouter, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from inception import main
from inception.config import settings

router = APIRouter()


@router.get("/")
async def heartbeat():
    """Simple heartbeat endpoint"""
    return "Heartbeat detected."


@router.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy" if main.embedding_service else "service_unavailable",
        "model_loaded": main.embedding_service is not None,
        "gpu_available": gpu_available and not settings.force_cpu,
    }


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not settings.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics not enabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
