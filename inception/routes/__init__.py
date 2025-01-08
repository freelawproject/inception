from fastapi import APIRouter

from inception.routes.embedding import router as embedding_router
from inception.routes.monitoring import router as monitoring_router

api_router = APIRouter()
api_router.include_router(monitoring_router, tags=["General"])
api_router.include_router(embedding_router, tags=["Embedding"])
