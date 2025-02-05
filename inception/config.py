from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    transformer_model_name: str = Field(
        "nomic-ai/modernbert-embed-base",
        description="Name of the transformer model to use",
    )
    max_tokens: int = Field(
        8192, ge=1, le=10000, description="Maximum tokens per chunk"
    )
    sentence_overlap: int = Field(
        10, ge=1, le=100, description="Number of sentence overlap between chunks"
    )
    min_text_length: int = 1
    max_query_length: int = 100
    max_text_length: int = 10_000_000
    max_batch_size: int = 100
    processing_batch_size: int = 8
    pool_timeout: int = (
        3600  # Timeout for multi-process pool operations (seconds)
    )
    force_cpu: bool = False
    enable_metrics: bool = True


settings = Settings()  # type: ignore[call-arg]
