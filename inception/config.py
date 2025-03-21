from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    transformer_model_name: str = Field(
        "freelawproject/modernbert-embed-base_finetune_512",
        description="Name of the transformer model to use",
    )
    transformer_model_version: str = Field(
        "main",
        description="Version of the transformer model to use",
    )
    max_tokens: int = Field(
        512, ge=256, le=10000, description="Maximum tokens per chunk"
    )
    overlap_ratio: float = Field(
        0.004,
        ge=0,
        le=0.01,
        description="Ratio to calculate number of sentence overlap between chunks",
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
