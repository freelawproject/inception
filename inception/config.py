from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    transformer_model_name: str = Field(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        description="Name of the transformer model to use",
    )
    max_words: int = Field(
        350, ge=1, le=1000, description="Maximum words per chunk"
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

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()  # type: ignore[call-arg]
