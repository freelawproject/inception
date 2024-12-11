from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    id: int
    text: str = Field(..., description="The text content of the opinion")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "text": "The Supreme Court's decision in Brown v. Board of Education was a landmark ruling."
            }
        }


class BatchTextRequest(BaseModel):
    documents: list[TextRequest] = Field(
        ...,
        description="List of documents to process. Each document should have an ID and text content."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": 1,
                        "text": """The First Amendment protects freedom of speech and religion.

This fundamental right is crucial to democracy."""
                    },
                    {
                        "id": 2,
                        "text": """Marbury v. Madison (1803) established judicial review.

This case expanded judicial power significantly."""
                    }
                ]
            }
        }

class ChunkEmbedding(BaseModel):
    chunk_number: int
    chunk: str
    embedding: list[float]

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_number": 1,
                "chunk": "This is a sample chunk of text from a legal opinion.",
                "embedding": [0.123, 0.456, 0.789]
            }
        }


class TextResponse(BaseModel):
    id: int | None = None
    embeddings: list[ChunkEmbedding]

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "embeddings": [
                    {
                        "chunk_number": 1,
                        "chunk": "First chunk of the legal opinion text.",
                        "embedding": [0.123, 0.456, 0.789]
                    },
                    {
                        "chunk_number": 2,
                        "chunk": "Second chunk of the legal opinion text.",
                        "embedding": [0.321, 0.654, 0.987]
                    }
                ]
            }
        }


class QueryRequest(BaseModel):
    text: str


class QueryResponse(BaseModel):
    embedding: list[float]
