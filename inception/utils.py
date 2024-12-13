import logging
import re

from fastapi import HTTPException
from torch.cuda import OutOfMemoryError

from inception.config import settings
from inception.metrics import ERROR_COUNT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_text_for_json(text: str) -> str | None:
    """
    Clean and prepare text for JSON encoding.
    Handles special characters, line breaks, and other potential JSON issues.

    :param text: Text to clean.
    :return: Cleaned text or None if there was an error.
    """

    if not text:
        return ""

    try:
        # Remove null bytes and other control characters except newlines and tabs
        text = "".join(
            char
            for char in text
            if char == "\n" or char == "\t" or (32 <= ord(char) < 127)
        )

        # Replace tabs with spaces
        text = text.replace("\t", " ")

        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove spaces at the beginning and end of each line
        text = "\n".join(line.strip() for line in text.split("\n"))

        # Remove multiple consecutive empty lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    except Exception as e:
        raise ValueError(f"Error cleaning text: {str(e)}")


def preprocess_text(text: str) -> str:
    """
    Preprocess text for embedding generation.
    Includes cleaning and validation steps.
    """
    try:
        cleaned_text = clean_text_for_json(text)
        if not cleaned_text:
            raise ValueError("Text is empty after cleaning.")

        return cleaned_text
    except Exception as e:
        raise ValueError(f"Error preprocessing text: {str(e)}")


def validate_text_length(
    text: str, endpoint: str, doc_id: int | None = None
) -> None:
    """Validate text length against given constraints and log ERROR_COUNT.

    :param text: The text to validate.
    :param endpoint: The name of the endpoint.
    :param doc_id: Optional the document ID if the request belong to batch endpoint.
    :return: None it raises HTTPException if text length is below minimum or exceeds maximum.
    """
    text_length = len(text.strip())
    if text_length < settings.min_text_length:
        ERROR_COUNT.labels(
            endpoint=endpoint, error_type="text_too_short"
        ).inc()
        error_msg = f"Text length ({text_length}) below minimum ({settings.min_text_length})"
        if doc_id is not None:
            error_msg = f"Document {doc_id}: Text length ({text_length}) below minimum ({settings.min_text_length})."
        raise ValueError(error_msg)

    if text_length > settings.max_text_length:
        ERROR_COUNT.labels(endpoint=endpoint, error_type="text_too_long").inc()
        raise ValueError(
            f"Text length ({text_length}) exceeds maximum ({settings.max_text_length})"
        )


def handle_exception(e: Exception, endpoint: str) -> None:
    """Handle a general exception, increment error counters and re-raise as
    HTTPException if needed.

    :param e: The exception that occurred.
    :param endpoint: The endpoint name for logging errors.
    :raises HTTPException: Re-raised if needed with appropriate status code and message.
    """

    if isinstance(e, UnicodeDecodeError):
        ERROR_COUNT.labels(endpoint=endpoint, error_type="decode_error").inc()
        raise HTTPException(
            status_code=422, detail="Invalid UTF-8 encoding in text"
        )
    elif isinstance(e, ValueError):
        # Validation error
        ERROR_COUNT.labels(
            endpoint=endpoint, error_type="validation_error"
        ).inc()
        raise HTTPException(status_code=422, detail=str(e))
    elif isinstance(e, OutOfMemoryError):
        # GPU memory error
        ERROR_COUNT.labels(endpoint=endpoint, error_type="gpu_error").inc()
        raise HTTPException(status_code=503, detail="GPU memory exhausted")
    else:
        # Other processing errors
        ERROR_COUNT.labels(
            endpoint=endpoint, error_type="processing_error"
        ).inc()
        raise e
