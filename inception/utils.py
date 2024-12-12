import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_text_for_json(text: str) -> str:
    """
    Clean and prepare text for JSON encoding.
    Handles special characters, line breaks, and other potential JSON issues.
    """

    if not text:
        return ""

    try:
        # Remove null bytes and other control characters except newlines and tabs
        text = "".join(
            char
            for char in text
            if char == "\n"
            or char == "\t"
            or (ord(char) >= 32 and ord(char) < 127)
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
