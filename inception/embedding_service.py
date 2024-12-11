import time
import asyncio
import torch
import logging
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from inception.schemas import ChunkEmbedding
from inception.config import settings
from inception.utils import preprocess_text
from inception.metrics import MODEL_LOAD_TIME, CHUNK_COUNT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: SentenceTransformer, max_words: int):
        start_time = time.time()
        try:
            self.model = model
            device = "cpu" if settings.force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
            self.gpu_model = model.to(device)
            self.max_words = max_words
            if device == "cuda":
                self.pool = self.gpu_model.start_multi_process_pool()
            MODEL_LOAD_TIME.observe(time.time() - start_time)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise

    def __del__(self):
        try:
            if hasattr(self, "pool"):
                self.gpu_model.stop_multi_process_pool(self.pool)
                logger.info("Model pool stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping model pool: {str(e)}")

    def split_text_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks based on sentences, not exceeding max_words"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)

            if current_word_count + sentence_word_count <= self.max_words:
                current_chunk.append(sentence)
                current_word_count += sentence_word_count
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def generate_query_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single query text"""

        # Preprocess the query text
        processed_text = preprocess_text(text)

        print("processed_text", processed_text)

        # Use run_in_executor to make the synchronous encode call async
        embedding = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(
                sentences=[processed_text],
                batch_size=1
            )
        )
        print("<<<< embedding", embedding)
        return embedding[0].tolist()


    async def generate_text_embeddings(self, texts: list[str]) -> list[list[ChunkEmbedding]]:
        """Generate embeddings for a list of texts"""
        if not texts:
            raise ValueError("Empty text list")

        all_embeddings = []
        all_chunks = []
        chunk_counts = []

        # Collect chunks
        for text in texts:
            chunks = self.split_text_into_chunks(text)
            CHUNK_COUNT.labels(endpoint="text").inc(len(chunks))
            all_chunks.extend(chunks)
            chunk_counts.append(len(chunks))

        # Generate embeddings
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(
                sentences=all_chunks,
                batch_size=8
            )
        )

        # Process results
        start_index = 0
        for count in chunk_counts:
            text_embeddings = []
            for j in range(count):
                chunk = all_chunks[start_index + j]
                embedding = embeddings[start_index + j]
                text_embeddings.append(ChunkEmbedding(
                    chunk_number=j + 1,
                    chunk=chunk,
                    embedding=embedding.tolist()
                ))
            all_embeddings.append(text_embeddings)
            start_index += count

        return all_embeddings

    def cleanup_gpu_memory(self):
        """Clean up GPU memory if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()