import asyncio
import time
from itertools import islice

import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from inception.config import settings
from inception.metrics import CHUNK_COUNT, MODEL_LOAD_TIME
from inception.schemas import ChunkEmbedding
from inception.utils import logger, preprocess_text


class EmbeddingService:
    def __init__(
        self,
        model: SentenceTransformer,
        tokenizer: AutoTokenizer,
        max_tokens: int,
        sentence_overlap: int,
        processing_batch_size: int,
    ):
        start_time = time.time()
        try:
            self.model = model
            self.tokenizer = tokenizer
            device = (
                "cpu"
                if settings.force_cpu
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.gpu_model = model.to(device)
            self.max_tokens = max_tokens
            self.sentence_overlap = sentence_overlap
            self.processing_batch_size = processing_batch_size
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
        """Split text into chunks based on sentences, not exceeding max_tokens, with sentence overlap"""
        # Set-up input format for the model
        lead_text = "search_document:"
        lead_tokens = len(self.tokenizer(lead_text)["input_ids"])
        
        sentences = sent_tokenize(text)  # Tokenize into sentences
        chunks = []
        start_idx = 0  # Start index of the current chunk
        
        while start_idx < len(sentences):
            current_chunk = []
            current_token_count = lead_tokens
            idx = start_idx
            
            # Build a chunk until max_tokens is reached
            while idx < len(sentences):
                token_count = len(self.tokenizer(sentences[idx], add_special_tokens=False)["input_ids"])
                if current_token_count + token_count >= self.max_tokens:
                    break
                current_chunk.append(sentences[idx])
                current_token_count += token_count
                idx += 1  # Move to next sentence
            
            if current_chunk:
                chunks.append(" ".join([lead_text] + current_chunk))
            
            # Stop if the last chunk reaches the end of the text
            if idx >= len(sentences):
                break
            
            # Move start index forward but keep overlap
            start_idx = max(idx - self.sentence_overlap, start_idx + 1)
        
        return chunks

    async def generate_query_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single query text"""

        # Preprocess the query text
        processed_text = preprocess_text(text)

        # Use run_in_executor to make the synchronous encode call async
        embedding = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(
                sentences=['search_query: ' + processed_text], batch_size=1
            ),
        )
        return embedding[0].tolist()

    async def generate_text_embeddings(
        self, texts: list[str]
    ) -> list[list[ChunkEmbedding]]:
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
                sentences=all_chunks, batch_size=self.processing_batch_size
            ),
        )

        # Create pairs of embeddings and corresponding chunks
        embedding_chunk_pairs = zip(embeddings, all_chunks)
        # Split the pairs into groups based on the number of chunks per text
        sliced_results = [
            list(islice(embedding_chunk_pairs, 0, i)) for i in chunk_counts
        ]
        logger.info(f"sliced_results {sliced_results}")
        for text_embedding in sliced_results:
            text_embeddings_list = []
            for idx, embedding_chunk_pair in enumerate(text_embedding):
                embedding, chunk = embedding_chunk_pair
                text_embeddings_list.append(
                    ChunkEmbedding(
                        chunk_number=idx + 1,
                        chunk=chunk,
                        embedding=embedding.tolist(),
                    )
                )
            all_embeddings.append(text_embeddings_list)

        return all_embeddings

    @staticmethod
    def cleanup_gpu_memory():
        """Clean up GPU memory if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
