import asyncio
import time
from itertools import islice

import torch

torch.set_float32_matmul_precision("high")

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
        overlap_ratio: float,
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
            if device == "cuda":
                logger.info(f"CUDA device: {torch.cuda.current_device()}")
            self.gpu_model = model.to(device)
            self.max_tokens = max_tokens
            self.num_overlap_sentences = int(max_tokens * overlap_ratio)
            self.processing_batch_size = processing_batch_size
            MODEL_LOAD_TIME.observe(time.time() - start_time)
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {str(e)}")
            raise

    def split_text_into_chunks(self, text: str) -> list[str]:
        """Split text into chunks based on sentences, not exceeding max_tokens, with sentence overlap"""

        # Split the text to sentences & encode sentences with tokenizer
        sentences = sent_tokenize(text)
        encoded_sentences = [
            self.tokenizer.encode(sentence, add_special_tokens=False)
            for sentence in sentences
        ]
        lead_text = "search_document: "
        lead_tokens = self.tokenizer.encode(lead_text)
        lead_len = len(lead_tokens)
        chunks = []
        current_chunks: list[str] = []
        current_token_counts = len(lead_tokens)

        for sentence_tokens in encoded_sentences:
            sentence_len = len(sentence_tokens)
            # if the current sentence itself is above max_tokens
            if lead_len + sentence_len > self.max_tokens:
                # store the previous chunk
                if current_chunks:
                    chunks.append(lead_text + " ".join(current_chunks))
                # truncate the sentence and store the truncated sentence as its own chunk
                truncated_sentence = self.tokenizer.decode(
                    sentence_tokens[: (self.max_tokens - len(lead_tokens))]
                )
                chunks.append(lead_text + truncated_sentence)

                # start a new chunk with no overlap (because adding the current sentence will exceed the max_tokens)
                current_chunks = []
                current_token_counts = lead_len
                continue

            # if adding the new sentence will cause the chunk to exceed max_tokens
            if current_token_counts + sentence_len > self.max_tokens:
                overlap_sentences = current_chunks[
                    -max(0, self.num_overlap_sentences) :
                ]
                # store the previous chunk
                if current_chunks:
                    chunks.append(lead_text + " ".join(current_chunks))

                overlap_token_counts = self.tokenizer.encode(
                    " ".join(overlap_sentences), add_special_tokens=False
                )
                # If the sentence with the overlap exceeds the limit, start a new chunk without overlap.
                if (
                    lead_len + len(overlap_token_counts) + sentence_len
                    > self.max_tokens
                ):
                    current_chunks = [self.tokenizer.decode(sentence_tokens)]
                    current_token_counts = lead_len + sentence_len
                else:
                    current_chunks = overlap_sentences + [
                        self.tokenizer.decode(sentence_tokens)
                    ]
                    current_token_counts = (
                        lead_len + len(overlap_token_counts) + sentence_len
                    )
                continue

            # if within max_tokens, continue to add the new sentence to the current chunk
            current_chunks.append(self.tokenizer.decode(sentence_tokens))
            current_token_counts += len(sentence_tokens)

        # store the last chunk if it has any content
        if current_chunks:
            chunks.append(lead_text + " ".join(current_chunks))
        return chunks

    async def generate_query_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single query text"""

        # Preprocess the query text
        processed_text = preprocess_text(text)

        # Use run_in_executor to make the synchronous encode call async
        embedding = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(
                sentences=[f"search_query: {processed_text}"], batch_size=1
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

        logger.info(f"Generating embedding for {len(texts)} documents of {sum(len(s) for s in texts)} charsacters")

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

        # Clean up the chunks
        clean_chunks = [
            chunk.replace("search_document: ", "") for chunk in all_chunks
        ]

        # Create pairs of embeddings and corresponding chunks
        embedding_chunk_pairs = zip(embeddings, clean_chunks)
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
