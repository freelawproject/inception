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
            self.gpu_model = model.to(device)
            self.max_tokens = max_tokens
            self.num_overlap_sentences = int(max_tokens * overlap_ratio)
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

        # Split the text to sentences & encode sentences with tokenizer
        sentences = sent_tokenize(text)
        sentence_dict = {}
        for i, sentence in enumerate(sentences):
            sentence_tokens = self.tokenizer.encode(
                sentence, add_special_tokens=False
            )
            sentence_dict[i] = sentence_tokens

        lead_text = "search_document: "
        lead_tokens = self.tokenizer.encode(lead_text)
        chunks = []
        current_chunks: list[str] = []
        current_token_counts = len(lead_tokens)

        for sentence_index, sentence_tokens in sentence_dict.items():

            # if the current sentence itself is above max_tokens
            if len(lead_tokens) + len(sentence_tokens) > self.max_tokens:
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
                current_token_counts = len(lead_tokens)

            # if adding the new sentence will cause the chunk to exceed max_tokens
            elif current_token_counts + len(sentence_tokens) > self.max_tokens:
                # store the previous chunk
                if current_chunks:
                    chunks.append(lead_text + " ".join(current_chunks))

                # find out the sentences to overlap
                overlap_sentences = current_chunks[
                    -max(0, self.num_overlap_sentences) :
                ]
                overlap_token_counts = self.tokenizer.encode(
                    " ".join(overlap_sentences), add_special_tokens=False
                )

                # if adding the overlap will cause the sentence to exceed the token limits,
                # start a new chunk with only the current sentence and no overlap
                if (
                    len(lead_tokens)
                    + len(overlap_token_counts)
                    + len(sentence_tokens)
                    > self.max_tokens
                ):
                    current_chunks = [self.tokenizer.decode(sentence_tokens)]
                    current_token_counts = len(lead_tokens) + len(
                        sentence_tokens
                    )

                # overwise, add the overlap to the new chunk in addition to the current sentence
                else:
                    current_chunks = overlap_sentences + [
                        self.tokenizer.decode(sentence_tokens)
                    ]
                    current_token_counts = (
                        len(lead_tokens)
                        + len(overlap_token_counts)
                        + len(sentence_tokens)
                    )

            # if within max_tokens, continue to add the new sentence to the current chunk
            else:
                current_chunks.append(self.tokenizer.decode(sentence_tokens))
                current_token_counts += len(sentence_tokens)

            # store the last chunk if it has any content
            if (
                sentence_index == len(sentence_dict.keys()) - 1
                and current_chunks
            ):
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
