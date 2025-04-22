import asyncio
import time

import torch

torch.set_float32_matmul_precision("high")

from concurrent.futures import ThreadPoolExecutor, as_completed

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from inception.config import settings
from inception.metrics import CHUNK_COUNT, MODEL_LOAD_TIME
from inception.schemas import ChunkEmbedding, TextResponse
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
        self, texts: dict[int, str]
    ) -> list[TextResponse]:
        """Generate embeddings for a dict of texts"""
        if not texts:
            raise ValueError("Empty text dict")

        logger.info(f"Generating embedding for {len(texts)} documents")

        start_time = time.time()

        # Chunk the texts
        all_chunks = []
        chunk_counts = []
        chunks_by_id = {}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    lambda doc_id_text: (
                        doc_id_text[0],
                        self.split_text_into_chunks(doc_id_text[1]),
                    ),
                    item,
                ): item[0]
                for item in texts.items()
            }

            for future in as_completed(futures):
                doc_id, chunks = future.result()
                CHUNK_COUNT.labels(endpoint="text").inc(len(chunks))
                all_chunks.extend(chunks)
                chunk_counts.append(len(chunks))
                chunks_by_id[doc_id] = chunks

        chunk_time = time.time()
        logger.info(
            f"Producing {len(all_chunks)} chunks took {chunk_time - start_time:.2f} seconds"
        )

        # Generate embeddings
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.gpu_model.encode(
                sentences=all_chunks, batch_size=self.processing_batch_size
            ),
        )

        embed_time = time.time()
        logger.info(
            f"Generating embedding took {embed_time - chunk_time:.2f} seconds"
        )

        # Clean up the chunks
        clean_chunks = [
            chunk.replace("search_document: ", "") for chunk in all_chunks
        ]

        # Clean up the embeddings
        results = []
        embedding_idx = 0

        for doc_id, chunk_count in zip(chunks_by_id.keys(), chunk_counts):
            # Slice the embeddings for the current document's chunks
            document_embeddings = embeddings[
                embedding_idx : embedding_idx + chunk_count
            ]
            document_chunks = clean_chunks[
                embedding_idx : embedding_idx + chunk_count
            ]

            # Store embeddings for the current document
            doc_results = [
                ChunkEmbedding(
                    chunk_number=idx + 1,
                    chunk=chunk,
                    embedding=embedding.tolist(),
                )
                for idx, (embedding, chunk) in enumerate(
                    zip(document_embeddings, document_chunks)
                )
            ]
            results.append(TextResponse(id=doc_id, embeddings=doc_results))

            # Increment index for the next document
            embedding_idx += chunk_count

        end_time = time.time()
        logger.info(f"Wrap-up took {end_time - embed_time:.2f} seconds")

        return results

    @staticmethod
    def cleanup_gpu_memory():
        """Clean up GPU memory if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
