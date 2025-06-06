"""
This module contains the logic for searching the database, with no dependencies on streamlit.
"""

import asyncio
from collections import defaultdict
from functools import lru_cache

from litellm import batch_completion
from pydantic import BaseModel

from constants import SYNC_PATTERN
from core_types import Chunk, Document
from storage import Database
from strategies.embed_chunks import EmbedChunksStrategy


class SearchStats(BaseModel):
    num_embeddings: int
    num_documents: int


class ChunkSearchResult(BaseModel):
    chunk: Chunk
    score: float
    nice_extract: str | None


class DocSearchResult(BaseModel):
    document: Document
    chunks: list[ChunkSearchResult]

    @property
    def max_score(self):
        return max(chunk.score for chunk in self.chunks)


class SearchEngine:
    def __init__(self, db: Database):
        self.db = db
        self.embeddings, self.chunk_to_idx = db.load_embeddings()
        self.idx_to_chunk = {v: k for k, v in self.chunk_to_idx.items()}

    def stats(self):
        return SearchStats(
            num_embeddings=len(self.embeddings),
            num_documents=self.db.count_documents(),
        )

    def search_chunks(self, query: str, nb_results: int = 10) -> list[ChunkSearchResult]:
        embedding = self.embed(query)

        distances = self.embeddings @ embedding
        top_n = distances.argsort()[-nb_results:][::-1]
        top_chunks_ids = [self.idx_to_chunk[i] for i in top_n]
        top_chunks = self.db.get_chunks(top_chunks_ids)
        chunks = {chunk.id: chunk for chunk in top_chunks}

        nice_extracts = self.make_nice_extracts(top_chunks, query)

        return [
            ChunkSearchResult(
                chunk=chunks[chunk_id].model_dump(),  # dump to avoid hot relaod problems
                score=distances[self.chunk_to_idx[chunk_id]],
                nice_extract=nice_extract,
            )
            for chunk_id, nice_extract in zip(top_chunks_ids, nice_extracts, strict=True)
        ]

    def search(self, query: str, nb_results: int = 10) -> list[DocSearchResult]:
        chunks = self.search_chunks(query, nb_results)

        results_by_doc = defaultdict(list)
        for chunk in chunks:
            results_by_doc[chunk.chunk.document_id].append(chunk)

        doc_results = [
            DocSearchResult(
                document=self.db.get_document(
                    doc_id
                ).model_dump(),  # dump to avoid hot relaod problems
                chunks=chunks,
            )
            for doc_id, chunks in results_by_doc.items()
        ]

        doc_results.sort(key=lambda x: x.max_score, reverse=True)

        return doc_results

    def make_nice_extracts(self, chunks: list[Chunk], query: str) -> list[str]:
        text_cleaned = [SYNC_PATTERN.sub("", chunk.content) for chunk in chunks]
        return self._make_nice_extracts(query, *text_cleaned)

    @lru_cache(maxsize=10_000)
    def _make_nice_extracts(self, query: str, *texts: str) -> list[str]:
        PROMPT = """
You are part of a search engine.  You see chunks of texts, which are the result of a search, and a user query.
Your goal is to print the sentence that are the most informative of the content of the chunk, with respect to the user query.
The user will not see the chunk, but only the sentences you output, which need to be as informative as possible.

Your response should:
- Contain only the most informative excerpt from the text
- Be the most related to the above query
- Include 1 or 2 full sentences
- Include no additional text, introductions, or explanations
- Preserve the exact wording from the original text
- If there is no apparent relation between the chunk and the query, output the most informative sentence from the chunk
- If some terms are especially relevant you can **bold** them using markdown.
"""

        responses = batch_completion(
            model="groq/llama-3.3-70b-versatile",
            messages=[
                [
                    dict(role="system", content=PROMPT),
                    dict(role="user", content=f"USER QUERY: {query}\nCHUNK: {text}\n"),
                ]
                for text in texts
            ],
        )

        nice_extracts = [response.choices[0].message.content for response in responses]

        return nice_extracts

    @lru_cache(maxsize=10_000)
    def embed(self, text: str):
        embedding = asyncio.run(EmbedChunksStrategy(None, self.db).embed_texts([text]))[0]
        return embedding
