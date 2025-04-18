"""
This module contains the logic for searching the database, with no dependencies on streamlit.
"""

import functools
import time
from collections import defaultdict

from litellm import batch_completion
from pydantic import BaseModel

from logs import logger
from constants import SYNC_PATTERN
from core_types import AssetType, Chunk, Document, DocumentStructure
from storage import Database
from strategies.embed_chunks import EmbedChunksStrategy


class SearchStats(BaseModel):
    num_embeddings: int
    num_documents: int


class ChunkSearchResult(BaseModel):
    chunk: Chunk
    score: float
    nice_extract: str | None
    path: list[str]

class DocSearchResult(BaseModel):
    document: Document
    chunks: list[ChunkSearchResult]

    @property
    def max_score(self):
        return max(chunk.score for chunk in self.chunks)


def check_in_cache(func):
    """
    Alternative to functools.cache, which uses self.cache as the cache, where self is the method's owner.
    """
    name = func.__name__

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        key = (name, *args, *kwargs.items())
        try:
            return self.cache[key]
        except KeyError:
            ret = func(self, *args, **kwargs)
            self.cache[key] = ret
            return ret

    return wrapper


class SearchEngine:
    def __init__(self, db: Database, cache: dict):
        self.db = db
        self.cache = cache
        # self.embeddings, self.chunk_to_idx = self.db.load_embeddings()
        # self.idx_to_chunk = {v: k for k, v in self.chunk_to_idx.items()}
        self.embeddings, self.chunk_to_idx, self.idx_to_chunk = self.get_embedding_and_mapping()

    @property
    def last_load_embeddings(self):
        return self.cache.get("last_load_embeddings", 0)

    @last_load_embeddings.setter
    def last_load_embeddings(self, value):
        self.cache["last_load_embeddings"] = value

    @check_in_cache
    def get_embedding_and_mapping(self):
        embeddings, chunk_to_idx = self.db.load_embeddings()
        idx_to_chunk = {v: k for k, v in chunk_to_idx.items()}
        self.last_load_embeddings = time.time()
        return embeddings, chunk_to_idx, idx_to_chunk

    def stats(self):
        return SearchStats(
            num_embeddings=len(self.embeddings),
            num_documents=self.db.count_documents(),
        )

    def search_chunks(self, query: str, nb_results: int = 10) -> list[ChunkSearchResult]:
        embedding = self.embed(query)
        self.reload_embeddings_if_changed()

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
                path=self.get_chunk_path(chunks[chunk_id]),
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
        return [chunk.content[:100] for chunk in chunks]
        text_cleaned = [SYNC_PATTERN.sub("", chunk.content) for chunk in chunks]
        return self._make_nice_extracts(query, *text_cleaned)

    @check_in_cache
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
- If some terms are especially relevant you can **bold** them using markdown.
- If there is no apparent relation between the chunk and the query, output the most informative sentence from the chunk.
- Always output a part of the text, never output comments on the tasks.
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

    def get_chunk_path(self, chunk: Chunk) -> list[str]:
        """Return a list of section titles that contain the chunk, if any."""
        structure_assets = self.db.get_assets_for_document(chunk.document_id, AssetType.STRUCTURE)
        syncted_text_assets = self.db.get_assets_for_document(chunk.document_id, AssetType.SYNCED_TEXT_FILE)
        if not structure_assets or not syncted_text_assets:
            return []

        structure = DocumentStructure.model_validate_json(structure_assets[0].path.read_text())
        chunk_start = syncted_text_assets[0].path.read_text().find(chunk.content)

        assert chunk_start != -1, "Chunk not found in synced text??"

        path = []
        while structure.subsections:
            for subsection in structure.subsections:
                if chunk_start >= subsection.start_idx and chunk_start < subsection.subsections_end_idx:
                    path.append(subsection.title)
                    structure = subsection
                    break

        return path

    @check_in_cache
    def embed(self, text: str):
        return EmbedChunksStrategy(None, self.db).embed_texts_sync([text])[0]

    def reload_embeddings_if_changed(self):
        paths = [self.db.embeddings_path, self.db.embeddings_json_path]
        if any(path.stat().st_mtime > self.last_load_embeddings for path in paths):
            self.cache.pop(self.get_embedding_and_mapping.__name__, None)
            self.embeddings, self.chunk_to_idx, self.idx_to_chunk = self.get_embedding_and_mapping()
            self.last_load_embeddings = time.time()
            logger.debug("Reloaded embeddings, as modified on disk.")
