from typing import List

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from .schema_docs import load_schema_chunks, SchemaChunk
from .config import get_settings


class PineconeSchemaVectorStore:
    """
    Stores schema chunks in Pinecone and supports similarity search.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

        # Pinecone client
        self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)

        # Create index if needed
        existing = self.pc.list_indexes().names()
        if self.settings.PINECONE_INDEX not in existing:
            self.pc.create_index(
                name=self.settings.PINECONE_INDEX,
                dimension=1536,  # text-embedding-3-small
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",
                ),
            )

        self.index = self.pc.Index(self.settings.PINECONE_INDEX)

        # OpenAI client
        self.client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

        # Load schema chunks from maindata.sql (via schema_docs.py)
        self.chunks: List[SchemaChunk] = load_schema_chunks()

        # Ensure Pinecone has schema vectors
        self._ensure_index_populated()

    # ----------------- internal helpers -----------------

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using OpenAI embeddings.
        NOTE: caller is responsible for not sending too many tokens at once.
        """
        resp = self.client.embeddings.create(
            model=self.settings.OPENAI_EMBED_MODEL,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    def _ensure_index_populated(self) -> None:
        """
        Upload schema chunks to Pinecone if index is empty.
        We do this in SMALL BATCHES to avoid hitting the
        OpenAI 'max tokens per request' limit.
        """
        stats = self.index.describe_index_stats()
        if stats.get("total_vector_count", 0) > 0:
            print("[Pinecone] Index already populated.")
            return

        print("[Pinecone] Uploading schema vectors in batches...")

        # small batch size = fewer tokens per request
        BATCH_SIZE = 10

        total = len(self.chunks)
        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)
            batch_chunks = self.chunks[start:end]
            batch_texts = [chunk.text for chunk in batch_chunks]

            # Embed this small batch
            embeddings = self._embed(batch_texts)

            # Prepare vectors for this batch
            vectors = []
            for chunk, emb in zip(batch_chunks, embeddings):
                vectors.append(
                    {
                        "id": chunk.id,
                        "values": emb,
                        "metadata": {
                            "text": chunk.text,
                        },
                    }
                )

            # Upsert this batch
            self.index.upsert(vectors=vectors)
            print(f"[Pinecone] Uploaded tables {start + 1}–{end} of {total}")

        print("[Pinecone] Upload complete.")

    # ----------------- public API -----------------

    def search(self, query: str, top_k: int = 5) -> List[SchemaChunk]:
        """
        Semantic search over schema, returns SchemaChunks.
        """
        # Single query => safe to embed directly
        query_vec = self._embed([query])[0]

        res = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
        )

        chunks: List[SchemaChunk] = []
        for match in res.matches:
            chunks.append(
                SchemaChunk(
                    id=match.id,
                    text=match.metadata["text"],
                )
            )

        return chunks
