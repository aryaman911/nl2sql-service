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

    def _embed_single(self, text: str) -> List[float]:
        """
        Embed a SINGLE text using OpenAI embeddings.
        This guarantees we stay within context limits.
        """
        resp = self.client.embeddings.create(
            model=self.settings.OPENAI_EMBED_MODEL,
            input=[text],
        )
        return resp.data[0].embedding

    def _ensure_index_populated(self) -> None:
        """
        Upload schema chunks to Pinecone if index is empty.
        We embed ONE TABLE at a time to avoid hitting any token limits.
        """
        stats = self.index.describe_index_stats()
        if stats.get("total_vector_count", 0) > 0:
            print("[Pinecone] Index already populated.")
            return

        print(f"[Pinecone] Uploading schema vectors for {len(self.chunks)} tables...")

        total = len(self.chunks)
        for idx, chunk in enumerate(self.chunks, start=1):
            # Embed this single table's text
            emb = self._embed_single(chunk.text)

            vector = {
                "id": chunk.id,
                "values": emb,
                "metadata": {
                    "text": chunk.text,
                },
            }

            self.index.upsert(vectors=[vector])

            if idx % 10 == 0 or idx == total:
                print(f"[Pinecone] Uploaded {idx}/{total} tables")

        print("[Pinecone] Upload complete.")

    # ----------------- public API -----------------

    def search(self, query: str, top_k: int = 5) -> List[SchemaChunk]:
        """
        Semantic search over schema, returns SchemaChunks.
        """
        # Single query => embed once
        emb = self._embed_single(query)

        res = self.index.query(
            vector=emb,
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
