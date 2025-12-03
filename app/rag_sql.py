from typing import List

from openai import OpenAI

from .config import get_settings
from .vector_store import PineconeSchemaVectorStore
from .schema_docs import SchemaChunk

settings = get_settings()


class NL2SQLPipeline:
    """
    Orchestrates: question -> Pinecone search -> prompt -> LLM -> SQL.
    """

    def __init__(self, vector_store: PineconeSchemaVectorStore) -> None:
        self.vector_store = vector_store
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _build_system_prompt(self) -> str:
        return (
            "You are an expert SQL generator for a known relational database.\n"
            "You must output ONLY a single valid SQL query and nothing else.\n"
            "Do NOT include backticks, comments, or explanation.\n"
            "Use only tables and columns that appear in the provided schema context.\n"
            "Prefer SELECT queries. Assume database is read-only.\n"
            "If the user's question cannot be answered with the schema, "
            "return: SELECT 'unable to answer with current schema' AS message;"
        )

    def _build_context(self, chunks: List[SchemaChunk]) -> str:
        return "\n\n".join(c.text for c in chunks)

    def _build_user_prompt(self, question: str, context: str) -> str:
        return (
            f"SCHEMA CONTEXT:\n{context}\n\n"
            "USER QUESTION:\n"
            f"{question}\n\n"
            "TASK:\n"
            "Write a single SQL query that correctly answers the user question, "
            "using only the tables and columns from the schema context above. "
            "Output only the SQL, no explanation."
        )

    def question_to_sql(self, question: str) -> str:
        # 1) Retrieve relevant schema chunks from Pinecone
        retrieved_chunks = self.vector_store.search(question, top_k=5)
        context = self._build_context(retrieved_chunks)

        system_msg = self._build_system_prompt()
        user_msg = self._build_user_prompt(question, context)

        # 2) Call LLM
        resp = self.client.chat.completions.create(
            model=settings.OPENAI_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
        )

        sql = resp.choices[0].message.content or ""
        return sql.strip()
