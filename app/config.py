import os
from functools import lru_cache


# app/config.py

from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    Central config object for the nl2sql-service.

    All values are read from environment variables.
    On Render, set them in the dashboard.
    """

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4.1-mini"  # or whatever you picked

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "nl2sql-schema-index"
    PINECONE_CLOUD: str = "aws"         # used by the new Pinecone client
    PINECONE_REGION: str = "us-east-1"  # adjust if your index is elsewhere
    PINECONE_NAMESPACE: str = "default"

    class Config:
        # optional: if you want local .env support
        env_file = ".env"
        # prefix env vars like NL2SQL_OPENAI_API_KEY, NL2SQL_PINECONE_API_KEY, etc.
        env_prefix = "NL2SQL_"
        extra = "allow"  # ignore any extra env vars


# This is what rag_sql.py and vector_store.py import.
settings = Settings()
