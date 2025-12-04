# app/config.py

import os
from dataclasses import dataclass


def _env(name: str, default=None, required: bool = False):
    """
    Helper to read environment variables.

    - If `required=True` and the var is missing, we raise a RuntimeError
      so the service fails fast at startup.
    - We also check for both bare names and NL2SQL_ prefixed names
      for convenience.
    """
    # allow either NAME or NL2SQL_NAME
    value = os.getenv(name)
    if value is None:
        value = os.getenv(f"NL2SQL_{name}")

    if value is None:
        if required and default is None:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default

    return value


@dataclass
class Settings:
    """
    Simple config object for the nl2sql-service.

    Attributes are in ALL_CAPS so existing code like
    `settings.OPENAI_API_KEY` continues to work.
    """

    OPENAI_API_KEY: str
    OPENAI_MODEL: str
    OPENAI_EMBED_MODEL: str        # <--- NEW
    PINECONE_API_KEY: str
    PINECONE_INDEX: str
    PINECONE_CLOUD: str
    PINECONE_REGION: str
    PINECONE_NAMESPACE: str


def get_settings() -> Settings:
    """
    Factory used by other modules (vector_store, rag_sql, etc.).
    Reads from environment variables once and returns a Settings object.
    """
    return Settings(
        OPENAI_API_KEY=_env("OPENAI_API_KEY", required=True),
        OPENAI_MODEL=_env("OPENAI_MODEL", default="gpt-4.1-mini"),
        # Default embedding model; override via env if you want.
        OPENAI_EMBED_MODEL=_env(
            "OPENAI_EMBED_MODEL",
            default="text-embedding-3-small",
        ),
        PINECONE_API_KEY=_env("PINECONE_API_KEY", required=True),
        PINECONE_INDEX=_env("PINECONE_INDEX", default="nl2sql-schema-index"),
        PINECONE_CLOUD=_env("PINECONE_CLOUD", default="aws"),
        PINECONE_REGION=_env("PINECONE_REGION", default="us-east-1"),
        PINECONE_NAMESPACE=_env("PINECONE_NAMESPACE", default="default"),
    )


# Optional: module-level singleton for imports like `from app.config import settings`
settings = get_settings()
