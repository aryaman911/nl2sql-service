import os
from functools import lru_cache


class Settings:
    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_CHAT_MODEL: str = "gpt-4.1-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "nl2sql-schema"

    # Optional DB sanity-check URL (not used yet)
    DB_URL: str | None = None

    def __init__(self) -> None:
        # Required
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
        if not self.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
        if not self.PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is not set")

        # Optional overrides
        self.OPENAI_CHAT_MODEL = os.environ.get(
            "OPENAI_CHAT_MODEL", self.OPENAI_CHAT_MODEL
        )
        self.OPENAI_EMBED_MODEL = os.environ.get(
            "OPENAI_EMBED_MODEL", self.OPENAI_EMBED_MODEL
        )
        self.PINECONE_INDEX = os.environ.get(
            "PINECONE_INDEX", self.PINECONE_INDEX
        )

        self.DB_URL = os.environ.get("DB_URL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

