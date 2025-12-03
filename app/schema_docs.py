"""
schema_docs.py

This version reads your schema from a .sql file and extracts each
CREATE TABLE ... statement as a SchemaChunk for RAG.
"""

from dataclasses import dataclass
from typing import List
import os
import re


@dataclass
class SchemaChunk:
    id: str
    text: str


SQL_FILE_PATH = os.environ.get("SCHEMA_SQL_PATH", "schema.sql")


def _read_sql_file() -> str:
    if not os.path.exists(SQL_FILE_PATH):
        raise FileNotFoundError(
            f"Schema SQL file not found at {SQL_FILE_PATH}. "
            "Set SCHEMA_SQL_PATH env or place schema.sql in project root."
        )
    with open(SQL_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _extract_create_table_blocks(sql_text: str) -> List[SchemaChunk]:
    """
    Very simple parser: finds blocks like

      CREATE TABLE table_name (
          ...
      );

    and returns each block as a chunk.
    """
    # Normalize whitespace a bit
    cleaned = sql_text

    # Regex to match CREATE TABLE ... ; (not bulletproof, but works for most schemas)
    pattern = re.compile(
        r"CREATE\s+TABLE\s+`?([A-Za-z0-9_]+)`?\s*\((.*?)\);",
        re.IGNORECASE | re.DOTALL,
    )

    chunks: List[SchemaChunk] = []

    for match in pattern.finditer(cleaned):
        table_name = match.group(1)
        body = match.group(0).strip()  # full CREATE TABLE ... ;

        # You can enrich this with extra description if you want
        text = (
            f"Table: {table_name}\n"
            "DDL:\n"
            f"{body}\n"
        )

        chunks.append(SchemaChunk(id=table_name, text=text))

    return chunks


def load_schema_chunks() -> List[SchemaChunk]:
    sql_text = _read_sql_file()
    chunks = _extract_create_table_blocks(sql_text)

    if not chunks:
        raise RuntimeError(
            "No CREATE TABLE statements found in schema SQL file. "
            "Check SCHEMA_SQL_PATH or SQL format."
        )

    return chunks
