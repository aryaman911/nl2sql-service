"""
schema_docs.py

Reads your schema from maindata.sql and extracts each
CREATE TABLE ... statement as a SchemaChunk for RAG.
Supports large schemas (100+ tables) by truncating long text.
"""

from dataclasses import dataclass
from typing import List
import os
import re


@dataclass
class SchemaChunk:
    id: str
    text: str


# Default location now points to maindata.sql
SQL_FILE_PATH = os.environ.get("SCHEMA_SQL_PATH", "maindata.sql")


def _read_sql_file() -> str:
    if not os.path.exists(SQL_FILE_PATH):
        raise FileNotFoundError(
            f"Schema SQL file not found at {SQL_FILE_PATH}. "
            "Set SCHEMA_SQL_PATH env or place maindata.sql in project root."
        )
    with open(SQL_FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    """
    Ensure that any single schema chunk is not insanely large.
    4000 characters ~ 1000 tokens (roughly), safe for embeddings.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars] + "\n-- (truncated for embedding)\n"


def _extract_create_table_blocks(sql_text: str) -> List[SchemaChunk]:
    """
    Extract all CREATE TABLE blocks as SchemaChunk objects.

    Handles MySQL/Postgres-style:
        CREATE TABLE table_name ( ... );
    """

    # Remove comments to reduce noise
    no_single_line_comments = re.sub(r"--.*?$", "", sql_text, flags=re.MULTILINE)
    no_block_comments = re.sub(r"/\*.*?\*/", "", no_single_line_comments, flags=re.DOTALL)

    pattern = re.compile(
        r"CREATE\s+TABLE\s+`?([A-Za-z0-9_]+)`?\s*\((.*?)\);",
        re.IGNORECASE | re.DOTALL,
    )

    chunks: List[SchemaChunk] = []

    for match in pattern.finditer(no_block_comments):
        table_name = match.group(1)
        table_body = match.group(2)

        lines = [line.strip() for line in table_body.splitlines() if line.strip()]

        cleaned_cols: List[str] = []
        constraints: List[str] = []

        for line in lines:
            lower_line = line.lower()
            if (
                " primary key" in lower_line
                or " foreign key" in lower_line
                or lower_line.startswith("constraint")
                or lower_line.startswith("unique")
                or lower_line.startswith("index")
                or lower_line.startswith("key ")
            ):
                constraints.append(line.rstrip(","))
            else:
                cleaned_cols.append(line.rstrip(","))

        # Clip very wide tables: only first N column lines and constraint lines
        MAX_COL_LINES = 40
        if len(cleaned_cols) > MAX_COL_LINES:
            cleaned_cols = cleaned_cols[:MAX_COL_LINES]
            cleaned_cols.append("-- (columns truncated for brevity)")

        MAX_CONSTRAINT_LINES = 15
        if len(constraints) > MAX_CONSTRAINT_LINES:
            constraints = constraints[:MAX_CONSTRAINT_LINES]
            constraints.append("-- (constraints truncated for brevity)")

        compact_body_lines = cleaned_cols + constraints
        compact_body = ",\n  ".join(compact_body_lines)

        raw_text = (
            f"Table: {table_name}\n"
            "Columns and constraints (simplified):\n"
            f"CREATE TABLE {table_name} (\n"
            f"  {compact_body}\n"
            ");"
        )

        text = _truncate_text(raw_text)  # <-- hard length cap per table

        chunks.append(SchemaChunk(id=table_name, text=text))

    return chunks


def load_schema_chunks() -> List[SchemaChunk]:
    sql_text = _read_sql_file()
    chunks = _extract_create_table_blocks(sql_text)

    if not chunks:
        raise RuntimeError(
            "No CREATE TABLE statements found in maindata.sql. "
            "Check SCHEMA_SQL_PATH or the SQL format."
        )

    print(f"[schema_docs] Loaded {len(chunks)} tables from schema.")
    return chunks
