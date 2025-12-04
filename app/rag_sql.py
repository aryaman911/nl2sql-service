# app/rag_sql.py

from typing import Iterable, List, Optional, Union

from openai import OpenAI

from app.config import settings
from app.schema_docs import SchemaChunk

# OpenAI client created from your config
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# e.g. "gpt-4.1-mini"
MODEL_NAME = settings.OPENAI_MODEL


def _chunks_to_text(chunks: Iterable[Union[str, SchemaChunk]]) -> str:
    """
    Convert a list of SchemaChunk objects (or strings) into a single
    text block for the prompt.

    This prevents:
      TypeError: sequence item 0: expected str instance, SchemaChunk found
    """
    texts: List[str] = []

    for ch in chunks:
        if isinstance(ch, str):
            texts.append(ch)
        elif isinstance(ch, SchemaChunk):
            # Be defensive: try common fields, then fallback to repr
            if getattr(ch, "text", None):
                texts.append(ch.text)
            elif getattr(ch, "ddl", None):
                texts.append(ch.ddl)
            elif getattr(ch, "create_sql", None):
                texts.append(ch.create_sql)
            else:
                texts.append(str(ch))
        else:
            texts.append(str(ch))

    return "\n\n".join(texts)


def generate_sql(
    question: str,
    schema_chunks: Iterable[Union[str, SchemaChunk]],
    roster_id: Optional[int] = None,
    client_id: Optional[int] = None,
) -> str:
    """
    Turn a natural language question + schema context into a single SQL string.

    - Patient-centric: treat patient as the main entity.
    - Use MySQL syntax.
    - Runs even if roster_id / client_id are None.

    Returns:
      sql (str): a single SQL statement. No explanation, no Markdown.
    """

    schema_context = _chunks_to_text(schema_chunks)

    # ------------------------------------------------------------------
    # Dynamic scoping instructions
    # ------------------------------------------------------------------
    scope_lines: List[str] = [
        "Think from a PATIENT-CENTRIC perspective.",
        "Whenever it makes sense, return one row per patient.",
        "Use the `patient` table as the driving table, aliased as `p`.",
        "Unless the user explicitly asks only for an aggregate (like COUNT), "
        "start the SELECT clause with `SELECT p.id` as the first column, "
        "then add any other needed patient columns.",
        "Use MySQL-compatible SQL syntax.",
    ]

    if roster_id is not None:
        scope_lines.append(
            f"- The selected roster_id is {roster_id}. "
            "Restrict results to patients in that roster.\n"
            "  Typically you should join the roster/patient association table, "
            "for example:\n"
            "  `FROM patient p\n"
            "   JOIN roster_patient rp ON rp.patient_id = p.id\n"
            "     AND rp.is_deleted = 0\n"
            "     AND rp.is_active = 1\n"
            f"     AND rp.roster_id = {roster_id}`\n"
            "  Adjust table and column names to match the actual schema, but "
            "always limit to this roster when roster_id is given."
        )

    if client_id is not None:
        scope_lines.append(
            f"- The selected client_id is {client_id}. "
            "If relevant tables have a `client_id` column, add conditions like "
            f"`AND <table>.client_id = {client_id}` so the query is scoped to that client."
        )

    scope_instructions = "\n".join(scope_lines)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    system_prompt = f"""
You are an expert SQL generator for a complex healthcare MySQL database.

You must:
- Use ONLY the tables and columns that exist in the provided schema context.
- Choose the correct joins and filters based on the question.
- Prefer patient-level queries where each row corresponds to a patient.
- Use the `patient` table (aliased as `p`) as the main driving table whenever possible.
- Unless the user explicitly wants only an aggregate (e.g. COUNT(*)), start your query with:
    SELECT p.id
  and then any additional columns needed.
- Write valid MySQL SQL.
- Output a single valid SQL statement.
- Do NOT include explanations, comments, or Markdown.
- Do NOT wrap the SQL in backticks.
- Do NOT use placeholders like <value>; use concrete values when the question implies them.

SCOPING INSTRUCTIONS:
{scope_instructions}

DATABASE SCHEMA CONTEXT (excerpts):
{schema_context}
""".strip()

    # ------------------------------------------------------------------
    # User prompt
    # ------------------------------------------------------------------
    user_prompt = f"""
User question:
{question}

Return ONLY the SQL statement.
""".strip()

    # ------------------------------------------------------------------
    # Call OpenAI
    # ------------------------------------------------------------------
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    sql = (resp.choices[0].message.content or "").strip()

    # Some models sometimes wrap in ```sql ... ```
    if sql.startswith("```"):
        # strip surrounding backticks
        sql = sql.strip("`")
        # remove leading "sql" if present
        if sql.lower().startswith("sql"):
            sql = sql[3:].lstrip()

    return sql
