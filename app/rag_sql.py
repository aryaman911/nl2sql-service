# app/rag_sql.py

from typing import Iterable, List, Optional, Union
from openai import OpenAI

from app.config import settings
from app.schema_docs import SchemaChunk

client = OpenAI(api_key=settings.OPENAI_API_KEY)

MODEL_NAME = settings.OPENAI_MODEL  # e.g. "gpt-4.1-mini"


def _chunks_to_text(chunks: Iterable[Union[str, SchemaChunk]]) -> str:
    """
    Convert a list of SchemaChunk objects (or strings) into a single
    text block for the prompt.

    This is what fixes the error:
      TypeError: sequence item 0: expected str instance, SchemaChunk found
    """
    texts: List[str] = []

    for ch in chunks:
        if isinstance(ch, str):
            texts.append(ch)
        elif isinstance(ch, SchemaChunk):
            # Pick the most useful field from your SchemaChunk dataclass.
            # Adjust these attribute names if your dataclass is different.
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

    Returns:
      sql (str): a single SQL statement. No explanation, no Markdown.
    """
    schema_context = _chunks_to_text(schema_chunks)

    scope_lines: List[str] = [
        "- Always generate SQL from the PATIENT'S perspective.",
        "- Use patient and roster tables to restrict scope when applicable.",
    ]
    if roster_id:
        scope_lines.append(
            f"- Restrict results to the selected roster_id = {roster_id} "
            "(use the appropriate roster/patient join columns)."
        )
    if client_id:
        scope_lines.append(
            f"- Assume data belongs to client_id = {client_id} if the schema includes a client field."
        )

    scope_instructions = "\n".join(scope_lines)

    system_prompt = f"""
You are an expert SQL generator for a complex healthcare MySQL database.

You must:
- Use ONLY the tables/columns that exist in the provided schema.
- Choose the correct joins and filters based on the question.
- Output a single valid SQL statement.
- Do NOT include explanations, comments, comments markers, or Markdown.
- Do NOT wrap the SQL in backticks.
- Do NOT use placeholders like <value>; fill in concrete values when possible.

{scope_instructions}

DATABASE SCHEMA CONTEXT (excerpts):
{schema_context}
""".strip()

    user_prompt = f"""
User question:
{question}

Return ONLY the SQL statement.
""".strip()

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    sql = resp.choices[0].message.content.strip()
    # Sometimes models prepend ```sql ...```, so strip that just in case.
    if sql.startswith("```"):
        sql = sql.strip("`")
        # remove possible leading "sql\n"
        if sql.lower().startswith("sql"):
            sql = sql[3:].lstrip()

    return sql
