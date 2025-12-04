# app/rag_sql.py
import os
from typing import List, Optional

from openai import OpenAI

# OpenAI client (uses OPENAI_API_KEY from environment)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _build_system_prompt(
    schema_snippets: str,
    roster_id: Optional[int],
    client_id: Optional[int],
) -> str:
    """
    Build the system prompt that instructs the model how to write SQL.

    We inject roster_id / client_id so that all queries are scoped
    to the currently selected roster (and client if relevant).
    """
    extra_constraints: List[str] = [
        "Use ONLY the tables and columns from the provided schema context.",
        "Return exactly ONE valid SQL statement, no explanations.",
        "Do NOT wrap SQL in backticks or markdown.",
        "Prefer starting queries from patient-related tables when possible.",
    ]

    if roster_id is not None:
        extra_constraints.append(
            f"Always restrict queries to the selected roster_id = {roster_id}. "
            "This filter must appear in the WHERE / JOIN conditions, usually via the roster or roster_patient table."
        )

    if client_id is not None:
        extra_constraints.append(
            f"If a client dimension exists, also filter by client_id = {client_id} when appropriate."
        )

    constraints_text = "\n- ".join(extra_constraints)

    system_prompt = f"""
You are an expert SQL generator for a healthcare roster/patient database.

You receive:
- A natural language question from a care manager.
- A schema context containing table and column information.
- Optional roster_id and client_id to scope the query.

Your job:
- Decide which tables and joins are needed based on the question.
- Use the schema context to pick correct table and column names.
- Obey the roster/client constraints if they are provided.
- Return ONE SQL statement that can be executed directly.

IMPORTANT RULES:
- {constraints_text}

SCHEMA CONTEXT (tables, columns, relationships):
{schema_snippets}
"""
    return system_prompt.strip()


def generate_sql(
    question: str,
    schema_chunks: List[str],
    roster_id: Optional[int] = None,
    client_id: Optional[int] = None,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Main RAG + SQL generation function.

    - `schema_chunks` are the top-k relevant schema snippets from Pinecone.
    - `roster_id` and `client_id` are used only to guide the model;
      you don't interpolate them yourself into the SQL string here.
    """
    schema_text = "\n\n".join(schema_chunks) if schema_chunks else "NO SCHEMA CONTEXT"

    system_prompt = _build_system_prompt(
        schema_snippets=schema_text,
        roster_id=roster_id,
        client_id=client_id,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "User question:\n"
                f"{question}\n\n"
                "Return only the final SQL query as plain text."
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
    )

    content = resp.choices[0].message.content or ""
    # In case the model adds ```sql ``` fences etc, strip them.
    cleaned = content.strip()

    if cleaned.startswith("```"):
        # Remove any markdown fences
        cleaned = cleaned.strip("`")
        # Sometimes there's a leading "sql\n"
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned.split("\n", 1)[-1].strip()

    return cleaned
