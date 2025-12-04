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
    """
    texts: List[str] = []

    for ch in chunks:
        if isinstance(ch, str):
            texts.append(ch)
        elif isinstance(ch, SchemaChunk):
            # Prefer a human-readable field; fall back to str(ch) if needed.
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

    # High-level scope instructions (patient + tenant)
    scope_lines: List[str] = [
        "- Always interpret questions from the PATIENT'S perspective.",
        "- Prefer using patient + roster tables to restrict scope when applicable.",
    ]
    if roster_id is not None:
        scope_lines.append(
            f"- The selected roster_id is {roster_id}; scope results to this roster "
            "via the appropriate roster/patient join (for example using a "
            "`roster_patient` table if present)."
        )
    else:
        scope_lines.append(
            "- No specific roster_id is selected; you may query patients without a roster filter."
        )

    if client_id is not None:
        scope_lines.append(
            f"- The current tenant / client_id is {client_id}; if a client field "
            "exists in relevant tables, filter to this client."
        )
    else:
        scope_lines.append(
            "- If no client_id is given, do NOT add any client/tenant filter."
        )

    scope_instructions = "\n".join(scope_lines)

    # Extra, more concrete rules for how to shape queries.
    # These are phrased generically so nothing breaks if a particular column
    # name is slightly different – the model will line it up with the schema.
    roster_rules = """
ROSTER + PATIENT QUERY RULES (VERY IMPORTANT):

- Patients live in the `patient` table. Always alias this table as `p`.
- If there is a junction table linking rosters to patients
  (for example `roster_patient`), alias it as `rp`.

WHEN A roster_id IS PROVIDED (non-null):

- Prefer the following structure:

  SELECT
      p.id,                  -- ALWAYS the first selected column
      ...other columns...
  FROM roster_patient rp     -- or the equivalent roster-patient link table
  JOIN patient p ON p.id = rp.patient_id
  WHERE
      rp.roster_id = <SELECTED_ROSTER_ID>
      AND (rp.is_active = 1  IF that column exists)
      AND (rp.is_deleted = 0 IF that column exists)
      -- plus any demographic filters (age, gender, insurance, etc.)
      -- derived from the user question.

- Age filters should be implemented using the date-of-birth column on `p`,
  for example (adjust to the actual column name):

  FLOOR(DATEDIFF(CURDATE(), p.date_of_birth) / 365.25) BETWEEN <MIN_AGE> AND <MAX_AGE>

WHEN roster_id IS NULL:

- You may query directly from `patient p` without joining the roster table.
- Still start SELECT with `p.id` whenever the query is about patients.

GENERAL DEMOGRAPHIC QUERY STYLE:

- Always include `p.id` as the first column in SELECT for patient-level lists.
- Add additional fields that are clearly relevant (name, DOB, gender, etc.)
  based on the schema and the question.
- Apply status / dropped / added time windows using the most appropriate
  date / status columns in the roster-patient or patient tables.
""".strip()

    system_prompt = f"""
You are an expert SQL generator for a complex healthcare MySQL database.

You must:
- Use ONLY the tables/columns that exist in the provided schema.
- Choose the correct joins and filters based on the question.
- Output a single valid SQL statement.
- Do NOT include explanations, comments, or Markdown.
- Do NOT wrap the SQL in backticks.
- Do NOT use placeholders like <value>; fill in concrete values when possible.

{scope_instructions}

{roster_rules}

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
        if sql.lower().startswith("sql"):
            sql = sql[3:].lstrip()

    return sql
