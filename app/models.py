# app/models.py
from typing import Optional
from pydantic import BaseModel


class NL2SQLRequest(BaseModel):
    """
    Request body from your Django backend (CareOpenAI).

    It now includes optional roster_id and client_id so that
    SQL can be scoped to the currently selected roster/client.
    """
    question: str
    roster_id: Optional[int] = None
    client_id: Optional[int] = None


class NL2SQLResponse(BaseModel):
    """
    Response returned to Django.

    `sql` is the generated SQL query. You can also expose it as
    `sql_one` from Django if you like.
    """
    question: str
    roster_id: Optional[int] = None
    client_id: Optional[int] = None
    sql: str
