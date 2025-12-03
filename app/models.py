from pydantic import BaseModel


class NL2SQLRequest(BaseModel):
    question: str


class NL2SQLResponse(BaseModel):
    sql: str
