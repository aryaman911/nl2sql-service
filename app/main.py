# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.models import NL2SQLRequest, NL2SQLResponse
from app.vector_store import PineconeSchemaVectorStore
from app.rag_sql import generate_sql

app = FastAPI(
    title="NL2SQL Service",
    version="1.0.0",
)

# CORS – allow your Django frontend / backend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vector store (loads schema + Pinecone)
vector_store = PineconeSchemaVectorStore()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/nl2sql", response_model=NL2SQLResponse)
def nl2sql(request: NL2SQLRequest) -> NL2SQLResponse:
    """
    Main endpoint called by Django's CareOpenAI.

    It now understands roster_id and client_id and passes them
    to the SQL generator so all queries can be scoped properly.
    """
    # Retrieve relevant schema chunks via vector search
    schema_chunks = vector_store.search(request.question, top_k=8)

    sql = generate_sql(
        question=request.question,
        schema_chunks=schema_chunks,
        roster_id=request.roster_id,
        client_id=request.client_id,
    )

    return NL2SQLResponse(
        question=request.question,
        roster_id=request.roster_id,
        client_id=request.client_id,
        sql=sql,
    )
