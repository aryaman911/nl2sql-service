# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.models import NL2SQLRequest, NL2SQLResponse
from app.vector_store import PineconeSchemaVectorStore
from app.rag_sql import generate_sql

app = FastAPI(
    title="NL2SQL Service",
    version="1.0.0",
)

# Allow your local/backend origins; keep "*" for now while testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vector store – loads schema + pinecone once at startup
vector_store = PineconeSchemaVectorStore()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/nl2sql", response_model=NL2SQLResponse)
def nl2sql(request: NL2SQLRequest):
    """
    Main endpoint called by Django's CareOpenAI.

    It:
      1. Uses the question to find relevant schema chunks from Pinecone
      2. Calls OpenAI with schema + question (+ roster/client context)
      3. Returns a single SQL string in `sql`
    """
    try:
        # If your class has a different method name, adapt this line,
        # e.g. similarity_search() or query().
        schema_chunks = vector_store.search(request.question, top_k=8)
    except Exception as e:
        # Don't crash if vector search fails – just log and continue
        print("[/nl2sql] Error during vector search:", repr(e))
        schema_chunks = []

    try:
        sql = generate_sql(
            question=request.question,
            schema_chunks=schema_chunks,
            roster_id=request.roster_id,
            client_id=request.client_id,
        )
    except Exception as e:
        # Surface a *useful* error instead of plain 500 HTML
        print("[/nl2sql] Error during SQL generation:", repr(e))
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Error generating SQL",
                "error": str(e),
            },
        )

    return NL2SQLResponse(
        question=request.question,
        roster_id=request.roster_id,
        client_id=request.client_id,
        sql=sql,
    )
