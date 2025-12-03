from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .models import NL2SQLRequest, NL2SQLResponse
from .vector_store import PineconeSchemaVectorStore
from .rag_sql import NL2SQLPipeline

settings = get_settings()

app = FastAPI(
    title="NL2SQL RAG Service",
    version="0.1.0",
    description="Cloud microservice that converts natural language to SQL using RAG over schema in Pinecone.",
)

# CORS (you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build vector store & pipeline once at startup
vector_store = PineconeSchemaVectorStore()
pipeline = NL2SQLPipeline(vector_store=vector_store)


@app.post("/nl2sql", response_model=NL2SQLResponse)
async def nl2sql_endpoint(payload: NL2SQLRequest) -> NL2SQLResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        sql = pipeline.question_to_sql(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return NL2SQLResponse(sql=sql)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
