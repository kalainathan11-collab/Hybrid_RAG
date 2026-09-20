from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
#from rag_pipeline import HybridRAGEngine
#from .rag_pipeline import HybridRAGEngine
try:
    from rag_pipeline import HybridRAGEngine
except ImportError:
    from .rag_pipeline import HybridRAGEngine
# 1. Initialize FastAPI Application
app = FastAPI(
    title="Spotify Architecture Hybrid RAG API",
    description="Backend service linking ChromaDB Vector Search and Neo4j Knowledge Graph",
    version="1.0.0"
)

rag_engine = None

# 2. Lifecycle Event: Load RAG Pipeline on Startup
@app.on_event("startup")
def startup_event():
    global rag_engine
    print("Initializing Hybrid RAG Engine (Loading ChromaDB & Neo4j connections)...")
    rag_engine = HybridRAGEngine()
    print("Hybrid RAG Engine successfully loaded.")

# 3. Data Schema for Request and Response
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    vector_context: str
    graph_context: str

# 4. Health Check Endpoint
@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "Spotify Hybrid RAG API is active. Send POST requests to /query."
    }

# 5. Main Query Endpoint
@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = rag_engine.query(request.question)
        return QueryResponse(
            answer=result["answer"],
            vector_context=result["vector_context"],
            graph_context=result["graph_context"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")