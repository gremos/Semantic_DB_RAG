from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from orchestration.pipeline import SemanticPipeline
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Semantic Engine API", version="1.0.0")

# Global pipeline instance
pipeline = SemanticPipeline()

class DiscoverRequest(BaseModel):
    bypass_cache: bool = False

class ModelRequest(BaseModel):
    domain_hints: str = ""
    bypass_cache: bool = False

class QueryRequest(BaseModel):
    question: str
    domain_hints: Optional[str] = ""

@app.post("/api/discover")
async def discover(request: DiscoverRequest):
    """Run discovery phase."""
    success, error = pipeline.initialize(request.bypass_cache)
    if not success:
        raise HTTPException(status_code=500, detail=error)
    
    return {
        "status": "success",
        "data": pipeline.get_discovery_data()
    }

@app.post("/api/model")
async def create_model(request: ModelRequest):
    """Create semantic model."""
    # Ensure discovery ran
    if not pipeline.discovery_data:
        success, error = pipeline.initialize()
        if not success:
            raise HTTPException(status_code=500, detail=error)
    
    success, error = pipeline.create_semantic_model(
        request.domain_hints,
        request.bypass_cache
    )
    
    if not success:
        raise HTTPException(status_code=500, detail=error)
    
    return {
        "status": "success",
        "data": pipeline.get_semantic_model()
    }

@app.post("/api/query")
async def query(request: QueryRequest):
    """Answer natural language question."""
    # Ensure pipeline is ready
    if not pipeline.discovery_data:
        success, error = pipeline.initialize()
        if not success:
            raise HTTPException(status_code=500, detail=error)
    
    if not pipeline.semantic_model:
        success, error = pipeline.create_semantic_model(request.domain_hints)
        if not success:
            raise HTTPException(status_code=500, detail=error)
    
    success, answer, error = pipeline.answer_question(request.question)
    
    if not success:
        raise HTTPException(status_code=500, detail=error)
    
    return {
        "status": "success",
        "data": answer
    }

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    pipeline.cleanup()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)