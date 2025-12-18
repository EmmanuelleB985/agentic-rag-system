"""
FastAPI server for Agentic RAG System
Provides REST and WebSocket endpoints for multimodal RAG
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
import json
import uuid
import base64
from PIL import Image
import io
import tempfile

from agentic_rag import AgenticRAG, MultimodalAgenticRAG
from agentic_rag.multimodal import Modality
from agentic_rag.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG System API",
    description="Multimodal Retrieval-Augmented Generation API",
    version="2.0.0"
)

# Load configuration
config = load_config("../config.yaml")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors", {}).get("origins", ["*"]),
    allow_credentials=True,
    allow_methods=config.get("api", {}).get("cors", {}).get("methods", ["GET", "POST"]),
    allow_headers=["*"],
)

# Global RAG instance
rag_system = None
sessions = {}
MAX_SESSIONS = 1000
SESSION_TIMEOUT = 3600  # 1 hour

# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text")
    use_agent: bool = Field(True, description="Use agent for reasoning")
    top_k: int = Field(5, description="Number of documents to retrieve")
    modalities: Optional[List[str]] = Field(None, description="Modalities to search")
    session_id: Optional[str] = Field(None, description="Session ID for context")

class ImageQueryRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    query: Optional[str] = Field(None, description="Optional text query")
    use_agent: bool = Field(True, description="Use agent for reasoning")

class DocumentUpload(BaseModel):
    content: str = Field(..., description="Document content")
    modality: str = Field("text", description="Document modality")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    session_id: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    gpu_available: bool
    memory_usage: Dict[str, float]

# Session Management
class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.history = []
        self.documents = []
    
    def update(self):
        self.last_accessed = datetime.now()

def get_or_create_session(session_id: Optional[str] = None) -> Session:
    """Get existing or create new session"""
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session.update()
    else:
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        sessions[session_id] = session
    
    # Clean old sessions
    if len(sessions) > MAX_SESSIONS:
        oldest = sorted(sessions.items(), key=lambda x: x[1].last_accessed)
        for sid, _ in oldest[:len(sessions) - MAX_SESSIONS]:
            del sessions[sid]
    
    return session

# Initialize RAG System
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    
    logger.info("Initializing RAG System...")
    
    try:
        # Determine system type from config
        if config.get("multimodal", {}).get("process_images", False):
            rag_system = MultimodalAgenticRAG(
                model_name=config["models"]["language_model"]["name"],
                device=config["system"]["device"],
                quantization=config["models"]["language_model"]["quantization"]
            )
        else:
            rag_system = AgenticRAG(
                model_name=config["models"]["language_model"]["name"],
                device=config["system"]["device"],
                quantization=config["models"]["language_model"]["quantization"]
            )
        
        logger.info("RAG System initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down RAG System...")
    # Save indices if configured
    if rag_system and config.get("storage", {}).get("auto_save", False):
        rag_system.save_index(config["storage"]["index_path"])

# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "name": "Agentic RAG System API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": [
            "/docs",
            "/health",
            "/api/query",
            "/api/upload",
            "/api/search"
        ]
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    import torch
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1e9
    
    gpu_available = torch.cuda.is_available()
    gpu_memory = 0
    if gpu_available:
        gpu_memory = torch.cuda.memory_allocated(0) / 1e9
    
    return HealthResponse(
        status="healthy" if rag_system else "initializing",
        timestamp=datetime.now().isoformat(),
        gpu_available=gpu_available,
        memory_usage={
            "cpu_gb": cpu_memory,
            "gpu_gb": gpu_memory,
            "total_gb": cpu_memory + gpu_memory
        }
    )

@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """Process text query"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        session = get_or_create_session(request.session_id)
        
        # Process query
        if isinstance(rag_system, MultimodalAgenticRAG):
            # Multimodal query
            modalities = []
            if request.modalities:
                modalities = [Modality[m.upper()] for m in request.modalities]
            
            result = await rag_system.multimodal_query(
                request.query,
                modalities=modalities if modalities else None,
                use_agent=request.use_agent,
                top_k=request.top_k
            )
        else:
            # Standard text query
            result = rag_system.query(
                request.query,
                use_agent=request.use_agent,
                top_k=request.top_k
            )
        
        # Update session
        session.history.append({
            "query": request.query,
            "answer": result.answer,
            "timestamp": datetime.now().isoformat()
        })
        
        # Prepare response
        sources = []
        for source in result.sources[:request.top_k]:
            sources.append({
                "content": str(source.content)[:200] if hasattr(source, 'content') else str(source)[:200],
                "score": getattr(source, 'score', 0.0),
                "metadata": getattr(source, 'metadata', {})
            })
        
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            confidence=result.confidence,
            processing_time=result.processing_time or 0.0,
            session_id=session.id
        )
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/image", response_model=QueryResponse, tags=["Query"])
async def query_image(request: ImageQueryRequest):
    """Process image-based query"""
    if not isinstance(rag_system, MultimodalAgenticRAG):
        raise HTTPException(status_code=400, detail="Multimodal system required")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        
        # Create query
        if request.query:
            query = (request.query, image)
        else:
            query = image
        
        # Process query
        result = await rag_system.multimodal_query(
            query,
            modalities=[Modality.IMAGE, Modality.TEXT],
            use_agent=request.use_agent
        )
        
        # Prepare response
        sources = []
        for source in result.sources[:5]:
            sources.append({
                "modality": source.modality.value if hasattr(source, 'modality') else "unknown",
                "score": getattr(source, 'score', 0.0)
            })
        
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            confidence=result.confidence,
            processing_time=result.processing_time or 0.0,
            session_id=str(uuid.uuid4())
        )
    
    except Exception as e:
        logger.error(f"Image query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload", tags=["Documents"])
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a file"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Determine modality from file extension
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext in ['.txt', '.md', '.pdf', '.docx']:
            modality = Modality.TEXT
        elif file_ext in ['.jpg', '.jpeg', '.png', '.gif']:
            modality = Modality.IMAGE
        elif file_ext in ['.mp3', '.wav', '.flac']:
            modality = Modality.AUDIO
        elif file_ext in ['.mp4', '.avi', '.mov']:
            modality = Modality.VIDEO
        else:
            modality = Modality.TEXT
        
        # Process file
        if isinstance(rag_system, MultimodalAgenticRAG):
            result = await rag_system.add_multimodal_document(
                tmp_path,
                modality,
                metadata={"filename": file.filename}
            )
        else:
            # For text RAG, read the content
            with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            num_chunks = rag_system.add_documents([text_content])
            result = f"Added {num_chunks} chunks"
        
        # Clean up
        Path(tmp_path).unlink()
        
        return {
            "status": "success",
            "message": result,
            "filename": file.filename,
            "modality": modality.value if hasattr(modality, 'value') else str(modality)
        }
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload/batch", tags=["Documents"])
async def upload_batch(files: List[UploadFile] = File(...)):
    """Upload multiple files"""
    results = []
    for file in files:
        try:
            result = await upload_file(file)
            results.append(result)
        except Exception as e:
            results.append({
                "status": "error",
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

@app.get("/api/stream", tags=["Streaming"])
async def stream_query(query: str, use_agent: bool = True):
    """Stream responses using Server-Sent Events"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    async def generate():
        """Generate streaming response"""
        try:
            # Simulate streaming (would be actual streaming in production)
            result = rag_system.query(query, use_agent=use_agent)
            
            # Stream tokens
            tokens = result.answer.split()
            for i, token in enumerate(tokens):
                yield f"data: {json.dumps({'token': token + ' ', 'done': False})}\n\n"
                await asyncio.sleep(0.05)  # Simulate streaming delay
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "query":
                # Process query
                query = message.get("query", "")
                
                if rag_system:
                    result = rag_system.query(query)
                    response = {
                        "type": "response",
                        "answer": result.answer,
                        "session_id": session_id
                    }
                else:
                    response = {
                        "type": "error",
                        "message": "System not initialized"
                    }
                
                await websocket.send_text(json.dumps(response))
            
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@app.post("/api/search/crossmodal", tags=["Search"])
async def cross_modal_search(
    text_query: Optional[str] = None,
    image_query: Optional[str] = None,
    top_k: int = 5
):
    """Cross-modal search"""
    if not isinstance(rag_system, MultimodalAgenticRAG):
        raise HTTPException(status_code=400, detail="Multimodal system required")
    
    try:
        query = None
        
        if text_query and image_query:
            # Both text and image
            image_data = base64.b64decode(image_query)
            image = Image.open(io.BytesIO(image_data))
            query = (text_query, image)
        elif text_query:
            query = text_query
        elif image_query:
            image_data = base64.b64decode(image_query)
            query = Image.open(io.BytesIO(image_data))
        else:
            raise HTTPException(status_code=400, detail="No query provided")
        
        result = await rag_system.multimodal_query(
            query,
            modalities=[Modality.TEXT, Modality.IMAGE],
            use_agent=False,
            top_k=top_k
        )
        
        return {
            "results": [
                {
                    "modality": doc.modality.value if hasattr(doc, 'modality') else "unknown",
                    "score": getattr(doc, 'score', 0.0)
                }
                for doc in result.sources
            ]
        }
    
    except Exception as e:
        logger.error(f"Cross-modal search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/transcribe", tags=["Audio"])
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file"""
    if not isinstance(rag_system, MultimodalAgenticRAG):
        raise HTTPException(status_code=400, detail="Multimodal system required")
    
    try:
        # Save audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Transcribe
        transcription = await rag_system.mm_agent._tool_transcribe_audio({
            "audio_path": tmp_path
        })
        
        # Clean up
        Path(tmp_path).unlink()
        
        return {
            "transcription": transcription,
            "filename": file.filename
        }
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/session/{session_id}", tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete a session"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/sessions", tags=["Sessions"])
async def list_sessions():
    """List active sessions"""
    return {
        "sessions": [
            {
                "id": session.id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "history_length": len(session.history)
            }
            for session in sessions.values()
        ]
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get config
    api_config = config.get("api", {})
    
    # Run server
    uvicorn.run(
        app,
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        workers=api_config.get("workers", 4),
        reload=api_config.get("reload", False)
    )
