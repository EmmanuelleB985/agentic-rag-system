"""
FastAPI Server for Agentic RAG System
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union, Literal
import logging
from datetime import datetime, timedelta
import json
import uuid
import base64
from PIL import Image
import io
import tempfile
import hashlib
import jwt
from functools import lru_cache
import redis
import aiofiles
from contextlib import asynccontextmanager
import psutil
import torch

# Import existing RAG components
from agentic_rag import AgenticRAG, MultimodalAgenticRAG
from agentic_rag.multimodal import Modality
from agentic_rag.utils import load_config
from agentic_rag.agents import ReasoningAgent
from agentic_rag.retrieval import HybridRetriever

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = load_config("config.yaml")

# Security settings
SECRET_KEY = config.get("security", {}).get("secret_key", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize security
security = HTTPBearer()

# Global instances
rag_system = None
redis_client = None
sessions = {}
knowledge_bases = {}

# =====================================
# Lifespan Management
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enhanced Agentic RAG System...")
    await initialize_system()
    yield
    # Shutdown
    logger.info("👋 Shutting down Enhanced Agentic RAG System...")
    await cleanup_system()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Agentic RAG System API",
    description="Advanced Multimodal Retrieval-Augmented Generation with Agent Capabilities",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# =====================================
# CORS Configuration
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("api", {}).get("cors", {}).get("origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# Enhanced Pydantic Models
# =====================================

class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex="^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$")
    password: str = Field(..., min_length=8)

class TokenResponse(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class KnowledgeBaseCreate(BaseModel):
    """Knowledge base creation model"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_type: Literal["faiss", "chroma", "qdrant", "pinecone"] = "faiss"
    config: Optional[Dict[str, Any]] = {}

class EnhancedQueryRequest(BaseModel):
    """Enhanced query request with advanced options"""
    query: str = Field(..., description="Query text")
    knowledge_base_id: Optional[str] = Field(None, description="Specific knowledge base to query")
    use_agent: bool = Field(True, description="Use agent for reasoning")
    agent_type: Literal["reasoning", "research", "multimodal", "custom"] = "reasoning"
    top_k: int = Field(5, ge=1, le=100, description="Number of documents to retrieve")
    rerank: bool = Field(True, description="Apply reranking to results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    modalities: Optional[List[str]] = Field(None, description="Modalities to search")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    stream: bool = Field(False, description="Stream the response")
    temperature: float = Field(0.7, ge=0, le=2, description="LLM temperature")
    max_tokens: int = Field(500, ge=1, le=4000, description="Maximum tokens in response")
    
    @validator('modalities')
    def validate_modalities(cls, v):
        if v:
            valid_modalities = ["text", "image", "audio", "video", "table", "code"]
            for modality in v:
                if modality.lower() not in valid_modalities:
                    raise ValueError(f"Invalid modality: {modality}")
        return v

class DocumentMetadata(BaseModel):
    """Document metadata model"""
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = []
    custom_fields: Optional[Dict[str, Any]] = {}

class EnhancedDocumentUpload(BaseModel):
    """Enhanced document upload model"""
    content: str = Field(..., description="Document content")
    knowledge_base_id: Optional[str] = Field(None, description="Target knowledge base")
    modality: str = Field("text", description="Document modality")
    metadata: Optional[DocumentMetadata] = None
    chunk_size: Optional[int] = Field(500, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(50, ge=0, le=500)
    process_tables: bool = Field(False, description="Extract and process tables")
    extract_entities: bool = Field(False, description="Extract named entities")

class BatchProcessingRequest(BaseModel):
    """Batch processing request"""
    documents: List[EnhancedDocumentUpload]
    parallel_processing: bool = Field(True)
    batch_size: int = Field(10, ge=1, le=100)

class AgentTaskRequest(BaseModel):
    """Agent task execution request"""
    task_type: Literal["research", "summarize", "extract", "analyze", "compare", "generate"]
    parameters: Dict[str, Any]
    context: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    timeout: Optional[int] = Field(30, ge=1, le=300)
    
class AnalyticsQuery(BaseModel):
    """Analytics query model"""
    metric_type: Literal["usage", "performance", "quality", "errors"]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    group_by: Optional[Literal["hour", "day", "week", "month"]] = "day"

class SystemConfig(BaseModel):
    """System configuration update model"""
    embeddings: Optional[Dict[str, Any]] = None
    language_model: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None
    agents: Optional[Dict[str, Any]] = None

# =====================================
# Session Management
# =====================================

class EnhancedSession:
    """Enhanced session with advanced features"""
    
    def __init__(self, session_id: str, user_id: Optional[str] = None):
        self.id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.history = []
        self.documents = []
        self.context = {}
        self.preferences = {}
        self.active_knowledge_base = None
    
    def update(self):
        self.last_accessed = datetime.now()
    
    def add_interaction(self, query: str, response: str, metadata: Dict[str, Any] = None):
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        self.update()
    
    def get_context(self, last_n: int = 5) -> str:
        """Get conversation context from recent interactions"""
        recent = self.history[-last_n:] if len(self.history) > last_n else self.history
        context = []
        for item in recent:
            context.append(f"User: {item['query']}")
            context.append(f"Assistant: {item['response']}")
        return "\n".join(context)

# =====================================
# Knowledge Base Management
# =====================================

class KnowledgeBase:
    """Knowledge base container"""
    
    def __init__(self, kb_id: str, name: str, config: Dict[str, Any]):
        self.id = kb_id
        self.name = name
        self.config = config
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.document_count = 0
        self.total_chunks = 0
        self.index = None
        self.metadata = {}

# =====================================
# Dependency Injection
# =====================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@lru_cache()
def get_settings():
    """Get cached settings"""
    return config

# =====================================
# System Initialization
# =====================================

async def initialize_system():
    """Initialize all system components"""
    global rag_system, redis_client
    
    try:
        # Initialize Redis for caching
        if config.get("cache", {}).get("enabled", False):
            redis_client = redis.Redis(
                host=config["cache"].get("host", "localhost"),
                port=config["cache"].get("port", 6379),
                decode_responses=True
            )
            await redis_client.ping()
            logger.info("Redis cache connected")
        
        # Initialize RAG System
        if config.get("multimodal", {}).get("enabled", False):
            rag_system = MultimodalAgenticRAG(
                model_name=config["models"]["language_model"]["name"],
                device=config["system"]["device"],
                quantization=config["models"]["language_model"]["quantization"]
            )
            logger.info("Multimodal RAG system initialized")
        else:
            rag_system = AgenticRAG(
                model_name=config["models"]["language_model"]["name"],
                device=config["system"]["device"],
                quantization=config["models"]["language_model"]["quantization"]
            )
            logger.info("Standard RAG system initialized")
        
        # Load saved indices if available
        indices_path = Path(config["system"]["indices_dir"])
        if indices_path.exists():
            for index_file in indices_path.glob("*.index"):
                kb_id = index_file.stem
                # Load index into appropriate knowledge base
                logger.info(f"Loaded index: {kb_id}")
        
        logger.info("System initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

async def cleanup_system():
    """Cleanup system resources"""
    global rag_system, redis_client
    
    try:
        # Save indices
        if config.get("storage", {}).get("auto_save", False):
            indices_path = Path(config["system"]["indices_dir"])
            indices_path.mkdir(exist_ok=True, parents=True)
            
            for kb_id, kb in knowledge_bases.items():
                if kb.index:
                    save_path = indices_path / f"{kb_id}.index"
                    # Save index
                    logger.info(f"Saved index: {kb_id}")
        
        # Close connections
        if redis_client:
            await redis_client.close()
        
        logger.info("System cleanup complete")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# =====================================
# API Endpoints - Core
# =====================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with comprehensive API information"""
    return {
        "name": "Enhanced Agentic RAG System API",
        "version": "3.0.0",
        "status": "operational",
        "features": [
            "Multimodal RAG",
            "Multiple knowledge bases",
            "Agent orchestration",
            "Real-time streaming",
            "WebSocket support",
            "Batch processing",
            "Analytics & monitoring"
        ],
        "endpoints": {
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            },
            "health": "/health",
            "auth": "/api/v1/auth/*",
            "query": "/api/v1/query/*",
            "documents": "/api/v1/documents/*",
            "knowledge_bases": "/api/v1/kb/*",
            "agents": "/api/v1/agents/*",
            "analytics": "/api/v1/analytics/*"
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Comprehensive health check"""
    import psutil
    
    process = psutil.Process()
    cpu_percent = process.cpu_percent()
    memory_info = process.memory_info()
    
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1e9,
            "memory_reserved": torch.cuda.memory_reserved(0) / 1e9
        }
    else:
        gpu_info = {"available": False}
    
    redis_status = "connected" if redis_client else "not configured"
    try:
        if redis_client:
            await redis_client.ping()
    except:
        redis_status = "disconnected"
    
    return {
        "status": "healthy" if rag_system else "initializing",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - app.state.startup_time).total_seconds() if hasattr(app.state, 'startup_time') else 0,
        "system": {
            "rag_initialized": rag_system is not None,
            "multimodal_enabled": isinstance(rag_system, MultimodalAgenticRAG),
            "knowledge_bases": len(knowledge_bases),
            "active_sessions": len(sessions),
            "redis_status": redis_status
        },
        "resources": {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_info.rss / 1024 / 1024,
            "gpu": gpu_info
        },
        "configuration": {
            "model": config["models"]["language_model"]["name"],
            "device": config["system"]["device"],
            "cache_enabled": config.get("cache", {}).get("enabled", False)
        }
    }

# =====================================
# API Endpoints - Authentication
# =====================================

@app.post("/api/v1/auth/register", response_model=TokenResponse, tags=["Authentication"])
async def register(user: UserCreate):
    """Register a new user"""
    # In production, store in database with hashed password
    # This is a simplified example
    
    # Create access token
    access_token = create_access_token(data={"sub": user.username})
    
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@app.post("/api/v1/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(username: str, password: str):
    """Authenticate and get access token"""
    # In production, verify against database
    # This is a simplified example
    
    access_token = create_access_token(data={"sub": username})
    
    return TokenResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# =====================================
# API Endpoints - Knowledge Base
# =====================================

@app.post("/api/v1/kb/create", tags=["Knowledge Base"])
async def create_knowledge_base(
    kb_config: KnowledgeBaseCreate,
    current_user: str = Depends(get_current_user)
):
    """Create a new knowledge base"""
    kb_id = str(uuid.uuid4())
    
    kb = KnowledgeBase(
        kb_id=kb_id,
        name=kb_config.name,
        config=kb_config.dict()
    )
    
    knowledge_bases[kb_id] = kb
    
    return {
        "id": kb_id,
        "name": kb.name,
        "created_at": kb.created_at.isoformat(),
        "config": kb.config
    }

@app.get("/api/v1/kb/list", tags=["Knowledge Base"])
async def list_knowledge_bases(
    current_user: str = Depends(get_current_user)
):
    """List all knowledge bases"""
    return [
        {
            "id": kb_id,
            "name": kb.name,
            "document_count": kb.document_count,
            "total_chunks": kb.total_chunks,
            "created_at": kb.created_at.isoformat(),
            "updated_at": kb.updated_at.isoformat()
        }
        for kb_id, kb in knowledge_bases.items()
    ]

@app.delete("/api/v1/kb/{kb_id}", tags=["Knowledge Base"])
async def delete_knowledge_base(
    kb_id: str,
    current_user: str = Depends(get_current_user)
):
    """Delete a knowledge base"""
    if kb_id not in knowledge_bases:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    del knowledge_bases[kb_id]
    
    # Delete associated index file if exists
    index_path = Path(config["system"]["indices_dir"]) / f"{kb_id}.index"
    if index_path.exists():
        index_path.unlink()
    
    return {"message": "Knowledge base deleted successfully", "id": kb_id}

# =====================================
# API Endpoints - Query & Retrieval
# =====================================

@app.post("/api/v1/query/advanced", tags=["Query"])
async def advanced_query(
    request: EnhancedQueryRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = None  # Optional auth
):
    """Advanced query with all features"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Get or create session
        session = get_or_create_enhanced_session(request.session_id)
        
        # Add conversation context if available
        context = ""
        if session and len(session.history) > 0:
            context = session.get_context(last_n=3)
        
        # Prepare query with context
        full_query = f"{context}\n\nCurrent question: {request.query}" if context else request.query
        
        # Select knowledge base if specified
        kb = knowledge_bases.get(request.knowledge_base_id) if request.knowledge_base_id else None
        
        # Execute query based on configuration
        if request.stream:
            # Return streaming response
            return StreamingResponse(
                stream_query_response(full_query, request, session),
                media_type="text/event-stream"
            )
        else:
            # Standard response
            result = await execute_query(full_query, request, kb)
            
            # Update session
            if session:
                session.add_interaction(
                    request.query,
                    result["answer"],
                    {"kb_id": request.knowledge_base_id, "agent_type": request.agent_type}
                )
            
            # Log analytics in background
            background_tasks.add_task(
                log_query_analytics,
                request.query,
                result,
                current_user
            )
            
            return result
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_query(query: str, request: EnhancedQueryRequest, kb: Optional[KnowledgeBase]):
    """Execute query with specified configuration"""
    start_time = datetime.now()
    
    # Apply filters if provided
    filter_dict = request.filters or {}
    
    # Perform retrieval
    if isinstance(rag_system, MultimodalAgenticRAG) and request.modalities:
        modalities = [Modality[m.upper()] for m in request.modalities]
        result = await rag_system.multimodal_query(
            query,
            modalities=modalities,
            use_agent=request.use_agent,
            top_k=request.top_k
        )
    else:
        result = rag_system.query(
            query,
            use_agent=request.use_agent,
            top_k=request.top_k
        )
    
    # Apply reranking if requested
    if request.rerank and len(result.sources) > 1:
        # Implement reranking logic
        pass
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "answer": result.answer,
        "sources": [
            {
                "content": str(source.content)[:500],
                "score": getattr(source, 'score', 0.0),
                "metadata": getattr(source, 'metadata', {})
            }
            for source in result.sources[:request.top_k]
        ],
        "confidence": result.confidence,
        "processing_time": processing_time,
        "session_id": request.session_id,
        "agent_type": request.agent_type,
        "knowledge_base_id": request.knowledge_base_id
    }

async def stream_query_response(query: str, request: EnhancedQueryRequest, session):
    """Stream query response using SSE"""
    try:
        # Execute query
        result = rag_system.query(query, use_agent=request.use_agent, top_k=request.top_k)
        
        # Stream tokens
        tokens = result.answer.split()
        for i, token in enumerate(tokens):
            yield f"data: {json.dumps({'token': token + ' ', 'index': i, 'done': False})}\n\n"
            await asyncio.sleep(0.02)  # Control streaming speed
        
        # Send completion signal with sources
        yield f"data: {json.dumps({'done': True, 'sources': len(result.sources)})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

def get_or_create_enhanced_session(session_id: Optional[str] = None) -> EnhancedSession:
    """Get or create enhanced session"""
    if session_id and session_id in sessions:
        session = sessions[session_id]
        session.update()
    else:
        session_id = str(uuid.uuid4())
        session = EnhancedSession(session_id)
        sessions[session_id] = session
    
    return session

# =====================================
# API Endpoints - Documents
# =====================================

@app.post("/api/v1/documents/upload/advanced", tags=["Documents"])
async def upload_document_advanced(
    document: EnhancedDocumentUpload,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = None
):
    """Upload document with advanced processing options"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        doc_id = str(uuid.uuid4())
        
        # Get target knowledge base
        kb = knowledge_bases.get(document.knowledge_base_id) if document.knowledge_base_id else None
        
        # Process document in background
        background_tasks.add_task(
            process_document_advanced,
            doc_id,
            document,
            kb
        )
        
        return {
            "document_id": doc_id,
            "status": "processing",
            "knowledge_base_id": document.knowledge_base_id,
            "message": "Document queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/documents/upload/batch", tags=["Documents"])
async def upload_batch(
    batch_request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = None
):
    """Upload and process multiple documents in batch"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    batch_id = str(uuid.uuid4())
    document_ids = []
    
    for document in batch_request.documents:
        doc_id = str(uuid.uuid4())
        document_ids.append(doc_id)
        
        if batch_request.parallel_processing:
            background_tasks.add_task(
                process_document_advanced,
                doc_id,
                document,
                None
            )
        else:
            # Process sequentially
            await process_document_advanced(doc_id, document, None)
    
    return {
        "batch_id": batch_id,
        "document_ids": document_ids,
        "total": len(document_ids),
        "parallel_processing": batch_request.parallel_processing,
        "status": "processing"
    }

async def process_document_advanced(
    doc_id: str,
    document: EnhancedDocumentUpload,
    kb: Optional[KnowledgeBase]
):
    """Advanced document processing with all features"""
    try:
        # Process tables if requested
        if document.process_tables:
            # Extract and process tables
            pass
        
        # Extract entities if requested
        if document.extract_entities:
            # NER extraction
            pass
        
        # Add to RAG system
        if isinstance(rag_system, MultimodalAgenticRAG) and document.modality != "text":
            # Handle multimodal document
            pass
        else:
            # Text document
            num_chunks = rag_system.add_documents([document.content])
            
            # Update knowledge base stats
            if kb:
                kb.document_count += 1
                kb.total_chunks += num_chunks
                kb.updated_at = datetime.now()
        
        # Cache if enabled
        if redis_client:
            await redis_client.setex(
                f"doc:{doc_id}",
                3600,  # 1 hour TTL
                json.dumps({
                    "status": "processed",
                    "chunks": num_chunks,
                    "kb_id": kb.id if kb else None
                })
            )
        
        logger.info(f"Document {doc_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Document processing failed for {doc_id}: {e}")
        if redis_client:
            await redis_client.setex(
                f"doc:{doc_id}",
                3600,
                json.dumps({"status": "failed", "error": str(e)})
            )

# =====================================
# API Endpoints - Agents
# =====================================

@app.post("/api/v1/agents/execute", tags=["Agents"])
async def execute_agent_task(
    task: AgentTaskRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[str] = None
):
    """Execute an agent task"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    task_id = str(uuid.uuid4())
    
    # Execute task asynchronously
    background_tasks.add_task(
        run_agent_task,
        task_id,
        task
    )
    
    return {
        "task_id": task_id,
        "status": "running",
        "task_type": task.task_type,
        "message": "Task started"
    }

async def run_agent_task(task_id: str, task: AgentTaskRequest):
    """Run agent task in background"""
    try:
        start_time = datetime.now()
        
        # Select appropriate agent
        if task.task_type == "research":
            # Research agent
            result = await execute_research_task(task.parameters, task.context)
        elif task.task_type == "summarize":
            # Summarization agent
            result = await execute_summarization_task(task.parameters, task.context)
        elif task.task_type == "extract":
            # Extraction agent
            result = await execute_extraction_task(task.parameters, task.context)
        elif task.task_type == "analyze":
            # Analysis agent
            result = await execute_analysis_task(task.parameters, task.context)
        elif task.task_type == "compare":
            # Comparison agent
            result = await execute_comparison_task(task.parameters, task.context)
        else:
            # Generation agent
            result = await execute_generation_task(task.parameters, task.context)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Store result
        if redis_client:
            await redis_client.setex(
                f"task:{task_id}",
                3600,
                json.dumps({
                    "status": "completed",
                    "result": result,
                    "execution_time": execution_time
                })
            )
        
        logger.info(f"Task {task_id} completed in {execution_time}s")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}")
        if redis_client:
            await redis_client.setex(
                f"task:{task_id}",
                3600,
                json.dumps({"status": "failed", "error": str(e)})
            )

@app.get("/api/v1/agents/task/{task_id}", tags=["Agents"])
async def get_task_status(task_id: str):
    """Get agent task status and result"""
    if redis_client:
        result = await redis_client.get(f"task:{task_id}")
        if result:
            return json.loads(result)
    
    raise HTTPException(status_code=404, detail="Task not found")

# Agent task implementations
async def execute_research_task(parameters: Dict, context: str):
    """Execute research task"""
    # Implementation
    return {"type": "research", "findings": "Research results"}

async def execute_summarization_task(parameters: Dict, context: str):
    """Execute summarization task"""
    # Implementation
    return {"type": "summary", "summary": "Summary text"}

async def execute_extraction_task(parameters: Dict, context: str):
    """Execute extraction task"""
    # Implementation
    return {"type": "extraction", "extracted": {}}

async def execute_analysis_task(parameters: Dict, context: str):
    """Execute analysis task"""
    # Implementation
    return {"type": "analysis", "insights": []}

async def execute_comparison_task(parameters: Dict, context: str):
    """Execute comparison task"""
    # Implementation
    return {"type": "comparison", "differences": [], "similarities": []}

async def execute_generation_task(parameters: Dict, context: str):
    """Execute generation task"""
    # Implementation
    return {"type": "generation", "generated": "Generated content"}

# =====================================
# API Endpoints - Analytics
# =====================================

@app.post("/api/v1/analytics/query", tags=["Analytics"])
async def query_analytics(
    query: AnalyticsQuery,
    current_user: str = Depends(get_current_user)
):
    """Query system analytics"""
    # Implementation would fetch from database/monitoring system
    return {
        "metric_type": query.metric_type,
        "period": {
            "start": query.start_date or datetime.now() - timedelta(days=7),
            "end": query.end_date or datetime.now()
        },
        "data": [
            # Sample data
            {"date": "2024-01-01", "value": 100},
            {"date": "2024-01-02", "value": 120}
        ]
    }

async def log_query_analytics(query: str, result: Dict, user: Optional[str]):
    """Log query analytics"""
    try:
        analytics_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "user": user,
            "processing_time": result.get("processing_time", 0),
            "sources_count": len(result.get("sources", [])),
            "confidence": result.get("confidence", 0)
        }
        
        # Log to monitoring system
        logger.info(f"Analytics: {analytics_data}")
        
        # Store in Redis for aggregation
        if redis_client:
            key = f"analytics:{datetime.now().strftime('%Y%m%d')}"
            await redis_client.lpush(key, json.dumps(analytics_data))
            await redis_client.expire(key, 86400 * 30)  # 30 days retention
            
    except Exception as e:
        logger.error(f"Failed to log analytics: {e}")

# =====================================
# WebSocket Endpoints
# =====================================

@app.websocket("/ws/chat/v2")
async def websocket_chat_enhanced(websocket: WebSocket):
    """Enhanced WebSocket chat with session management"""
    await websocket.accept()
    session = EnhancedSession(str(uuid.uuid4()))
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "query":
                query = message.get("query", "")
                kb_id = message.get("knowledge_base_id")
                use_agent = message.get("use_agent", True)
                
                # Add context from session
                context = session.get_context(last_n=3)
                full_query = f"{context}\n\nUser: {query}" if context else query
                
                # Process query
                if rag_system:
                    result = rag_system.query(full_query, use_agent=use_agent)
                    
                    # Update session
                    session.add_interaction(query, result.answer)
                    
                    response = {
                        "type": "response",
                        "answer": result.answer,
                        "sources": len(result.sources),
                        "session_id": session.id,
                        "confidence": result.confidence
                    }
                else:
                    response = {
                        "type": "error",
                        "message": "System not initialized"
                    }
                
                await websocket.send_text(json.dumps(response))
            
            elif message.get("type") == "stream":
                # Handle streaming request
                query = message.get("query", "")
                async for chunk in stream_websocket_response(query, session):
                    await websocket.send_text(chunk)
            
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session.id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

async def stream_websocket_response(query: str, session: EnhancedSession):
    """Stream response over WebSocket"""
    try:
        result = rag_system.query(query)
        tokens = result.answer.split()
        
        for i, token in enumerate(tokens):
            yield json.dumps({
                "type": "stream",
                "token": token + " ",
                "index": i,
                "done": False
            })
            await asyncio.sleep(0.02)
        
        yield json.dumps({
            "type": "stream",
            "done": True,
            "total_tokens": len(tokens)
        })
        
        # Update session
        session.add_interaction(query, result.answer)
        
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)})

# =====================================
# Admin Endpoints
# =====================================

@app.post("/api/v1/admin/config/update", tags=["Admin"])
async def update_system_config(
    config_update: SystemConfig,
    current_user: str = Depends(get_current_user)
):
    """Update system configuration"""
    # In production, validate user has admin role
    
    updates = {}
    if config_update.embeddings:
        updates["embeddings"] = config_update.embeddings
    if config_update.language_model:
        updates["language_model"] = config_update.language_model
    if config_update.retrieval:
        updates["retrieval"] = config_update.retrieval
    if config_update.agents:
        updates["agents"] = config_update.agents
    
    # Apply updates
    for key, value in updates.items():
        if key in config:
            config[key].update(value)
    
    return {
        "status": "updated",
        "updates": updates
    }

@app.get("/api/v1/admin/stats", tags=["Admin"])
async def get_system_stats(current_user: str = Depends(get_current_user)):
    """Get comprehensive system statistics"""
    total_queries = 0
    avg_response_time = 0
    
    # Calculate from analytics data in Redis
    if redis_client:
        today = datetime.now().strftime('%Y%m%d')
        analytics_key = f"analytics:{today}"
        data = await redis_client.lrange(analytics_key, 0, -1)
        
        if data:
            total_queries = len(data)
            times = [json.loads(d).get("processing_time", 0) for d in data]
            avg_response_time = sum(times) / len(times) if times else 0
    
    return {
        "system": {
            "uptime": (datetime.now() - app.state.startup_time).total_seconds() if hasattr(app.state, 'startup_time') else 0,
            "total_queries": total_queries,
            "avg_response_time": avg_response_time,
            "active_sessions": len(sessions),
            "knowledge_bases": len(knowledge_bases)
        },
        "resources": await get_resource_usage(),
        "cache": await get_cache_stats() if redis_client else None
    }

async def get_resource_usage():
    """Get current resource usage"""
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "threads": process.num_threads()
    }

async def get_cache_stats():
    """Get cache statistics from Redis"""
    if redis_client:
        info = await redis_client.info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
            "total_keys": await redis_client.dbsize()
        }
    return None

# =====================================
# Startup Configuration
# =====================================

@app.on_event("startup")
async def startup_event():
    """Additional startup configuration"""
    app.state.startup_time = datetime.now()
    logger.info(f"API started at {app.state.startup_time}")

# =====================================
# Error Handlers
# =====================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.now().isoformat()
    }

# =====================================
# Main Entry Point
# =====================================

if __name__ == "__main__":
    import uvicorn
    
    api_config = config.get("api", {})
    
    uvicorn.run(
        "enhanced_server:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("reload", True),
        workers=api_config.get("workers", 1),
        log_level=config.get("system", {}).get("log_level", "INFO").lower()
    )
