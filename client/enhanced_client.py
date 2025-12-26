"""
Python Client SDK for Agentic RAG System API
"""

import asyncio
import aiohttp
import requests
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
import json
from pathlib import Path
import base64
from datetime import datetime
import websockets
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Agent task types"""
    RESEARCH = "research"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"
    ANALYZE = "analyze"
    COMPARE = "compare"
    GENERATE = "generate"

class Modality(Enum):
    """Document modalities"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TABLE = "table"
    CODE = "code"

@dataclass
class QueryResult:
    """Query result container"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    session_id: str
    agent_type: Optional[str] = None
    knowledge_base_id: Optional[str] = None

@dataclass
class DocumentUploadResult:
    """Document upload result"""
    document_id: str
    status: str
    knowledge_base_id: Optional[str] = None
    message: Optional[str] = None

@dataclass
class KnowledgeBase:
    """Knowledge base information"""
    id: str
    name: str
    document_count: int
    total_chunks: int
    created_at: datetime
    updated_at: datetime

@dataclass
class AgentTask:
    """Agent task information"""
    task_id: str
    status: str
    task_type: str
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None

class AgenticRAGClient:
    """Synchronous client for Agentic RAG System API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set headers
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })
    
    # =====================================
    # Authentication
    # =====================================
    
    def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user"""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def login(self, username: str, password: str) -> str:
        """Login and get access token"""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            params={"username": username, "password": password},
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # Update session with token
        self.api_key = data["access_token"]
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}"
        })
        
        return data["access_token"]
    
    # =====================================
    # Health & Status
    # =====================================
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health"""
        response = self.session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information"""
        response = self.session.get(
            f"{self.base_url}/",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =====================================
    # Knowledge Base Management
    # =====================================
    
    def create_knowledge_base(
        self,
        name: str,
        description: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_type: str = "faiss",
        config: Optional[Dict[str, Any]] = None
    ) -> KnowledgeBase:
        """Create a new knowledge base"""
        response = self.session.post(
            f"{self.base_url}/api/v1/kb/create",
            json={
                "name": name,
                "description": description,
                "embedding_model": embedding_model,
                "vector_store_type": vector_store_type,
                "config": config or {}
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return KnowledgeBase(
            id=data["id"],
            name=data["name"],
            document_count=0,
            total_chunks=0,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["created_at"])
        )
    
    def list_knowledge_bases(self) -> List[KnowledgeBase]:
        """List all knowledge bases"""
        response = self.session.get(
            f"{self.base_url}/api/v1/kb/list",
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return [
            KnowledgeBase(
                id=kb["id"],
                name=kb["name"],
                document_count=kb["document_count"],
                total_chunks=kb["total_chunks"],
                created_at=datetime.fromisoformat(kb["created_at"]),
                updated_at=datetime.fromisoformat(kb["updated_at"])
            )
            for kb in data
        ]
    
    def delete_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """Delete a knowledge base"""
        response = self.session.delete(
            f"{self.base_url}/api/v1/kb/{kb_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =====================================
    # Query & Retrieval
    # =====================================
    
    def query(
        self,
        query: str,
        knowledge_base_id: Optional[str] = None,
        use_agent: bool = True,
        agent_type: str = "reasoning",
        top_k: int = 5,
        rerank: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        modalities: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> QueryResult:
        """Execute an advanced query"""
        response = self.session.post(
            f"{self.base_url}/api/v1/query/advanced",
            json={
                "query": query,
                "knowledge_base_id": knowledge_base_id,
                "use_agent": use_agent,
                "agent_type": agent_type,
                "top_k": top_k,
                "rerank": rerank,
                "filters": filters,
                "modalities": modalities,
                "session_id": session_id,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            timeout=self.timeout * 2  # Longer timeout for queries
        )
        response.raise_for_status()
        data = response.json()
        
        return QueryResult(
            answer=data["answer"],
            sources=data["sources"],
            confidence=data["confidence"],
            processing_time=data["processing_time"],
            session_id=data["session_id"],
            agent_type=data.get("agent_type"),
            knowledge_base_id=data.get("knowledge_base_id")
        )
    
    def stream_query(
        self,
        query: str,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream query response"""
        params = {
            "query": query,
            "stream": True,
            **kwargs
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/query/advanced",
            json=params,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if data.get("token"):
                        yield data["token"]
                    elif data.get("done"):
                        break
    
    # =====================================
    # Document Management
    # =====================================
    
    def upload_document(
        self,
        content: str,
        knowledge_base_id: Optional[str] = None,
        modality: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        process_tables: bool = False,
        extract_entities: bool = False
    ) -> DocumentUploadResult:
        """Upload a document with advanced options"""
        response = self.session.post(
            f"{self.base_url}/api/v1/documents/upload/advanced",
            json={
                "content": content,
                "knowledge_base_id": knowledge_base_id,
                "modality": modality,
                "metadata": metadata,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "process_tables": process_tables,
                "extract_entities": extract_entities
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return DocumentUploadResult(
            document_id=data["document_id"],
            status=data["status"],
            knowledge_base_id=data.get("knowledge_base_id"),
            message=data.get("message")
        )
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        knowledge_base_id: Optional[str] = None,
        **processing_options
    ) -> DocumentUploadResult:
        """Upload a file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        # Read file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Determine modality from extension
        ext = file_path.suffix.lower()
        if ext in ['.txt', '.md', '.pdf', '.docx']:
            modality = "text"
            content = content.decode('utf-8', errors='ignore')
        elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
            modality = "image"
            content = base64.b64encode(content).decode('utf-8')
        elif ext in ['.mp3', '.wav', '.flac']:
            modality = "audio"
            content = base64.b64encode(content).decode('utf-8')
        elif ext in ['.mp4', '.avi', '.mov']:
            modality = "video"
            content = base64.b64encode(content).decode('utf-8')
        else:
            modality = "text"
            content = content.decode('utf-8', errors='ignore')
        
        return self.upload_document(
            content=content,
            knowledge_base_id=knowledge_base_id,
            modality=modality,
            metadata={"filename": file_path.name},
            **processing_options
        )
    
    def upload_batch(
        self,
        documents: List[Dict[str, Any]],
        parallel_processing: bool = True,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """Upload multiple documents in batch"""
        response = self.session.post(
            f"{self.base_url}/api/v1/documents/upload/batch",
            json={
                "documents": documents,
                "parallel_processing": parallel_processing,
                "batch_size": batch_size
            },
            timeout=self.timeout * 3  # Longer timeout for batch
        )
        response.raise_for_status()
        return response.json()
    
    # =====================================
    # Agent Operations
    # =====================================
    
    def execute_agent_task(
        self,
        task_type: TaskType,
        parameters: Dict[str, Any],
        context: Optional[str] = None,
        knowledge_base_id: Optional[str] = None,
        timeout: int = 30
    ) -> str:
        """Execute an agent task"""
        response = self.session.post(
            f"{self.base_url}/api/v1/agents/execute",
            json={
                "task_type": task_type.value,
                "parameters": parameters,
                "context": context,
                "knowledge_base_id": knowledge_base_id,
                "timeout": timeout
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return data["task_id"]
    
    def get_task_result(self, task_id: str) -> AgentTask:
        """Get agent task result"""
        response = self.session.get(
            f"{self.base_url}/api/v1/agents/task/{task_id}",
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return AgentTask(
            task_id=task_id,
            status=data["status"],
            task_type=data.get("task_type", "unknown"),
            result=data.get("result"),
            execution_time=data.get("execution_time"),
            error=data.get("error")
        )
    
    def wait_for_task(
        self,
        task_id: str,
        poll_interval: int = 2,
        max_wait: int = 300
    ) -> AgentTask:
        """Wait for agent task to complete"""
        import time
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            result = self.get_task_result(task_id)
            if result.status in ["completed", "failed"]:
                return result
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {max_wait} seconds")
    
    # =====================================
    # Analytics
    # =====================================
    
    def get_analytics(
        self,
        metric_type: str = "usage",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """Query analytics"""
        params = {
            "metric_type": metric_type,
            "group_by": group_by
        }
        
        if start_date:
            params["start_date"] = start_date.isoformat()
        if end_date:
            params["end_date"] = end_date.isoformat()
        
        response = self.session.post(
            f"{self.base_url}/api/v1/analytics/query",
            json=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # =====================================
    # Admin Operations
    # =====================================
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        response = self.session.get(
            f"{self.base_url}/api/v1/admin/stats",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration"""
        response = self.session.post(
            f"{self.base_url}/api/v1/admin/config/update",
            json=config,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()


class AsyncAgenticRAGClient:
    """Asynchronous client for Agentic RAG System API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def query(
        self,
        query: str,
        **kwargs
    ) -> QueryResult:
        """Execute an advanced query asynchronously"""
        async with self.session.post(
            f"{self.base_url}/api/v1/query/advanced",
            json={
                "query": query,
                "stream": False,
                **kwargs
            }
        ) as response:
            response.raise_for_status()
            data = await response.json()
            
            return QueryResult(
                answer=data["answer"],
                sources=data["sources"],
                confidence=data["confidence"],
                processing_time=data["processing_time"],
                session_id=data["session_id"],
                agent_type=data.get("agent_type"),
                knowledge_base_id=data.get("knowledge_base_id")
            )
    
    async def stream_query(
        self,
        query: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream query response asynchronously"""
        async with self.session.post(
            f"{self.base_url}/api/v1/query/advanced",
            json={
                "query": query,
                "stream": True,
                **kwargs
            }
        ) as response:
            response.raise_for_status()
            
            async for line in response.content:
                if line:
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        if data.get("token"):
                            yield data["token"]
                        elif data.get("done"):
                            break


class WebSocketClient:
    """WebSocket client for real-time chat"""
    
    def __init__(self, url: str = "ws://localhost:8000/ws/chat/v2"):
        self.url = url
        self.websocket = None
    
    async def connect(self):
        """Connect to WebSocket"""
        self.websocket = await websockets.connect(self.url)
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
    
    async def send_query(
        self,
        query: str,
        knowledge_base_id: Optional[str] = None,
        use_agent: bool = True
    ) -> Dict[str, Any]:
        """Send query and get response"""
        await self.websocket.send(json.dumps({
            "type": "query",
            "query": query,
            "knowledge_base_id": knowledge_base_id,
            "use_agent": use_agent
        }))
        
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def stream_query(
        self,
        query: str
    ) -> AsyncGenerator[str, None]:
        """Stream query response"""
        await self.websocket.send(json.dumps({
            "type": "stream",
            "query": query
        }))
        
        while True:
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "stream":
                if data.get("token"):
                    yield data["token"]
                elif data.get("done"):
                    break
            elif data.get("type") == "error":
                raise Exception(data.get("message"))


# =====================================
# Convenience Functions
# =====================================

def create_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None
) -> AgenticRAGClient:
    """Create a synchronous client"""
    return AgenticRAGClient(base_url, api_key)

def create_async_client(
    base_url: str = "http://localhost:8000",
    api_key: Optional[str] = None
) -> AsyncAgenticRAGClient:
    """Create an asynchronous client"""
    return AsyncAgenticRAGClient(base_url, api_key)

def create_websocket_client(
    url: str = "ws://localhost:8000/ws/chat/v2"
) -> WebSocketClient:
    """Create a WebSocket client"""
    return WebSocketClient(url)


# =====================================
# Example Usage
# =====================================

if __name__ == "__main__":
    # Example: Synchronous usage
    client = create_client()
    
    # Check health
    health = client.health_check()
    print(f"System health: {health['status']}")
    
    # Create knowledge base
    kb = client.create_knowledge_base(
        name="Technical Documentation",
        description="Company technical docs"
    )
    print(f"Created knowledge base: {kb.id}")
    
    # Upload document
    doc_result = client.upload_document(
        content="This is a sample document about AI and machine learning.",
        knowledge_base_id=kb.id,
        extract_entities=True
    )
    print(f"Uploaded document: {doc_result.document_id}")
    
    # Query
    result = client.query(
        query="What do you know about AI?",
        knowledge_base_id=kb.id,
        use_agent=True
    )
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    
    # Execute agent task
    task_id = client.execute_agent_task(
        task_type=TaskType.RESEARCH,
        parameters={"topic": "machine learning"},
        context="Focus on recent developments"
    )
    print(f"Started task: {task_id}")
    
    # Wait for result
    task_result = client.wait_for_task(task_id)
    print(f"Task completed: {task_result.result}")
    
    # Example: Asynchronous usage
    async def async_example():
        async with create_async_client() as client:
            result = await client.query(
                query="Explain RAG systems",
                use_agent=True
            )
            print(f"Async answer: {result.answer}")
            
            # Stream response
            print("Streaming response:")
            async for token in client.stream_query("Tell me about agents"):
                print(token, end="", flush=True)
            print()
    
    # Run async example
    # asyncio.run(async_example())
    
    # Example: WebSocket usage
    async def websocket_example():
        ws_client = create_websocket_client()
        await ws_client.connect()
        
        try:
            # Send query
            response = await ws_client.send_query(
                query="Hello, how are you?",
                use_agent=True
            )
            print(f"WebSocket response: {response}")
            
            # Stream response
            print("Streaming via WebSocket:")
            async for token in ws_client.stream_query("Explain WebSockets"):
                print(token, end="", flush=True)
            print()
            
        finally:
            await ws_client.disconnect()
    
    # Run WebSocket example
    # asyncio.run(websocket_example())
