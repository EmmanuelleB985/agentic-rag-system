"""
Core Agentic RAG Implementation
Original functionality with enhancements
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Standard document container"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """Result from retrieval"""
    answer: str
    sources: List[Document]
    confidence: float
    processing_time: Optional[float] = None


class Embedder:
    """Handles text embedding generation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.device = device
        self.model = SentenceTransformer(model_name).to(device)
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        return self.model.encode(texts, convert_to_numpy=True)


class Retriever:
    """Handles document retrieval"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.documents = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to index"""
        if not documents:
            return
        
        embeddings = np.array([doc.embeddings for doc in documents])
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Document]:
        """Search for similar documents"""
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            min(top_k, len(self.documents))
        )
        
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results


class Agent:
    """Basic agent for reasoning"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", 
                 device: str = "cuda", quantization: str = "4bit"):
        self.device = device
        self.model_name = model_name
        self._initialize_model(quantization)
    
    def _initialize_model(self, quantization: str):
        """Initialize the language model"""
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, context: List[Document], 
                 max_tokens: int = 512) -> str:
        """Generate response based on prompt and context"""
        # Build context string
        context_str = "\n\n".join([doc.content[:500] for doc in context[:3]])
        
        full_prompt = f"""Context:
{context_str}

Query: {prompt}

Answer based on the context provided:"""
        
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        if "Answer based on the context provided:" in response:
            response = response.split("Answer based on the context provided:")[-1].strip()
        
        return response


class AgenticRAG:
    """Main Agentic RAG System"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
        quantization: str = "4bit",
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.device = device
        self.model_name = model_name
        
        logger.info("Initializing Agentic RAG System...")
        
        # Initialize components
        self.embedder = Embedder(embedding_model, device)
        self.retriever = Retriever(embedding_dim=384)
        self.agent = Agent(model_name, device, quantization)
        
        # Text processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info("Agentic RAG System initialized successfully!")
    
    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to the system"""
        if metadata is None:
            metadata = [{} for _ in texts]
        
        documents = []
        for i, text in enumerate(texts):
            # Split into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Generate embeddings
            embeddings = self.embedder.embed(chunks)
            
            # Create documents
            for j, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = Document(
                    id=f"doc_{i}_chunk_{j}",
                    content=chunk,
                    metadata={**metadata[i], "chunk_index": j},
                    embeddings=embedding
                )
                documents.append(doc)
        
        # Add to retriever
        self.retriever.add_documents(documents)
        
        return len(documents)
    
    def query(
        self,
        query: str,
        use_agent: bool = True,
        top_k: int = 5
    ) -> RetrievalResult:
        """Process a query"""
        import time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedder.embed([query])[0]
        
        # Retrieve relevant documents
        retrieved_docs = self.retriever.search(query_embedding, top_k)
        
        # Generate response
        if use_agent:
            answer = self.agent.generate(query, retrieved_docs)
        else:
            # Simple concatenation of retrieved content
            answer = "\n\n".join([doc.content for doc in retrieved_docs[:3]])
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            answer=answer,
            sources=retrieved_docs,
            confidence=0.85,  # Would be calculated based on retrieval scores
            processing_time=processing_time
        )
    
    def save_index(self, path: str):
        """Save the retrieval index"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.retriever.index, str(index_path))
        
        # Save documents
        import pickle
        with open(save_dir / "documents.pkl", "wb") as f:
            pickle.dump(self.retriever.documents, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load a saved index"""
        load_dir = Path(path)
        
        # Load FAISS index
        index_path = load_dir / "faiss_index.bin"
        if index_path.exists():
            self.retriever.index = faiss.read_index(str(index_path))
        
        # Load documents
        import pickle
        with open(load_dir / "documents.pkl", "rb") as f:
            self.retriever.documents = pickle.load(f)
        
        logger.info(f"Index loaded from {path}")
