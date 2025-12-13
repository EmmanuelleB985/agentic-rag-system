"""
Core RAG Pipeline implementation with agentic capabilities.
"""

import os
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Response object from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    reasoning_trace: List[Dict[str, Any]]
    confidence: float
    tokens_used: int
    retrieval_time: float
    generation_time: float


class AgenticRAG:
    """
    Main RAG pipeline with agentic decision-making capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        quantization: str = "4bit",
        index_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            model_name: Name of the language model to use
            embedding_model: Name of the embedding model
            device: Device to run models on
            quantization: Quantization level (none, 8bit, 4bit)
            index_path: Path to pre-built FAISS index
            config: Additional configuration parameters
        """
        self.device = device
        self.config = config or {}
        
        logger.info(f"Initializing AgenticRAG on {device}")
        
        # Initialize models
        self._initialize_llm(model_name, quantization)
        self._initialize_embeddings(embedding_model)
        
        # Initialize retrieval components
        self.documents = []
        self.index = None
        self.chunk_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50)
        )
        
        # Load index if provided
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
        
        # Initialize agent
        from ..agents.rag_agent import RAGAgent
        self.agent = RAGAgent(self)
    
    def _initialize_llm(self, model_name: str, quantization: str):
        """Initialize the language model with optional quantization."""
        logger.info(f"Loading LLM: {model_name} with {quantization} quantization")
        
        # Configure quantization
        bnb_config = None
        if quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16 if quantization else torch.float32,
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable()
    
    def _initialize_embeddings(self, embedding_model: str):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
    
    def load_dataset(self, dataset_name: str, sample_size: Optional[int] = None):
        """
        Load a public dataset for RAG.
        
        Args:
            dataset_name: Name of the dataset to load
            sample_size: Number of documents to sample (None for all)
        """
        from datasets import load_dataset
        
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Load based on dataset type
        if dataset_name.startswith("wikipedia"):
            dataset = load_dataset("wikipedia", "20220301.en", split="train")
            texts = dataset["text"][:sample_size] if sample_size else dataset["text"]
            
        elif dataset_name == "ms_marco":
            dataset = load_dataset("ms_marco", "v2.1", split="train")
            texts = dataset["passages"][:sample_size] if sample_size else dataset["passages"]
            
        elif dataset_name == "natural_questions":
            dataset = load_dataset("natural_questions", split="train")
            texts = [d["document"]["text"] for d in dataset[:sample_size]] if sample_size else [d["document"]["text"] for d in dataset]
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Process and index documents
        self.add_documents(texts)
        logger.info(f"Loaded {len(self.documents)} documents from {dataset_name}")
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        logger.info(f"Adding {len(documents)} documents")
        
        # Chunk documents
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = self.chunk_splitter.split_text(doc)
            for j, chunk in enumerate(chunks):
                chunk_metadata = {
                    "doc_id": i,
                    "chunk_id": j,
                    "text": chunk
                }
                if metadata and i < len(metadata):
                    chunk_metadata.update(metadata[i])
                all_chunks.append(chunk_metadata)
        
        # Store documents
        self.documents.extend(all_chunks)
        
        # Create embeddings and build index
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from documents."""
        logger.info("Building FAISS index")
        
        # Extract texts
        texts = [doc["text"] for doc in self.documents]
        
        # Create embeddings (batch for efficiency)
        batch_size = 32
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embedder.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        
        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add vectors to index
        self.index.add(embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
        
        Returns:
            List of retrieved documents with scores
        """
        if not self.index:
            logger.warning("No index available for retrieval")
            return []
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc["score"] = float(1 / (1 + dist))  # Convert distance to similarity
                results.append(doc)
        
        return results
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate text using the language model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=self.config.get("temperature", 0.7),
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_text
    
    def query(
        self,
        query: str,
        use_agent: bool = True,
        top_k: int = 5,
        return_sources: bool = True
    ) -> RAGResponse:
        """
        Main query interface for the RAG system.
        
        Args:
            query: User query
            use_agent: Whether to use agentic reasoning
            top_k: Number of documents to retrieve
            return_sources: Whether to return source documents
        
        Returns:
            RAGResponse object with answer and metadata
        """
        import time
        
        start_time = time.time()
        
        if use_agent:
            # Use agent for complex reasoning
            result = self.agent.run(query)
            return RAGResponse(
                answer=result["answer"],
                sources=result.get("sources", []),
                reasoning_trace=result.get("reasoning_trace", []),
                confidence=result.get("confidence", 0.0),
                tokens_used=result.get("tokens_used", 0),
                retrieval_time=result.get("retrieval_time", 0.0),
                generation_time=time.time() - start_time
            )
        else:
            # Simple RAG without agent
            retrieval_start = time.time()
            retrieved_docs = self.retrieve(query, top_k=top_k)
            retrieval_time = time.time() - retrieval_start
            
            # Build context
            context = "\n\n".join([doc["text"] for doc in retrieved_docs[:3]])
            
            # Generate prompt
            prompt = f"""Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate answer
            answer = self.generate(prompt)
            
            return RAGResponse(
                answer=answer,
                sources=retrieved_docs if return_sources else [],
                reasoning_trace=[],
                confidence=0.8,
                tokens_used=len(self.tokenizer.encode(prompt + answer)),
                retrieval_time=retrieval_time,
                generation_time=time.time() - start_time
            )
    
    def save_index(self, path: str):
        """Save FAISS index and documents to disk."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save index
        if self.index:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save documents
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(self.documents, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index and documents from disk."""
        # Load index
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        # Load documents
        docs_path = os.path.join(path, "documents.json")
        if os.path.exists(docs_path):
            with open(docs_path, "r") as f:
                self.documents = json.load(f)
        
        logger.info(f"Index loaded from {path}")
    
    def stream_query(self, query: str, **kwargs):
        """
        Stream tokens as they're generated.
        
        Args:
            query: User query
            **kwargs: Additional arguments for query()
        
        Yields:
            Generated tokens
        """
        # Simplified streaming implementation
        response = self.query(query, **kwargs)
        
        # Simulate streaming by yielding words
        words = response.answer.split()
        for word in words:
            yield word + " "
