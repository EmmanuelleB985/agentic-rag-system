"""
Advanced retrieval components for the RAG system
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """Vector storage and retrieval"""
    
    def __init__(self, dimension: int = 384, index_type: str = "Flat"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = self._create_index()
        self.documents = []
        self.metadata = []
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index"""
        if self.index_type == "Flat":
            return faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            return faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            return faiss.IndexFlatIP(self.dimension)
    
    def add(self, embeddings: np.ndarray, documents: List[str], 
            metadata: Optional[List[Dict]] = None):
        """Add documents to the store"""
        if metadata is None:
            metadata = [{} for _ in documents]
        
        # Train index if needed
        if self.index_type == "IVF" and not self.index.is_trained:
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[RetrievalResult]:
        """Search for similar documents"""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append(RetrievalResult(
                    document_id=str(idx),
                    content=self.documents[idx],
                    score=float(dist),
                    metadata=self.metadata[idx]
                ))
        
        return results
    
    def save(self, path: str):
        """Save the index"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        
        import pickle
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump((self.documents, self.metadata), f)
    
    def load(self, path: str):
        """Load the index"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        
        import pickle
        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents, self.metadata = pickle.load(f)


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda"
    ):
        self.device = device
        
        # Dense retrieval
        self.embedding_model = SentenceTransformer(embedding_model).to(device)
        self.vector_store = VectorStore(dimension=384)
        
        # Sparse retrieval
        self.bm25 = None
        self.corpus = []
        
        # Reranking
        self.reranker = CrossEncoder(rerank_model, device=device)
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """Add documents to both dense and sparse indices"""
        # Dense indexing
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True)
        self.vector_store.add(embeddings, documents, metadata)
        
        # Sparse indexing
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.corpus = documents
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(
        self,
        query: str,
        k: int = 10,
        rerank_top_k: int = 5,
        alpha: float = 0.5
    ) -> List[RetrievalResult]:
        """Hybrid search with reranking"""
        # Dense search
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        dense_results = self.vector_store.search(query_embedding[0], k)
        
        # Sparse search
        sparse_results = []
        if self.bm25:
            query_tokens = query.lower().split()
            sparse_scores = self.bm25.get_scores(query_tokens)
            top_indices = np.argsort(sparse_scores)[-k:][::-1]
            
            for idx in top_indices:
                if idx < len(self.corpus):
                    sparse_results.append(RetrievalResult(
                        document_id=str(idx),
                        content=self.corpus[idx],
                        score=float(sparse_scores[idx]),
                        metadata={}
                    ))
        
        # Combine results
        combined_results = self._combine_results(dense_results, sparse_results, alpha)
        
        # Rerank top results
        if len(combined_results) > 0:
            reranked = self._rerank(query, combined_results[:rerank_top_k * 2])
            return reranked[:rerank_top_k]
        
        return combined_results[:rerank_top_k]
    
    def _combine_results(
        self,
        dense: List[RetrievalResult],
        sparse: List[RetrievalResult],
        alpha: float
    ) -> List[RetrievalResult]:
        """Combine dense and sparse results"""
        # Create score dictionaries
        dense_scores = {r.content: r.score for r in dense}
        sparse_scores = {r.content: r.score for r in sparse}
        
        # Get all unique documents
        all_docs = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Combine scores
        combined = []
        for doc in all_docs:
            dense_score = dense_scores.get(doc, 0)
            sparse_score = sparse_scores.get(doc, 0)
            
            # Normalize scores
            if len(dense_scores) > 0:
                max_dense = max(dense_scores.values())
                dense_score = dense_score / max_dense if max_dense > 0 else 0
            
            if len(sparse_scores) > 0:
                max_sparse = max(sparse_scores.values())
                sparse_score = sparse_score / max_sparse if max_sparse > 0 else 0
            
            # Combined score
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score
            
            # Find metadata
            metadata = {}
            for r in dense + sparse:
                if r.content == doc:
                    metadata = r.metadata
                    break
            
            combined.append(RetrievalResult(
                document_id=f"combined_{len(combined)}",
                content=doc,
                score=combined_score,
                metadata=metadata
            ))
        
        # Sort by combined score
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined
    
    def _rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        if not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [[query, r.content] for r in results]
        
        # Get reranking scores
        scores = self.reranker.predict(pairs)
        
        # Update scores and sort
        for i, result in enumerate(results):
            result.score = float(scores[i])
        
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results


class SemanticChunker:
    """Semantic-based text chunking"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", device: str = "cuda"):
        self.embedding_model = SentenceTransformer(embedding_model).to(device)
        self.device = device
    
    def chunk(
        self,
        text: str,
        max_chunk_size: int = 512,
        similarity_threshold: float = 0.7
    ) -> List[str]:
        """Create semantic chunks"""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if not sentences:
            return []
        
        # Get embeddings
        embeddings = self.embedding_model.encode(sentences, convert_to_numpy=True)
        
        # Create chunks based on semantic similarity
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = embeddings[0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity
            similarity = np.dot(current_embedding, embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(embeddings[i])
            )
            
            # Check if should add to current chunk
            chunk_text = " ".join(current_chunk + [sentences[i]])
            
            if similarity >= similarity_threshold and len(chunk_text) <= max_chunk_size:
                current_chunk.append(sentences[i])
                # Update embedding (average)
                current_embedding = np.mean(
                    embeddings[i - len(current_chunk) + 1:i + 1],
                    axis=0
                )
            else:
                # Save current chunk and start new one
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
                current_embedding = embeddings[i]
        
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences


class GraphRetriever:
    """Graph-based retrieval for connected information"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.nodes = {}  # id -> content
        self.edges = {}  # id -> list of connected ids
        self.embeddings = {}  # id -> embedding
    
    def add_node(self, node_id: str, content: str, embedding: np.ndarray):
        """Add a node to the graph"""
        self.nodes[node_id] = content
        self.embeddings[node_id] = embedding
        if node_id not in self.edges:
            self.edges[node_id] = []
    
    def add_edge(self, node1: str, node2: str, weight: float = 1.0):
        """Add an edge between nodes"""
        if node1 not in self.edges:
            self.edges[node1] = []
        if node2 not in self.edges:
            self.edges[node2] = []
        
        self.edges[node1].append((node2, weight))
        self.edges[node2].append((node1, weight))
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        max_hops: int = 2
    ) -> List[Tuple[str, str, float]]:
        """Search with graph traversal"""
        # Find initial nodes
        similarities = {}
        for node_id, embedding in self.embeddings.items():
            sim = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities[node_id] = sim
        
        # Get top k initial nodes
        top_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Expand with graph traversal
        expanded = set()
        results = []
        
        for node_id, score in top_nodes:
            results.append((node_id, self.nodes[node_id], score))
            expanded.add(node_id)
            
            # Traverse neighbors
            self._traverse(node_id, query_embedding, max_hops, 1, expanded, results, score)
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:k]
    
    def _traverse(
        self,
        node_id: str,
        query_embedding: np.ndarray,
        max_hops: int,
        current_hop: int,
        expanded: set,
        results: list,
        parent_score: float
    ):
        """Traverse graph from a node"""
        if current_hop > max_hops:
            return
        
        for neighbor_id, edge_weight in self.edges.get(node_id, []):
            if neighbor_id not in expanded:
                # Calculate score
                sim = np.dot(query_embedding, self.embeddings[neighbor_id]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings[neighbor_id])
                )
                
                # Decay score based on hop distance
                score = sim * edge_weight * (0.8 ** current_hop)
                
                results.append((neighbor_id, self.nodes[neighbor_id], score))
                expanded.add(neighbor_id)
                
                # Recursive traversal
                self._traverse(
                    neighbor_id, query_embedding, max_hops,
                    current_hop + 1, expanded, results, score
                )
