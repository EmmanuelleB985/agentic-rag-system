"""
Tests for core RAG functionality
"""

import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from agentic_rag import AgenticRAG
from agentic_rag.core import Document, Embedder, Retriever, Agent


class TestEmbedder:
    """Test embedding functionality"""
    
    def test_initialization(self):
        """Test embedder initialization"""
        embedder = Embedder(device="cpu")
        assert embedder is not None
        assert embedder.device == "cpu"
    
    def test_embed_text(self):
        """Test text embedding"""
        embedder = Embedder(device="cpu")
        texts = ["Hello world", "Test embedding"]
        embeddings = embedder.embed(texts)
        
        assert embeddings is not None
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # Default dimension
    
    def test_empty_text(self):
        """Test empty text handling"""
        embedder = Embedder(device="cpu")
        embeddings = embedder.embed([""])
        assert embeddings.shape[0] == 1


class TestRetriever:
    """Test retrieval functionality"""
    
    def test_initialization(self):
        """Test retriever initialization"""
        retriever = Retriever(embedding_dim=384)
        assert retriever is not None
        assert retriever.embedding_dim == 384
        assert len(retriever.documents) == 0
    
    def test_add_documents(self):
        """Test adding documents"""
        retriever = Retriever()
        
        docs = [
            Document(
                id="1",
                content="Test document",
                metadata={},
                embeddings=np.random.rand(384).astype('float32')
            )
        ]
        
        retriever.add_documents(docs)
        assert len(retriever.documents) == 1
    
    def test_search(self):
        """Test document search"""
        retriever = Retriever()
        
        # Add documents
        docs = []
        for i in range(5):
            doc = Document(
                id=str(i),
                content=f"Document {i}",
                metadata={"index": i},
                embeddings=np.random.rand(384).astype('float32')
            )
            docs.append(doc)
        
        retriever.add_documents(docs)
        
        # Search
        query_embedding = np.random.rand(384).astype('float32')
        results = retriever.search(query_embedding, top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, Document) for r in results)


class TestAgenticRAG:
    """Test main RAG system"""
    
    def test_initialization(self):
        """Test RAG initialization"""
        rag = AgenticRAG(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cpu",
            quantization="4bit"
        )
        assert rag is not None
    
    def test_add_documents(self):
        """Test adding documents to RAG"""
        rag = AgenticRAG(device="cpu")
        
        texts = ["Document 1", "Document 2"]
        metadata = [{"source": "test1"}, {"source": "test2"}]
        
        num_chunks = rag.add_documents(texts, metadata)
        assert num_chunks > 0
    
    @pytest.mark.skipif(not Path("./models").exists(), reason="Models not downloaded")
    def test_query(self):
        """Test querying the RAG system"""
        rag = AgenticRAG(device="cpu")
        
        # Add documents
        rag.add_documents(["Machine learning is a subset of AI"])
        
        # Query
        result = rag.query("What is machine learning?", use_agent=False)
        
        assert result is not None
        assert result.answer is not None
        assert len(result.sources) > 0
    
    def test_save_load_index(self, tmp_path):
        """Test saving and loading index"""
        rag = AgenticRAG(device="cpu")
        
        # Add documents
        rag.add_documents(["Test document"])
        
        # Save index
        save_path = tmp_path / "test_index"
        rag.save_index(str(save_path))
        
        assert (save_path / "faiss_index.bin").exists()
        assert (save_path / "documents.pkl").exists()
        
        # Load index
        rag2 = AgenticRAG(device="cpu")
        rag2.load_index(str(save_path))
        
        assert len(rag2.retriever.documents) > 0


class TestDocument:
    """Test Document dataclass"""
    
    def test_creation(self):
        """Test document creation"""
        doc = Document(
            id="test",
            content="Test content",
            metadata={"key": "value"}
        )
        
        assert doc.id == "test"
        assert doc.content == "Test content"
        assert doc.metadata["key"] == "value"
        assert doc.embeddings is None
    
    def test_with_embeddings(self):
        """Test document with embeddings"""
        embeddings = np.random.rand(384)
        doc = Document(
            id="test",
            content="Test",
            metadata={},
            embeddings=embeddings
        )
        
        assert doc.embeddings is not None
        assert doc.embeddings.shape == (384,)


@pytest.mark.parametrize("chunk_size,chunk_overlap", [
    (512, 50),
    (256, 25),
    (1024, 100)
])
def test_chunking_parameters(chunk_size, chunk_overlap):
    """Test different chunking parameters"""
    rag = AgenticRAG(
        device="cpu",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    assert rag.text_splitter.chunk_size == chunk_size
    assert rag.text_splitter.chunk_overlap == chunk_overlap


@pytest.fixture
def sample_documents():
    """Sample documents for testing"""
    return [
        "Artificial intelligence is transforming technology.",
        "Machine learning enables pattern recognition.",
        "Deep learning uses neural networks."
    ]


def test_batch_processing(sample_documents):
    """Test batch document processing"""
    rag = AgenticRAG(device="cpu")
    num_chunks = rag.add_documents(sample_documents)
    assert num_chunks > 0
    assert num_chunks >= len(sample_documents)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
