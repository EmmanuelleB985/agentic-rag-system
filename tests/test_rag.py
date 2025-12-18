"""
Test suite for Agentic RAG System
"""

import pytest
import asyncio
import sys
import torch
from pathlib import Path
sys.path.append('..')

from agentic_rag import AgenticRAG, MultimodalAgenticRAG
from agentic_rag.multimodal import Modality, MultimodalDocument
from agentic_rag.agents import ReactiveAgent, PlanningAgent
from agentic_rag.retrieval import HybridRetriever, VectorStore
from PIL import Image
import numpy as np


class TestCoreRAG:
    """Test core RAG functionality"""
    
    @pytest.fixture
    def rag_system(self):
        """Create RAG system instance"""
        return AgenticRAG(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            quantization="4bit"
        )
    
    def test_initialization(self, rag_system):
        """Test system initialization"""
        assert rag_system is not None
        assert rag_system.embedder is not None
        assert rag_system.retriever is not None
        assert rag_system.agent is not None
    
    def test_add_documents(self, rag_system):
        """Test document addition"""
        documents = [
            "Machine learning is a subset of AI.",
            "Deep learning uses neural networks."
        ]
        
        num_chunks = rag_system.add_documents(documents)
        assert num_chunks > 0
    
    def test_query(self, rag_system):
        """Test querying"""
        rag_system.add_documents(["Python is a programming language."])
        
        result = rag_system.query("What is Python?")
        assert result is not None
        assert result.answer is not None
        assert len(result.sources) > 0
    
    def test_save_load_index(self, rag_system, tmp_path):
        """Test index persistence"""
        rag_system.add_documents(["Test document"])
        
        save_path = tmp_path / "test_index"
        rag_system.save_index(str(save_path))
        assert save_path.exists()
        
        rag_system.load_index(str(save_path))
        result = rag_system.query("test")
        assert result is not None


class TestMultimodalRAG:
    """Test multimodal RAG functionality"""
    
    @pytest.fixture
    async def mm_rag_system(self):
        """Create multimodal RAG system"""
        return MultimodalAgenticRAG(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cuda" if torch.cuda.is_available() else "cpu",
            quantization="4bit"
        )
    
    @pytest.mark.asyncio
    async def test_multimodal_initialization(self, mm_rag_system):
        """Test multimodal system initialization"""
        rag = await mm_rag_system
        assert rag is not None
        assert rag.mm_embedder is not None
        assert rag.mm_retriever is not None
        assert rag.mm_agent is not None
    
    @pytest.mark.asyncio
    async def test_add_text_document(self, mm_rag_system):
        """Test adding text document"""
        rag = await mm_rag_system
        doc_id = await rag.add_multimodal_document(
            "Test text content",
            Modality.TEXT
        )
        assert doc_id is not None
    
    @pytest.mark.asyncio
    async def test_add_image_document(self, mm_rag_system):
        """Test adding image document"""
        rag = await mm_rag_system
        
        # Create test image
        image = Image.new('RGB', (224, 224), color='red')
        
        doc_id = await rag.add_multimodal_document(
            image,
            Modality.IMAGE
        )
        assert doc_id is not None
    
    @pytest.mark.asyncio
    async def test_multimodal_query(self, mm_rag_system):
        """Test multimodal querying"""
        rag = await mm_rag_system
        
        # Add documents
        await rag.add_multimodal_document(
            "The sky is blue.",
            Modality.TEXT
        )
        
        # Query
        result = await rag.multimodal_query(
            "What color is the sky?",
            modalities=[Modality.TEXT]
        )
        
        assert result is not None
        assert result.answer is not None
        assert len(result.sources) > 0
    
    @pytest.mark.asyncio
    async def test_cross_modal_query(self, mm_rag_system):
        """Test cross-modal querying"""
        rag = await mm_rag_system
        
        # Add text and image
        await rag.add_multimodal_document(
            "A red circle",
            Modality.TEXT
        )
        
        image = Image.new('RGB', (224, 224), color='red')
        await rag.add_multimodal_document(
            image,
            Modality.IMAGE
        )
        
        # Cross-modal query
        result = await rag.multimodal_query(
            "Find red objects",
            modalities=[Modality.TEXT, Modality.IMAGE]
        )
        
        assert result is not None
        assert len(result.modalities_used) > 0


class TestAgents:
    """Test agent functionality"""
    
    @pytest.fixture
    def reactive_agent(self):
        """Create reactive agent"""
        return ReactiveAgent(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    @pytest.fixture
    def planning_agent(self):
        """Create planning agent"""
        return PlanningAgent(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    @pytest.mark.asyncio
    async def test_reactive_agent_act(self, reactive_agent):
        """Test reactive agent action"""
        observation = "User asks about weather"
        context = ["sunny", "warm"]
        
        action = await reactive_agent.act(observation, context)
        assert action is not None
        assert isinstance(action, str)
    
    @pytest.mark.asyncio
    async def test_planning_agent_plan(self, planning_agent):
        """Test planning agent"""
        goal = "Write a report"
        
        plan = await planning_agent.plan(goal)
        assert plan is not None
        assert isinstance(plan, list)
        assert len(plan) > 0
    
    @pytest.mark.asyncio
    async def test_agent_tool_use(self, reactive_agent):
        """Test agent tool usage"""
        # Add a simple tool
        async def simple_tool(**kwargs):
            return "Tool executed"
        
        reactive_agent.add_tool("test_tool", simple_tool)
        
        result = await reactive_agent.use_tool("test_tool")
        assert result == "Tool executed"


class TestRetrieval:
    """Test retrieval components"""
    
    def test_vector_store(self):
        """Test vector store"""
        store = VectorStore(dimension=384)
        
        # Add documents
        embeddings = np.random.rand(5, 384).astype('float32')
        documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        store.add(embeddings, documents)
        
        # Search
        query = np.random.rand(384).astype('float32')
        results = store.search(query, k=3)
        
        assert len(results) == 3
        assert results[0].content in documents
    
    def test_hybrid_retriever(self):
        """Test hybrid retriever"""
        retriever = HybridRetriever(
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        documents = [
            "Python is great for data science.",
            "JavaScript is used for web development.",
            "Machine learning requires data."
        ]
        
        retriever.add_documents(documents)
        
        results = retriever.search("data science Python", k=2)
        assert len(results) <= 2
        assert results[0].content in documents


class TestUtils:
    """Test utility functions"""
    
    def test_config_loading(self, tmp_path):
        """Test configuration loading"""
        from agentic_rag.utils import load_config, save_config
        
        config = {
            "model": {"name": "test"},
            "retrieval": {"top_k": 5}
        }
        
        config_path = tmp_path / "test_config.yaml"
        save_config(config, str(config_path))
        
        loaded = load_config(str(config_path))
        assert loaded["model"]["name"] == "test"
        assert loaded["retrieval"]["top_k"] == 5
    
    def test_text_chunking(self):
        """Test text chunking"""
        from agentic_rag.utils import chunk_text
        
        text = "a" * 1000  # 1000 character string
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 1
        assert all(len(c) <= 100 for c in chunks)
    
    def test_metrics_tracker(self):
        """Test metrics tracking"""
        from agentic_rag.utils import MetricsTracker
        
        tracker = MetricsTracker()
        
        # Log some metrics
        tracker.log_query("test query", 0.5, 0.85)
        tracker.log_error("test error")
        
        summary = tracker.get_summary()
        assert summary["total_queries"] == 1
        assert summary["total_errors"] == 1
        assert summary["avg_response_time"] == 0.5


@pytest.mark.integration
class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow"""
        # Initialize system
        rag = MultimodalAgenticRAG(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Add various documents
        await rag.add_multimodal_document(
            "Artificial intelligence is transforming technology.",
            Modality.TEXT
        )
        
        # Create and add image
        image = Image.new('RGB', (224, 224), color='blue')
        await rag.add_multimodal_document(
            image,
            Modality.IMAGE
        )
        
        # Query with agent
        result = await rag.multimodal_query(
            "What topics are covered in our documents?",
            use_agent=True
        )
        
        assert result is not None
        assert result.answer is not None
        assert result.confidence > 0
        assert result.processing_time > 0


def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
