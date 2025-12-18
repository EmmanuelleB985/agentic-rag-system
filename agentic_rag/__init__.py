"""
Agentic RAG System with Multimodal Support
"""

from .core import AgenticRAG
from .multimodal import (
    MultimodalAgenticRAG,
    MultimodalEmbedder,
    MultimodalRetriever,
    MultimodalAgent,
    Modality,
    DocumentType,
    MultimodalDocument,
    QueryResult
)
from .agents import BaseAgent, ReactiveAgent, PlanningAgent
from .retrieval import HybridRetriever, VectorStore
from .utils import setup_logging, load_config

__version__ = "2.0.0"
__all__ = [
    "AgenticRAG",
    "MultimodalAgenticRAG",
    "MultimodalEmbedder",
    "MultimodalRetriever",
    "MultimodalAgent",
    "Modality",
    "DocumentType",
    "MultimodalDocument",
    "QueryResult",
    "BaseAgent",
    "ReactiveAgent",
    "PlanningAgent",
    "HybridRetriever",
    "VectorStore",
    "setup_logging",
    "load_config"
]
