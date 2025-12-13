"""
Agentic RAG System - A production-ready Retrieval-Augmented Generation system
with autonomous decision-making capabilities.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

from .core.rag_pipeline import AgenticRAG
from .core.config import Config
from .agents.rag_agent import RAGAgent

__all__ = [
    "AgenticRAG",
    "Config",
    "RAGAgent",
]

# Package metadata
PACKAGE_NAME = "agentic_rag"
DESCRIPTION = "Agentic RAG system optimized for 20GB GPUs"
