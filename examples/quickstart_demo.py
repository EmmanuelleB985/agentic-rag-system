#!/usr/bin/env python3
"""
Quick example demonstrating the Agentic RAG system.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agentic_rag import AgenticRAG
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """A simple demonstration of the RAG system."""
    
    # Initialize the system
    print("\n Initializing RAG system...")
    
    # Use smaller models for demo to work on most GPUs
    rag = AgenticRAG(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        embedding_model="BAAI/bge-small-en-v1.5",  # Smaller embedding model
        device="cuda" if torch.cuda.is_available() else "cpu",
        quantization="4bit",  # Use 4-bit quantization to fit on smaller GPUs
        config={
            "chunk_size": 256,
            "temperature": 0.7
        }
    )
    
    print(" System initialized")
    
    # Add some sample documents
    print("\n Adding sample documents...")
    
    sample_docs = [
        """
        Quantum computing is a revolutionary computing paradigm that leverages quantum mechanical phenomena 
        such as superposition and entanglement. Unlike classical computers that use bits (0 or 1), 
        quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously.
        This allows quantum computers to perform certain calculations exponentially faster than classical computers.
        Key applications include cryptography, drug discovery, optimization problems, and machine learning.
        Major players in quantum computing include IBM, Google, Microsoft, and various startups.
        """,
        
        """
        Recent breakthroughs in quantum computing include Google's achievement of quantum supremacy in 2019,
        where their Sycamore processor performed a specific task in 200 seconds that would take classical
        supercomputers thousands of years. IBM has developed quantum computers with over 400 qubits,
        and continues to improve quantum error correction techniques. Quantum advantage has been demonstrated
        in various domains including optimization, simulation of quantum systems, and cryptographic tasks.
        """,
        
        """
        Machine learning and artificial intelligence have transformed numerous industries. Deep learning,
        a subset of machine learning, uses neural networks with multiple layers to learn from data.
        Transformers, introduced in 2017, revolutionized natural language processing and led to models
        like GPT, BERT, and T5. These models use attention mechanisms to process sequential data efficiently.
        Applications include language translation, text generation, image recognition, and autonomous driving.
        """,
        
        """
        The intersection of quantum computing and machine learning is an exciting frontier. Quantum machine
        learning algorithms could potentially provide exponential speedups for certain tasks. Variational
        quantum algorithms combine classical and quantum processing. Quantum neural networks are being
        explored for pattern recognition. However, current quantum hardware limitations, including noise
        and limited coherence time, present significant challenges for practical applications.
        """,
        
        """
        Climate change represents one of the most pressing challenges of our time. Global temperatures
        have risen by approximately 1.1 degrees Celsius since pre-industrial times. The Paris Agreement
        aims to limit warming to well below 2 degrees, preferably 1.5 degrees. Renewable energy sources
        like solar, wind, and hydroelectric power are rapidly becoming cost-competitive with fossil fuels.
        Carbon capture and storage technologies are being developed to remove CO2 from the atmosphere.
        """
    ]
    
    metadata = [
        {"topic": "quantum_computing", "subtopic": "basics"},
        {"topic": "quantum_computing", "subtopic": "breakthroughs"},
        {"topic": "machine_learning", "subtopic": "overview"},
        {"topic": "quantum_ml", "subtopic": "intersection"},
        {"topic": "climate", "subtopic": "overview"},
    ]
    
    rag.add_documents(sample_docs, metadata)
    print(f" Added {len(sample_docs)} documents to the knowledge base")
    
    # Demonstrate different query types
    print("\n Testing different query types:\n")
    
    queries = [
        {
            "question": "What is quantum computing and how does it differ from classical computing?",
            "use_agent": False,
            "description": "Simple retrieval query"
        },
        {
            "question": "Compare the recent achievements of IBM and Google in quantum computing",
            "use_agent": True,
            "description": "Comparative query requiring reasoning"
        },
        {
            "question": "How might quantum computing help address climate change?",
            "use_agent": True,
            "description": "Complex query requiring synthesis across topics"
        },
        {
            "question": "If a quantum computer has 50 qubits, how many possible states can it represent?",
            "use_agent": True,
            "description": "Query requiring calculation (2^50)"
        }
    ]
    
    for i, query_info in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query_info['description']}")
        print(f"Question: {query_info['question']}")
        print(f"Using Agent: {query_info['use_agent']}")
        print("-"*60)
        
        # Execute query
        response = rag.query(
            query_info['question'],
            use_agent=query_info['use_agent'],
            top_k=3
        )
        
        # Display results
        print(f"\n Answer:")
        print(response.answer)
        
        if response.sources:
            print(f"\n Sources Used ({len(response.sources)}):")
            for j, source in enumerate(response.sources[:2], 1):
                print(f"  {j}. Score: {source.get('score', 0):.3f}")
                print(f"     Topic: {source.get('topic', 'unknown')}")
                print(f"     Preview: {source['text'][:100]}...")
        
        if response.reasoning_trace and query_info['use_agent']:
            print(f"\n Reasoning Steps ({len(response.reasoning_trace)}):")
            for step in response.reasoning_trace[:3]:
                print(f"  Step {step['number']}: {step['action']}")
                print(f"    Thought: {step['thought'][:100]}...")
        
        print(f"\n⚡ Performance:")
        print(f"  Confidence: {response.confidence:.2%}")
        print(f"  Retrieval Time: {response.retrieval_time:.3f}s")
        print(f"  Total Time: {response.generation_time:.3f}s")
        print(f"  Tokens Used: {response.tokens_used}")
    
    # Interactive mode
    print("\n" + "="*60)
    print(" Interactive Mode")
    print("Type your questions (or 'quit' to exit)")
    print("="*60)
    
    while True:
        try:
            user_query = input("\n Your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print(" Goodbye!")
                break
            
            if not user_query:
                continue
            
            # Determine if agent should be used
            use_agent = any(keyword in user_query.lower() 
                          for keyword in ['compare', 'why', 'how', 'explain', 'analyze'])
            
            print(f"\n Processing{'with agent' if use_agent else ''}...")
            
            response = rag.query(user_query, use_agent=use_agent)
            
            print(f"\n Answer:")
            print(response.answer)
            
            print(f"\n Confidence: {response.confidence:.2%} | Time: {response.generation_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    main()
