"""
Basic usage example for the Agentic RAG System
"""

import asyncio
import sys
sys.path.append('..')

from agentic_rag import AgenticRAG, MultimodalAgenticRAG
from agentic_rag.multimodal import Modality
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_text_example():
    """Basic text-only RAG example"""
    print("Basic Text RAG Example")
 
    # Initialize the system
    rag = AgenticRAG(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda",
        quantization="4bit"
    )
    
    # Add documents
    documents = [
        """Machine learning is a subset of artificial intelligence that enables 
        systems to learn and improve from experience without being explicitly programmed.""",
        
        """Deep learning is a subset of machine learning that uses neural networks 
        with multiple layers to progressively extract higher-level features from raw input.""",
        
        """Natural language processing (NLP) is a branch of AI that helps computers 
        understand, interpret and manipulate human language."""
    ]
    
    num_chunks = rag.add_documents(documents)
    print(f"Added {num_chunks} document chunks")
    
    # Query the system
    queries = [
        "What is machine learning?",
        "How does deep learning relate to machine learning?",
        "What is NLP used for?"
    ]
    
    for query in queries:
        result = rag.query(query, use_agent=True)
        print(f"\nQuery: {query}")
        print(f"Answer: {result.answer[:200]}...")
        print(f"Sources: {len(result.sources)} documents")
        print(f"Processing time: {result.processing_time:.2f}s")


async def multimodal_example():
    """Multimodal RAG example"""
    print("\n" + "="*50)
    print("Multimodal RAG Example")
    print("="*50)
    
    # Initialize multimodal system
    rag = MultimodalAgenticRAG(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda",
        quantization="4bit"
    )
    
    # Add text documents
    await rag.add_multimodal_document(
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        Modality.TEXT,
        metadata={"topic": "landmarks"}
    )
    
    # Add more documents (in practice, add real images/audio/video)
    await rag.add_multimodal_document(
        "The Great Wall of China is a series of fortifications built across China.",
        Modality.TEXT,
        metadata={"topic": "landmarks"}
    )
    
    # Query the system
    result = await rag.multimodal_query(
        "Tell me about famous landmarks",
        modalities=[Modality.TEXT],
        use_agent=True
    )
    
    print(f"\nQuery: Tell me about famous landmarks")
    print(f"Answer: {result.answer[:300]}...")
    print(f"Modalities used: {[m.value for m in result.modalities_used]}")
    print(f"Processing time: {result.processing_time:.2f}s")


async def agent_example():
    """Example using different agent types"""
    print("\n" + "="*50)
    print("Agent Types Example")
    print("="*50)
    
    from agentic_rag.agents import ReactiveAgent, PlanningAgent
    
    # Reactive agent
    reactive_agent = ReactiveAgent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda"
    )
    
    observation = "The user is asking about climate change"
    context = ["Global warming", "Carbon emissions", "Renewable energy"]
    
    action = await reactive_agent.act(observation, context)
    print(f"\nReactive Agent:")
    print(f"Observation: {observation}")
    print(f"Action: {action[:200]}...")
    
    # Planning agent
    planning_agent = PlanningAgent(
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        device="cuda"
    )
    
    goal = "Create a comprehensive report on artificial intelligence"
    plan = await planning_agent.plan(goal)
    
    print(f"\nPlanning Agent:")
    print(f"Goal: {goal}")
    print("Plan:")
    for i, step in enumerate(plan[:5], 1):
        print(f"  {step}")


async def retrieval_example():
    """Advanced retrieval example"""
    print("\n" + "="*50)
    print("Advanced Retrieval Example")
    print("="*50)
    
    from agentic_rag.retrieval import HybridRetriever
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        embedding_model="all-MiniLM-L6-v2",
        device="cuda"
    )
    
    # Add documents
    documents = [
        "Python is a high-level programming language.",
        "JavaScript is the language of the web.",
        "Machine learning models can be trained using Python.",
        "Web applications often use JavaScript for interactivity.",
        "Data science commonly uses Python libraries like pandas and numpy."
    ]
    
    retriever.add_documents(documents)
    
    # Search with hybrid retrieval
    query = "What programming language is used for data science?"
    results = retriever.search(query, k=3, rerank_top_k=2)
    
    print(f"\nQuery: {query}")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result.score:.3f}")
        print(f"     Content: {result.content}")


async def streaming_example():
    """Streaming response example"""
    print("Streaming Response Example")
    
    # This would be implemented with actual streaming
    print("\nSimulated streaming response:")
    
    tokens = ["Artificial", " intelligence", " is", " transforming", " how", " we", 
              " interact", " with", " technology", "."]
    
    print("Response: ", end="")
    for token in tokens:
        print(token, end="", flush=True)
        await asyncio.sleep(0.1)  # Simulate streaming delay
    print()


async def main():
    """Run all examples"""
    print("Agentic RAG System - Examples")
    
    examples = [
        basic_text_example,
        multimodal_example,
        agent_example,
        retrieval_example,
        streaming_example
    ]
    
    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        
        await asyncio.sleep(1)
    
if __name__ == "__main__":
    asyncio.run(main())
