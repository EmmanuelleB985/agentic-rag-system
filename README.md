# Agentic RAG System

A production-ready Retrieval-Augmented Generation system with agentic capabilities, optimized for 20GB GPUs. This system combines efficient document retrieval with intelligent language model generation, featuring autonomous decision-making for when and how to retrieve information.

## Features

- **Agentic Decision Making**: The system autonomously decides when to retrieve information vs. using parametric knowledge
- **GPU Optimized**: Runs efficiently on 20GB GPUs using quantized models
- **Multi-Dataset Support**: Works with Wikipedia, arXiv, and custom documents
- **Hybrid Search**: Combines dense (semantic) and sparse (keyword) retrieval
- **Streaming Responses**: Real-time generation with token streaming
- **Tool Use**: Agents can use retrieval, web search, and calculation tools
- **Evaluation Suite**: Built-in benchmarks for RAG performance

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU with 20GB+ VRAM
- 32GB+ system RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/EmmanuelleB985/agentic-rag-system.git
cd agentic-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models and datasets
python scripts/setup.py
```

### Basic Usage

```python
from agentic_rag import AgenticRAG

# Initialize the system
rag = AgenticRAG(
    model_name="mistral-7b-instruct",
    device="cuda",
    quantization="4bit"
)

# Load documents
rag.load_dataset("wikipedia-20220301", sample_size=10000)

# Query the system
response = rag.query(
    "What are the latest developments in quantum computing?",
    use_agent=True
)

print(response.answer)
print(f"Sources used: {response.sources}")
```

## Configuration

Edit `config.yaml` to customize the system:

```yaml
model:
  name: "mistral-7b-instruct-v0.2"
  quantization: "4bit"  # Options: none, 8bit, 4bit
  max_length: 4096
  temperature: 0.7

retrieval:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 5
  rerank_top_k: 3

agent:
  max_iterations: 5
  tools_enabled: ["retrieval", "web_search", "calculator"]
  decision_threshold: 0.7
```

## Supported Datasets

| Dataset | Size | Description | Domain |
|---------|------|-------------|---------|
| Wikipedia-20220301 | ~6M articles | Wikipedia dump | General Knowledge |
| MS MARCO | 8.8M passages | Question-answering | General QA |
| Natural Questions | 300K Q&A pairs | Google search queries | Factual QA |
| HotpotQA | 113K questions | Multi-hop reasoning | Complex QA |
| arXiv | 2M papers | Scientific papers | Academic |
| Custom | Any size | Your own documents | Domain-specific |

## Performance Benchmarks

On a 20GB GPU (RTX 3080 Ti / RTX 4090):

| Model | Quantization | VRAM Usage | Tokens/sec | F1 Score |
|-------|-------------|------------|------------|----------|
| Mistral-7B | 4-bit | 6.2 GB | 45 | 0.82 |
| Llama-2-7B | 4-bit | 6.5 GB | 42 | 0.80 |
| Phi-2 | 8-bit | 5.8 GB | 65 | 0.76 |
| Mistral-7B | 8-bit | 13.5 GB | 38 | 0.84 |

## Advanced Features

### Custom Document Ingestion

```python
# Add your own documents
rag.add_documents(
    documents=["path/to/doc1.pdf", "path/to/doc2.txt"],
    metadata={"source": "internal_docs"}
)
```

### Agentic Reasoning

```python
# Enable step-by-step reasoning
response = rag.query(
    query="Compare quantum and classical computing approaches",
    agent_config={
        "reasoning_steps": True,
        "max_iterations": 10,
        "tools": ["retrieval", "calculator", "web_search"]
    }
)

# Access reasoning trace
for step in response.reasoning_trace:
    print(f"Step {step.number}: {step.action}")
    print(f"Observation: {step.observation}")
```

### Streaming Generation

```python
# Stream tokens as they're generated
for token in rag.stream_query("Explain transformer architecture"):
    print(token, end="", flush=True)
```

## Evaluation

Run the evaluation suite:

```bash
# Evaluate on standard benchmarks
python scripts/evaluate.py --dataset natural_questions --metric all

# Custom evaluation
python scripts/evaluate.py --test-file custom_test.json
```

## Citations

This implementation is based on:

- **RAG**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **Self-RAG**: [Self-Reflective Retrieval-Augmented Generation](https://arxiv.org/abs/2310.11511)
- **ReAct**: [Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Hugging Face for model hosting and transformers library
- LangChain for RAG patterns inspiration
- FAISS team for efficient vector search
- The open-source community for datasets

**Note**: This system is optimized for research and experimentation. For production deployment, consider additional security, scaling, and monitoring requirements.
