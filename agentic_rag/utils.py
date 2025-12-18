"""
Utility functions for the RAG system
"""

import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from datetime import datetime
import hashlib
import os


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix == '.yaml' or path.suffix == '.yml':
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file"""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix == '.yaml' or path.suffix == '.yml':
            yaml.dump(config, f, default_flow_style=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def get_device() -> str:
    """Get the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_gpu_memory() -> Dict[str, float]:
    """Get GPU memory statistics"""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "allocated": torch.cuda.memory_allocated(0) / 1e9,
        "cached": torch.cuda.memory_reserved(0) / 1e9,
        "free": (torch.cuda.get_device_properties(0).total_memory - 
                torch.cuda.memory_allocated(0)) / 1e9
    }


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Simple text chunking"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if end < len(text) else end
    
    return chunks


def generate_id(content: str) -> str:
    """Generate unique ID for content"""
    timestamp = datetime.now().isoformat()
    content_hash = hashlib.md5(f"{content[:100]}{timestamp}".encode()).hexdigest()
    return content_hash


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def batch_process(items: List[Any], batch_size: int = 32):
    """Yield batches of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        logging.info(f"{self.name} took {self.elapsed:.2f} seconds")


def ensure_dir(path: str):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_documents(file_path: str) -> List[str]:
    """Load documents from file"""
    path = Path(file_path)
    
    if path.suffix == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            return [f.read()]
    
    elif path.suffix == '.jsonl':
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                if isinstance(doc, dict) and 'text' in doc:
                    documents.append(doc['text'])
                elif isinstance(doc, str):
                    documents.append(doc)
        return documents
    
    elif path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return [d['text'] if isinstance(d, dict) else d for d in data]
            elif isinstance(data, dict) and 'documents' in data:
                return data['documents']
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_documents(documents: List[str], file_path: str):
    """Save documents to file"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if path.suffix == '.txt':
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(documents))
    
    elif path.suffix == '.jsonl':
        with open(path, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps({'text': doc}) + '\n')
    
    elif path.suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'documents': documents}, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def download_model(model_name: str, cache_dir: str = "./models"):
    """Download and cache a model"""
    from transformers import AutoModel, AutoTokenizer
    
    ensure_dir(cache_dir)
    
    try:
        # Download model
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            resume_download=True
        )
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            resume_download=True
        )
        
        logging.info(f"Successfully downloaded {model_name}")
        return True
    
    except Exception as e:
        logging.error(f"Failed to download {model_name}: {e}")
        return False


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate number of tokens in text"""
    return int(len(text) / chars_per_token)


def truncate_text(text: str, max_tokens: int, chars_per_token: float = 4.0) -> str:
    """Truncate text to approximately max tokens"""
    max_chars = int(max_tokens * chars_per_token)
    if len(text) <= max_chars:
        return text
    
    # Truncate and add ellipsis
    return text[:max_chars - 3] + "..."


class MetricsTracker:
    """Track system metrics"""
    
    def __init__(self):
        self.metrics = {
            'queries': [],
            'response_times': [],
            'retrieval_scores': [],
            'errors': []
        }
    
    def log_query(self, query: str, response_time: float, score: float = 0.0):
        """Log a query"""
        self.metrics['queries'].append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'response_time': response_time,
            'score': score
        })
        
        self.metrics['response_times'].append(response_time)
        if score > 0:
            self.metrics['retrieval_scores'].append(score)
    
    def log_error(self, error: str):
        """Log an error"""
        self.metrics['errors'].append({
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        response_times = self.metrics['response_times']
        scores = self.metrics['retrieval_scores']
        
        return {
            'total_queries': len(self.metrics['queries']),
            'total_errors': len(self.metrics['errors']),
            'avg_response_time': np.mean(response_times) if response_times else 0,
            'median_response_time': np.median(response_times) if response_times else 0,
            'avg_retrieval_score': np.mean(scores) if scores else 0,
            'last_query': self.metrics['queries'][-1] if self.metrics['queries'] else None
        }
    
    def save(self, file_path: str):
        """Save metrics to file"""
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load(self, file_path: str):
        """Load metrics from file"""
        with open(file_path, 'r') as f:
            self.metrics = json.load(f)
