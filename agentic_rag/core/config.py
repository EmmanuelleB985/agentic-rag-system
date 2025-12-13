"""
Configuration management for Agentic RAG System.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for Agentic RAG System."""
    
    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    quantization: str = "4bit"
    max_length: int = 4096
    temperature: float = 0.7
    device: str = "cuda"
    
    # Retrieval settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    rerank_top_k: int = 3
    similarity_threshold: float = 0.7
    
    # Agent settings
    max_iterations: int = 5
    decision_threshold: float = 0.7
    use_agent: bool = True
    
    # Performance settings
    batch_size: int = 8
    num_workers: int = 4
    
    # Directory paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    cache_dir: Path = field(default_factory=lambda: Path("./data/cache"))
    indices_dir: Path = field(default_factory=lambda: Path("./data/indices"))
    logs_dir: Path = field(default_factory=lambda: Path("./logs"))
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested configuration
        flat_config = {}
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_config[key] = value
            else:
                flat_config[section] = values
        
        # Create config with valid fields only
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in flat_config.items() if k in valid_fields}
        
        return cls(**filtered_config)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, Path):
                result[field_name] = str(value)
            else:
                result[field_name] = value
        return result
    
    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_attr in ['data_dir', 'cache_dir', 'indices_dir', 'logs_dir']:
            dir_path = getattr(self, dir_attr)
            if isinstance(dir_path, (str, Path)):
                Path(dir_path).mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()