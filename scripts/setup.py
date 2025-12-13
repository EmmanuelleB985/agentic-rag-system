#!/usr/bin/env python3
"""
Setup script for Agentic RAG System.
Downloads models, prepares datasets, and verifies installation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from tqdm import tqdm
import yaml
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_gpu():
    """Check GPU availability and specifications."""
    if not torch.cuda.is_available():
        logger.warning("No GPU detected. System will run on CPU (slower performance)")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Found {gpu_count} GPU(s)")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        if gpu_memory < 20:
            logger.warning(f"  GPU {i} has less than 20GB memory. May need stronger quantization.")
    
    return True


def download_models(config_path="config.yaml"):
    """Download required models based on configuration."""
    logger.info("Downloading models...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    
    # Download LLM
    model_name = config['model']['name']
    logger.info(f"Downloading LLM: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"   Tokenizer downloaded")
        
        # Just download, don't load (to save memory)
        AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir="./data/cache"
        )
        logger.info(f"   Model downloaded")
    except Exception as e:
        logger.error(f"   Failed to download LLM: {e}")
        return False
    
    # Download embedding model
    embedding_model = config['embeddings']['model']
    logger.info(f"Downloading embedding model: {embedding_model}")
    try:
        SentenceTransformer(embedding_model, cache_folder="./data/cache")
        logger.info(f"   Embedding model downloaded")
    except Exception as e:
        logger.error(f"   Failed to download embedding model: {e}")
        return False
    
    return True


def download_sample_data():
    """Download sample datasets for testing."""
    logger.info("Downloading sample datasets...")
    
    from datasets import load_dataset
    
    datasets_to_download = [
        ("wikipedia", "20220301.en", 1000),  # Sample of Wikipedia
        ("ms_marco", "v2.1", 1000),  # Sample of MS MARCO
    ]
    
    data_dir = Path("./data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset_name, config_name, sample_size in datasets_to_download:
        logger.info(f"Downloading {dataset_name} (sample: {sample_size} docs)")
        try:
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
            else:
                dataset = load_dataset(dataset_name, split="train", streaming=True)
            
            # Take sample
            samples = []
            for i, item in enumerate(dataset):
                if i >= sample_size:
                    break
                samples.append(item)
            
            # Save sample
            import json
            sample_path = data_dir / f"{dataset_name}_sample.json"
            with open(sample_path, 'w') as f:
                json.dump(samples, f)
            
            logger.info(f"  ✓ Saved {len(samples)} samples to {sample_path}")
            
        except Exception as e:
            logger.warning(f"  ✗ Failed to download {dataset_name}: {e}")
    
    return True


def create_directories():
    """Create necessary directory structure."""
    directories = [
        "./data/datasets",
        "./data/indices",
        "./data/cache",
        "./logs",
        "./notebooks",
        "./outputs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def verify_installation():
    """Verify that all components are properly installed."""
    logger.info("Verifying installation...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or python_version.minor < 9:
        logger.warning(f"Python {python_version.major}.{python_version.minor} detected. Python 3.9+ recommended.")
    
    # Check critical packages
    critical_packages = [
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "langchain",
        "datasets"
    ]
    
    missing_packages = []
    for package in critical_packages:
        try:
            __import__(package)
            logger.info(f"  ✓ {package} installed")
        except ImportError:
            logger.error(f"  ✗ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please run: pip install -r requirements.txt")
        return False
    
    # Test basic imports
    try:
        from agentic_rag import AgenticRAG
        logger.info("  Package imports working")
    except ImportError as e:
        logger.error(f"  Package import failed: {e}")
        return False
    
    return True


def run_quick_test():
    """Run a quick test of the system."""
    logger.info("Running quick test...")
    
    try:
        # Import with minimal resources
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        
        from agentic_rag import AgenticRAG
        
        # Initialize with small model for testing
        rag = AgenticRAG(
            model_name="gpt2",  # Small model for testing
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Small embedding model
            device="cuda" if torch.cuda.is_available() else "cpu",
            quantization="8bit"
        )
        
        # Add a test document
        rag.add_documents([
            "The Earth orbits around the Sun once every 365.25 days.",
            "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
        ])
        
        # Test query
        response = rag.query("How long does Earth take to orbit the Sun?", use_agent=False)
        
        logger.info(f"   Test query successful")
        logger.info(f"   Response: {response.answer[:100]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"   Test failed: {e}")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Agentic RAG System")
    parser.add_argument("--download-models", action="store_true", help="Download models")
    parser.add_argument("--download-data", action="store_true", help="Download sample datasets")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--full", action="store_true", help="Run full setup")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    
    # Create directories
    logger.info("Step 1: Creating directories...")
    create_directories()
    
    # Check GPU
    logger.info("\nStep 2: Checking GPU...")
    has_gpu = check_gpu()
    
    # Verify installation
    logger.info("\nStep 3: Verifying installation...")
    if not verify_installation():
        logger.error("Installation verification failed. Please fix issues before continuing.")
        return 1
    
    # Download models if requested
    if args.download_models or args.full:
        logger.info("\nStep 4: Downloading models...")
        if not download_models(args.config):
            logger.error("Model download failed.")
            return 1
    
    # Download data if requested
    if args.download_data or args.full:
        logger.info("\nStep 5: Downloading sample data...")
        if not download_sample_data():
            logger.warning("Some datasets failed to download, but setup can continue.")
    
    # Run test if requested
    if args.test or args.full:
        logger.info("\nStep 6: Running quick test...")
        if not run_quick_test():
            logger.warning("Test failed, but this might be due to memory constraints.")
    
    logger.info(" Setup complete!")
    logger.info("\nNext steps:")
    logger.info("1. Activate your virtual environment")
    logger.info("2. Run: python -c 'from agentic_rag import AgenticRAG'")
    logger.info("\nFor full documentation, see README.md")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
