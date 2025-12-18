#!/usr/bin/env python
"""
Setup script for Agentic RAG System
Downloads models and prepares the environment
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import requests
from tqdm import tqdm

sys.path.append('..')
from agentic_rag.utils import ensure_dir, download_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Setup:
    """Setup and initialization utilities"""
    
    def __init__(self, models_dir: str = "./models", data_dir: str = "./data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.required_models = {
            "text": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ],
            "multimodal": [
                "openai/clip-vit-base-patch32",
                "Salesforce/blip2-opt-2.7b",
                "facebook/wav2vec2-base",
                "openai/whisper-small"
            ]
        }
    
    def check_environment(self):
        """Check system environment"""
        print("Checking environment...")
        
        # Check Python version
        python_version = sys.version_info
        print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or python_version.minor < 9:
            logger.warning("Python 3.9+ is recommended")
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Check GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {total_memory:.1f} GB")
            
            if total_memory < 20:
                logger.warning(f"GPU has {total_memory:.1f}GB. 20GB+ recommended for best performance")
        else:
            logger.warning("No GPU detected. CPU mode will be slow")
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        print(f"Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 50:
            logger.warning(f"Only {free_gb:.1f}GB free. 50GB+ recommended")
        
        return cuda_available
    
    def create_directories(self):
        """Create necessary directories"""
        print("\nCreating directories...")
        
        directories = [
            self.models_dir,
            self.data_dir,
            self.data_dir / "documents",
            self.data_dir / "images",
            self.data_dir / "audio",
            self.data_dir / "video",
            Path("./indices"),
            Path("./cache"),
            Path("./logs"),
            Path("./temp")
        ]
        
        for directory in directories:
            ensure_dir(directory)
            print(f"  ✓ {directory}")
    
    def download_models(self, model_type: str = "all"):
        """Download required models"""
        print(f"\nDownloading {model_type} models...")
        
        if model_type == "all":
            models_to_download = []
            for models in self.required_models.values():
                models_to_download.extend(models)
        elif model_type in self.required_models:
            models_to_download = self.required_models[model_type]
        else:
            logger.error(f"Unknown model type: {model_type}")
            return
        
        success = 0
        failed = []
        
        for model_name in tqdm(models_to_download, desc="Downloading models"):
            try:
                # Check if already exists
                model_path = self.models_dir / model_name.replace("/", "_")
                
                if model_path.exists():
                    print(f"  ✓ {model_name} (already exists)")
                    success += 1
                    continue
                
                # Download model
                if download_model(model_name, str(self.models_dir)):
                    print(f"  ✓ {model_name}")
                    success += 1
                else:
                    failed.append(model_name)
                    print(f"  ✗ {model_name}")
            
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                failed.append(model_name)
        
        print(f"\nDownloaded {success}/{len(models_to_download)} models")
        
        if failed:
            print("Failed models:")
            for model in failed:
                print(f"  - {model}")
    
    def download_sample_data(self):
        """Download sample datasets"""
        print("\nDownloading sample data...")
        
        sample_datasets = {
            "wikipedia_sample": "https://example.com/wikipedia_sample.jsonl",
            "images_sample": "https://example.com/images_sample.zip",
            "audio_sample": "https://example.com/audio_sample.zip"
        }
        
        # Note: Replace with actual dataset URLs
        print("Sample datasets would be downloaded here")
        print("For now, creating placeholder files...")
        
        # Create sample text documents
        sample_docs = [
            "Artificial intelligence is transforming technology.",
            "Machine learning enables computers to learn from data.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        doc_file = self.data_dir / "documents" / "sample_docs.txt"
        with open(doc_file, 'w') as f:
            f.write('\n\n'.join(sample_docs))
        
        print(f"  ✓ Created {doc_file}")
    
    def install_dependencies(self):
        """Install additional dependencies"""
        print("\nChecking dependencies...")
        
        try:
            import faiss
            print("  ✓ FAISS installed")
        except ImportError:
            print("  ✗ FAISS not installed")
            print("    Run: pip install faiss-cpu or faiss-gpu")
        
        try:
            import chromadb
            print("  ✓ ChromaDB installed")
        except ImportError:
            print("  ✗ ChromaDB not installed")
            print("    Run: pip install chromadb")
        
        try:
            import cv2
            print("  ✓ OpenCV installed")
        except ImportError:
            print("  ✗ OpenCV not installed")
            print("    Run: pip install opencv-python")
        
        try:
            import librosa
            print("  ✓ Librosa installed")
        except ImportError:
            print("  ✗ Librosa not installed")
            print("    Run: pip install librosa")
    
    def test_basic_functionality(self):
        """Test basic system functionality"""
        print("\nTesting basic functionality...")
        
        try:
            # Test text embedding
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(["Test sentence"])
            print("  ✓ Text embedding working")
        except Exception as e:
            print(f"  ✗ Text embedding failed: {e}")
        
        try:
            # Test FAISS
            import faiss
            import numpy as np
            index = faiss.IndexFlatL2(384)
            index.add(np.random.rand(10, 384).astype('float32'))
            print("  ✓ FAISS indexing working")
        except Exception as e:
            print(f"  ✗ FAISS indexing failed: {e}")
        
        try:
            # Test model loading (small model)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            print("  ✓ Model loading working")
        except Exception as e:
            print(f"  ✗ Model loading failed: {e}")
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("=" * 60)
        print("Agentic RAG System Setup")
        print("=" * 60)
        
        # Check environment
        cuda_available = self.check_environment()
        
        # Create directories
        self.create_directories()
        
        # Install dependencies check
        self.install_dependencies()
        
        # Download models
        if cuda_available:
            self.download_models("all")
        else:
            print("\nSkipping large model downloads (CPU mode)")
            self.download_models("text")
        
        # Download sample data
        self.download_sample_data()
        
        # Test functionality
        self.test_basic_functionality()
        
        print("\n" + "=" * 60)
        print("Setup complete!")
        print("=" * 60)


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Agentic RAG System")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory for models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for data"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["all", "text", "multimodal"],
        default="all",
        help="Type of models to download"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model downloads"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run tests"
    )
    
    args = parser.parse_args()
    
    setup = Setup(args.models_dir, args.data_dir)
    
    if args.test_only:
        setup.check_environment()
        setup.test_basic_functionality()
    elif args.skip_models:
        setup.check_environment()
        setup.create_directories()
        setup.install_dependencies()
        setup.test_basic_functionality()
    else:
        setup.run_full_setup()


if __name__ == "__main__":
    main()
