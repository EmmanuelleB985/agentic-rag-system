#!/usr/bin/env python
"""
Evaluation script for Agentic RAG System
Benchmarks and evaluates system performance
"""

import sys
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
import logging

sys.path.append('..')
from agentic_rag import AgenticRAG, MultimodalAgenticRAG
from agentic_rag.multimodal import Modality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    metric_name: str
    score: float
    details: Dict[str, Any]


class Evaluator:
    """Evaluation framework for RAG system"""
    
    def __init__(self, system_type: str = "text"):
        self.system_type = system_type
        self.metrics = []
        self.results = []
    
    async def setup_system(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize the RAG system"""
        if self.system_type == "multimodal":
            self.rag = MultimodalAgenticRAG(
                model_name=model_name,
                device="cuda",
                quantization="4bit"
            )
        else:
            self.rag = AgenticRAG(
                model_name=model_name,
                device="cuda",
                quantization="4bit"
            )
    
    def load_test_data(self, dataset_path: str) -> List[Dict]:
        """Load test dataset"""
        path = Path(dataset_path)
        
        if not path.exists():
            # Create sample test data
            test_data = [
                {
                    "question": "What is machine learning?",
                    "answer": "Machine learning is a subset of AI that enables systems to learn from data.",
                    "context": "Machine learning is a subset of artificial intelligence..."
                },
                {
                    "question": "What is deep learning?",
                    "answer": "Deep learning uses neural networks with multiple layers.",
                    "context": "Deep learning is a subset of machine learning..."
                }
            ]
            return test_data
        
        with open(path, 'r') as f:
            if path.suffix == '.jsonl':
                return [json.loads(line) for line in f]
            else:
                return json.load(f)
    
    def calculate_f1_score(self, predicted: str, reference: str) -> float:
        """Calculate F1 score"""
        pred_tokens = set(predicted.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not ref_tokens:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def calculate_exact_match(self, predicted: str, reference: str) -> float:
        """Calculate exact match score"""
        return 1.0 if predicted.strip().lower() == reference.strip().lower() else 0.0
    
    def calculate_bleu(self, predicted: str, reference: str) -> float:
        """Calculate BLEU score (simplified)"""
        from collections import Counter
        
        pred_tokens = predicted.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        # Unigram precision
        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)
        
        overlap = 0
        for token in pred_counts:
            overlap += min(pred_counts[token], ref_counts.get(token, 0))
        
        precision = overlap / len(pred_tokens) if pred_tokens else 0
        
        # Brevity penalty
        bp = min(1.0, len(pred_tokens) / len(ref_tokens))
        
        return bp * precision
    
    async def evaluate_retrieval(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate retrieval performance"""
        print("\nEvaluating retrieval performance...")
        
        scores = []
        times = []
        
        for item in test_data:
            start_time = time.time()
            
            # Add context document
            self.rag.add_documents([item.get("context", item["question"])])
            
            # Query
            result = self.rag.query(item["question"], use_agent=False)
            
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Calculate relevance (simplified)
            if result.sources:
                score = 1.0  # Retrieved relevant document
            else:
                score = 0.0
            
            scores.append(score)
        
        avg_score = np.mean(scores)
        avg_time = np.mean(times)
        
        return EvaluationResult(
            metric_name="retrieval",
            score=avg_score,
            details={
                "avg_score": avg_score,
                "avg_time": avg_time,
                "total_queries": len(test_data)
            }
        )
    
    async def evaluate_generation(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate generation quality"""
        print("\nEvaluating generation quality...")
        
        f1_scores = []
        exact_matches = []
        bleu_scores = []
        
        for item in test_data[:10]:  # Limit for speed
            # Add context
            self.rag.add_documents([item.get("context", item["question"])])
            
            # Generate answer
            result = self.rag.query(item["question"], use_agent=True)
            predicted = result.answer
            reference = item["answer"]
            
            # Calculate metrics
            f1_scores.append(self.calculate_f1_score(predicted, reference))
            exact_matches.append(self.calculate_exact_match(predicted, reference))
            bleu_scores.append(self.calculate_bleu(predicted, reference))
        
        return EvaluationResult(
            metric_name="generation",
            score=np.mean(f1_scores),
            details={
                "f1_score": np.mean(f1_scores),
                "exact_match": np.mean(exact_matches),
                "bleu_score": np.mean(bleu_scores),
                "samples": len(f1_scores)
            }
        )
    
    async def evaluate_multimodal(self, test_data: List[Dict]) -> EvaluationResult:
        """Evaluate multimodal capabilities"""
        print("\nEvaluating multimodal capabilities...")
        
        if self.system_type != "multimodal":
            return EvaluationResult(
                metric_name="multimodal",
                score=0.0,
                details={"error": "Not a multimodal system"}
            )
        
        scores = []
        
        # Simple multimodal tests
        from PIL import Image
        
        # Test text
        await self.rag.add_multimodal_document(
            "This is a test document about AI.",
            Modality.TEXT
        )
        
        # Test image
        test_image = Image.new('RGB', (224, 224), color='red')
        await self.rag.add_multimodal_document(
            test_image,
            Modality.IMAGE
        )
        
        # Query
        result = await self.rag.multimodal_query(
            "What content do we have?",
            modalities=[Modality.TEXT, Modality.IMAGE]
        )
        
        # Simple scoring based on whether it found both modalities
        score = len(set(result.modalities_used)) / 2.0
        
        return EvaluationResult(
            metric_name="multimodal",
            score=score,
            details={
                "modalities_tested": ["text", "image"],
                "modalities_found": [m.value for m in result.modalities_used],
                "score": score
            }
        )
    
    async def evaluate_latency(self, num_queries: int = 20) -> EvaluationResult:
        """Evaluate system latency"""
        print("\nEvaluating latency...")
        
        queries = [
            "What is AI?",
            "Explain machine learning",
            "What is deep learning?",
            "How does NLP work?",
            "What are neural networks?"
        ] * (num_queries // 5)
        
        times = []
        
        for query in queries:
            start = time.time()
            _ = self.rag.query(query, use_agent=False)
            elapsed = time.time() - start
            times.append(elapsed)
        
        return EvaluationResult(
            metric_name="latency",
            score=np.mean(times),
            details={
                "avg_latency": np.mean(times),
                "min_latency": np.min(times),
                "max_latency": np.max(times),
                "p50_latency": np.percentile(times, 50),
                "p95_latency": np.percentile(times, 95),
                "total_queries": len(times)
            }
        )
    
    async def evaluate_memory(self) -> EvaluationResult:
        """Evaluate memory usage"""
        print("\nEvaluating memory usage...")
        
        import torch
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get memory info
        cpu_memory = process.memory_info().rss / 1e9  # GB
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(0) / 1e9  # GB
        
        return EvaluationResult(
            metric_name="memory",
            score=cpu_memory + gpu_memory,
            details={
                "cpu_memory_gb": cpu_memory,
                "gpu_memory_gb": gpu_memory,
                "total_memory_gb": cpu_memory + gpu_memory
            }
        )
    
    async def run_evaluation(
        self,
        test_data_path: str,
        metrics: List[str] = None
    ) -> List[EvaluationResult]:
        """Run complete evaluation"""
        print("=" * 60)
        print("Starting RAG System Evaluation")
        print("=" * 60)
        
        # Load test data
        test_data = self.load_test_data(test_data_path)
        print(f"Loaded {len(test_data)} test samples")
        
        # Setup system
        await self.setup_system()
        
        # Select metrics
        if metrics is None:
            metrics = ["retrieval", "generation", "latency", "memory"]
        
        results = []
        
        # Run evaluations
        for metric in metrics:
            if metric == "retrieval":
                result = await self.evaluate_retrieval(test_data)
            elif metric == "generation":
                result = await self.evaluate_generation(test_data)
            elif metric == "multimodal":
                result = await self.evaluate_multimodal(test_data)
            elif metric == "latency":
                result = await self.evaluate_latency()
            elif metric == "memory":
                result = await self.evaluate_memory()
            else:
                continue
            
            results.append(result)
            print(f"  {metric}: {result.score:.3f}")
        
        return results
    
    def save_results(self, results: List[EvaluationResult], output_path: str):
        """Save evaluation results"""
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_type": self.system_type,
            "results": [
                {
                    "metric": r.metric_name,
                    "score": r.score,
                    "details": r.details
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    def print_summary(self, results: List[EvaluationResult]):
        """Print evaluation summary"""
        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result.metric_name.upper()}:")
            print(f"  Score: {result.score:.3f}")
            
            for key, value in result.details.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)


async def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument(
        "--test-file",
        type=str,
        default="test_data.jsonl",
        help="Path to test data file"
    )
    parser.add_argument(
        "--system-type",
        type=str,
        choices=["text", "multimodal"],
        default="text",
        help="Type of system to evaluate"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        choices=["retrieval", "generation", "multimodal", "latency", "memory", "all"],
        default=["all"],
        help="Metrics to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Handle "all" metrics
    if "all" in args.metrics:
        if args.system_type == "multimodal":
            metrics = ["retrieval", "generation", "multimodal", "latency", "memory"]
        else:
            metrics = ["retrieval", "generation", "latency", "memory"]
    else:
        metrics = args.metrics
    
    # Run evaluation
    evaluator = Evaluator(system_type=args.system_type)
    results = await evaluator.run_evaluation(args.test_file, metrics)
    
    # Save and display results
    evaluator.save_results(results, args.output)
    evaluator.print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
