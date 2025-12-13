#!/usr/bin/env python3
"""
Evaluation script for Agentic RAG System.
Supports standard RAG benchmarks and custom evaluation.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agentic_rag import AgenticRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluator for RAG systems."""
    
    def __init__(self, rag_system: AgenticRAG):
        """Initialize evaluator with a RAG system."""
        self.rag = rag_system
        self.results = []
    
    def evaluate_dataset(
        self,
        dataset_name: str,
        sample_size: int = 100,
        metrics: List[str] = ["f1", "exact_match", "retrieval_accuracy"]
    ) -> Dict[str, Any]:
        """
        Evaluate on a standard dataset.
        
        Args:
            dataset_name: Name of the dataset
            sample_size: Number of samples to evaluate
            metrics: List of metrics to compute
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on {dataset_name} with {sample_size} samples")
        
        # Load evaluation data
        eval_data = self._load_eval_data(dataset_name, sample_size)
        
        if not eval_data:
            logger.error(f"Could not load evaluation data for {dataset_name}")
            return {}
        
        # Run evaluation
        results = []
        for item in tqdm(eval_data, desc="Evaluating"):
            result = self._evaluate_single(item)
            results.append(result)
        
        # Compute metrics
        metrics_results = self._compute_metrics(results, metrics)
        
        return {
            "dataset": dataset_name,
            "sample_size": len(results),
            "metrics": metrics_results,
            "detailed_results": results[:10]  # First 10 for inspection
        }
    
    def _load_eval_data(self, dataset_name: str, sample_size: int) -> List[Dict]:
        """Load evaluation data for a dataset."""
        from datasets import load_dataset
        
        eval_data = []
        
        try:
            if dataset_name == "natural_questions":
                dataset = load_dataset("natural_questions", split="validation")
                for i, item in enumerate(dataset):
                    if i >= sample_size:
                        break
                    
                    # Extract question and answer
                    eval_item = {
                        "id": item["id"],
                        "question": item["question"]["text"],
                        "answers": [ans["text"] for ans in item["annotations"][0]["short_answers"]],
                        "context": item["document"]["text"]
                    }
                    eval_data.append(eval_item)
            
            elif dataset_name == "squad":
                dataset = load_dataset("squad", split="validation")
                for i, item in enumerate(dataset):
                    if i >= sample_size:
                        break
                    
                    eval_item = {
                        "id": item["id"],
                        "question": item["question"],
                        "answers": item["answers"]["text"],
                        "context": item["context"]
                    }
                    eval_data.append(eval_item)
            
            elif dataset_name == "ms_marco":
                dataset = load_dataset("ms_marco", "v2.1", split="dev")
                for i, item in enumerate(dataset):
                    if i >= sample_size:
                        break
                    
                    eval_item = {
                        "id": str(i),
                        "question": item["query"],
                        "answers": item["answers"],
                        "relevant_passages": item["passages"]["passage_text"]
                    }
                    eval_data.append(eval_item)
            
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
        
        return eval_data
    
    def _evaluate_single(self, item: Dict) -> Dict:
        """Evaluate a single question-answer pair."""
        start_time = time.time()
        
        # Get RAG response
        response = self.rag.query(item["question"], use_agent=False)
        
        # Extract predicted answer
        predicted = response.answer
        
        # Get ground truth answers
        ground_truth = item.get("answers", [])
        if not isinstance(ground_truth, list):
            ground_truth = [ground_truth]
        
        # Compute metrics for this item
        result = {
            "id": item["id"],
            "question": item["question"],
            "predicted": predicted,
            "ground_truth": ground_truth,
            "exact_match": self._compute_exact_match(predicted, ground_truth),
            "f1_score": self._compute_f1(predicted, ground_truth),
            "retrieval_accuracy": self._compute_retrieval_accuracy(response.sources, item),
            "confidence": response.confidence,
            "latency": time.time() - start_time
        }
        
        return result
    
    def _compute_exact_match(self, predicted: str, ground_truth: List[str]) -> float:
        """Compute exact match score."""
        predicted_normalized = predicted.lower().strip()
        for answer in ground_truth:
            if answer.lower().strip() in predicted_normalized:
                return 1.0
        return 0.0
    
    def _compute_f1(self, predicted: str, ground_truth: List[str]) -> float:
        """Compute F1 score between predicted and ground truth."""
        def tokenize(text):
            return text.lower().split()
        
        pred_tokens = set(tokenize(predicted))
        
        best_f1 = 0.0
        for answer in ground_truth:
            gold_tokens = set(tokenize(answer))
            
            if not pred_tokens or not gold_tokens:
                continue
            
            # Compute precision and recall
            common = pred_tokens.intersection(gold_tokens)
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(gold_tokens) if gold_tokens else 0
            
            # Compute F1
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)
        
        return best_f1
    
    def _compute_retrieval_accuracy(self, sources: List[Dict], item: Dict) -> float:
        """Compute retrieval accuracy if relevant passages are known."""
        if "relevant_passages" not in item:
            return -1.0  # Not applicable
        
        relevant = item["relevant_passages"]
        if not sources:
            return 0.0
        
        # Check if any retrieved source matches relevant passages
        for source in sources[:3]:  # Top 3
            source_text = source.get("text", "").lower()
            for rel_passage in relevant:
                if isinstance(rel_passage, str):
                    if rel_passage.lower()[:100] in source_text or source_text[:100] in rel_passage.lower():
                        return 1.0
        
        return 0.0
    
    def _compute_metrics(self, results: List[Dict], metrics: List[str]) -> Dict[str, float]:
        """Compute aggregate metrics from results."""
        aggregated = {}
        
        for metric in metrics:
            if metric == "exact_match":
                scores = [r["exact_match"] for r in results]
                aggregated["exact_match"] = np.mean(scores)
            
            elif metric == "f1":
                scores = [r["f1_score"] for r in results]
                aggregated["f1"] = np.mean(scores)
            
            elif metric == "retrieval_accuracy":
                scores = [r["retrieval_accuracy"] for r in results if r["retrieval_accuracy"] >= 0]
                if scores:
                    aggregated["retrieval_accuracy"] = np.mean(scores)
            
            elif metric == "latency":
                latencies = [r["latency"] for r in results]
                aggregated["avg_latency"] = np.mean(latencies)
                aggregated["p95_latency"] = np.percentile(latencies, 95)
            
            elif metric == "confidence":
                confidences = [r["confidence"] for r in results]
                aggregated["avg_confidence"] = np.mean(confidences)
        
        return aggregated
    
    def evaluate_custom(self, test_file: str) -> Dict[str, Any]:
        """
        Evaluate on a custom test file.
        
        Args:
            test_file: Path to JSON file with test cases
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on custom test file: {test_file}")
        
        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Run evaluation
        results = []
        for item in tqdm(test_data, desc="Evaluating custom"):
            result = self._evaluate_single(item)
            results.append(result)
        
        # Compute metrics
        metrics_results = self._compute_metrics(
            results,
            ["exact_match", "f1", "latency", "confidence"]
        )
        
        return {
            "test_file": test_file,
            "sample_size": len(results),
            "metrics": metrics_results,
            "detailed_results": results
        }
    
    def generate_report(self, results: Dict[str, Any], output_file: str = None):
        """Generate evaluation report."""
        report = []
        report.append("="*60)
        report.append("RAG System Evaluation Report")
        report.append("="*60)
        
        # Summary metrics
        report.append("\n## Summary Metrics")
        report.append("-"*40)
        
        if "metrics" in results:
            for metric, value in results["metrics"].items():
                if isinstance(value, float):
                    report.append(f"{metric:20s}: {value:.4f}")
                else:
                    report.append(f"{metric:20s}: {value}")
        
        # Detailed results sample
        if "detailed_results" in results and results["detailed_results"]:
            report.append("\n## Sample Results")
            report.append("-"*40)
            
            for i, result in enumerate(results["detailed_results"][:5], 1):
                report.append(f"\n### Example {i}")
                report.append(f"Question: {result['question'][:100]}...")
                report.append(f"Predicted: {result['predicted'][:100]}...")
                report.append(f"Ground Truth: {str(result['ground_truth'])[:100]}...")
                report.append(f"Exact Match: {result['exact_match']:.2f}")
                report.append(f"F1 Score: {result['f1_score']:.4f}")
        
        # Performance stats
        report.append("\n## Performance Statistics")
        report.append("-"*40)
        report.append(f"Dataset: {results.get('dataset', 'custom')}")
        report.append(f"Sample Size: {results.get('sample_size', 0)}")
        
        report_text = "\n".join(report)
        
        # Save or print report
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_file}")
        else:
            print(report_text)
        
        return report_text


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Agentic RAG System")
    parser.add_argument("--dataset", type=str, default="squad",
                       help="Dataset to evaluate on (squad, natural_questions, ms_marco)")
    parser.add_argument("--test-file", type=str, help="Path to custom test JSON file")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--metric", type=str, default="all",
                       help="Metrics to compute (f1, exact_match, retrieval_accuracy, all)")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2",
                       help="Model to use")
    parser.add_argument("--output", type=str, help="Output file for report")
    parser.add_argument("--use-agent", action="store_true", help="Use agentic reasoning")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    logger.info("Initializing RAG system...")
    rag = AgenticRAG(
        model_name=args.model,
        quantization="4bit",
        config={"temperature": 0.3}  # Lower temperature for evaluation
    )
    
    # Load dataset if specified
    if not args.test_file:
        logger.info(f"Loading {args.dataset} dataset...")
        rag.load_dataset(args.dataset, sample_size=1000)  # Load more for retrieval
    
    # Initialize evaluator
    evaluator = RAGEvaluator(rag)
    
    # Determine metrics
    if args.metric == "all":
        metrics = ["f1", "exact_match", "retrieval_accuracy", "latency", "confidence"]
    else:
        metrics = [args.metric]
    
    # Run evaluation
    if args.test_file:
        results = evaluator.evaluate_custom(args.test_file)
    else:
        results = evaluator.evaluate_dataset(
            args.dataset,
            sample_size=args.sample_size,
            metrics=metrics
        )
    
    # Generate report
    evaluator.generate_report(results, args.output)
    
    # Save full results
    results_file = args.output.replace('.txt', '_full.json') if args.output else 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Full results saved to {results_file}")


if __name__ == "__main__":
    main()
