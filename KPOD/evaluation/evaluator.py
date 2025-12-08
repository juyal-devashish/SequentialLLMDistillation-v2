"""
Evaluation metrics for reasoning tasks
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List
import re
import json
import os


class ReasoningEvaluator:
    """Evaluate student model on reasoning tasks"""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.eval()
    
    def extract_answer(self, text: str, dataset_name: str = "gsm8k") -> str:
        """
        Extract answer from generated text
        
        Args:
            text: Generated text
            dataset_name: Name of dataset (affects answer extraction)
        Returns:
            Extracted answer
        """
        if dataset_name.lower() in ['gsm8k', 'asdiv', 'svamp']:
            # For math datasets, extract number after "the answer is"
            match = re.search(r'the answer is[:\s]+(-?\d+\.?\d*)', text.lower())
            if match:
                return match.group(1)
            
            # Fallback: look for last number
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                return numbers[-1]
            
            return ""
        
        elif dataset_name.lower() == 'commonsenseqa':
            # For CommonsenseQA, extract single letter answer
            match = re.search(r'the answer is[:\s]+([A-E])', text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            
            # Fallback: look for single capital letter
            match = re.search(r'\b([A-E])\b', text)
            if match:
                return match.group(1).upper()
            
            return ""
        
        else:
            return text.strip()
    
    def evaluate_sample(
        self,
        question: str,
        ground_truth: str,
        dataset_name: str = "gsm8k",
        max_length: int = 512
    ) -> Dict:
        """
        Evaluate a single sample
        
        Args:
            question: Question text
            ground_truth: Ground truth answer
            dataset_name: Dataset name
            max_length: Max generation length
        Returns:
            Result dictionary
        """
        # Prepare input
        input_text = f"Q: {question} A: Let's think step by step."
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=1,
                do_sample=False,
                early_stopping=True
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer
        predicted_answer = self.extract_answer(generated_text, dataset_name)
        
        # Check correctness
        if dataset_name.lower() in ['gsm8k', 'asdiv', 'svamp']:
            # Numeric comparison
            try:
                pred_num = float(predicted_answer)
                gt_num = float(ground_truth)
                correct = abs(pred_num - gt_num) < 1e-5
            except (ValueError, TypeError):
                correct = False
        else:
            # String comparison
            correct = predicted_answer.lower().strip() == ground_truth.lower().strip()
        
        return {
            'question': question,
            'ground_truth': ground_truth,
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'correct': correct
        }
    
    def evaluate_dataset(
        self,
        dataset,
        dataset_name: str = "gsm8k",
        batch_size: int = 1,
        max_samples: int = None,
        save_path: str = None
    ) -> Dict:
        """
        Evaluate on full dataset
        
        Args:
            dataset: Dataset to evaluate
            dataset_name: Dataset name
            batch_size: Batch size (keep at 1 for generation)
            max_samples: Max samples to evaluate
            save_path: Path to save detailed results
        Returns:
            Evaluation metrics
        """
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        results = []
        correct_count = 0
        
        for idx in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[idx]
            question = sample['question']
            answer = sample['answer']
            
            result = self.evaluate_sample(question, answer, dataset_name)
            results.append(result)
            
            if result['correct']:
                correct_count += 1
        
        # Calculate metrics
        accuracy = (correct_count / num_samples) * 100
        
        metrics = {
            'accuracy': accuracy,
            'correct': correct_count,
            'total': num_samples,
            'dataset': dataset_name
        }
        
        # Save detailed results if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            output = {
                'metrics': metrics,
                'results': results
            }
            
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"Detailed results saved to {save_path}")
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print evaluation metrics"""
        print(f"\n{'='*60}")
        print(f"Evaluation Results - {metrics['dataset']}")
        print(f"{'='*60}")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Correct: {metrics['correct']}/{metrics['total']}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    from models.base_model import StudentModel
    from data.load_datasets import load_gsm8k
    import os
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model_path = "./checkpoints/kpod_best_model"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please train a model first.")
        exit(1)
    
    print("Loading model...")
    model = StudentModel(
        model_name="google/flan-t5-small",
        lora_r=8,
        device=device
    )
    model.load_pretrained(model_path)
    
    # Load test set
    print("Loading test dataset...")
    _, _, test_dataset = load_gsm8k()
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = ReasoningEvaluator(
        model=model,
        tokenizer=model.tokenizer,
        device=device
    )
    
    # Evaluate
    print("Evaluating on test set...")
    metrics = evaluator.evaluate_dataset(
        dataset=test_dataset,
        dataset_name="gsm8k",
        max_samples=100,  # Evaluate on subset for testing
        save_path="./outputs/evaluation_results.json"
    )
    
    evaluator.print_metrics(metrics)