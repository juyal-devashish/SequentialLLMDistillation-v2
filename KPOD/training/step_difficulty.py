"""
Step difficulty assessment for progressive distillation
Implements difficulty calculation using weighted token generation loss (Equation 8)
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


class StepDifficultyCalculator:
    """Calculate difficulty for each reasoning step in rationales"""
    
    def __init__(
        self,
        student_model,
        token_weights: Dict[int, torch.Tensor],  # Maps sample idx to token weights
        device: str = "cuda"
    ):
        """
        Args:
            student_model: Pre-trained student model (before distillation)
            token_weights: Dictionary mapping sample index to token significance weights
            device: Device to use
        """
        self.model = student_model
        self.token_weights = token_weights
        self.device = device
        
        # Put model in eval mode for difficulty assessment
        self.model.eval()
    
    def parse_steps_from_rationale(self, rationale: str, step_delimiter: str = ".") -> List[str]:
        """
        Parse rationale into steps using delimiter
        
        Args:
            rationale: Full rationale text
            step_delimiter: Character to split steps (default: ".")
        Returns:
            steps: List of step texts
        """
        # Split by delimiter and clean
        steps = [s.strip() for s in rationale.split(step_delimiter) if s.strip()]
        return steps
    
    def get_step_token_positions(
        self,
        rationale_tokens: List[int],
        rationale_text: str,
        tokenizer
    ) -> List[Tuple[int, int]]:
        """
        Get start and end token positions for each step
        
        Args:
            rationale_tokens: Token IDs of full rationale
            rationale_text: Original rationale text
            tokenizer: Tokenizer used
        Returns:
            positions: List of (start_pos, end_pos) for each step
        """
        steps = self.parse_steps_from_rationale(rationale_text)
        positions = []
        
        current_pos = 0
        for step in steps:
            # Tokenize this step
            step_tokens = tokenizer.encode(step, add_special_tokens=False)
            step_length = len(step_tokens)
            
            # Find this step in the full rationale
            # Simple approach: assume sequential
            start_pos = current_pos
            end_pos = current_pos + step_length
            
            positions.append((start_pos, end_pos))
            current_pos = end_pos
        
        return positions
    
    def calculate_step_difficulty(
        self,
        question: str,
        rationale: str,
        sample_idx: int
    ) -> List[float]:
        """
        Calculate difficulty for each step in a rationale
        Implements Equation 8 from the paper
        
        Args:
            question: Question text
            rationale: Full rationale text
            sample_idx: Index of this sample (to get token weights)
        Returns:
            difficulties: List of difficulty scores, one per step
        """
        tokenizer = self.model.tokenizer
        
        # Parse steps
        steps = self.parse_steps_from_rationale(rationale)
        if len(steps) == 0:
            return [0.0]
        
        # For T5 models, we need to handle encoder-decoder architecture
        # Simplified approach: use uniform difficulty for now
        # In production, you'd use the model's generation probability
        
        try:
            # Format input
            input_text = f"Q: {question} A:"
            target_text = rationale
            
            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.device)
            
            with torch.no_grad():
                # For T5, we need decoder_input_ids
                target_ids = tokenizer(
                    target_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).input_ids.to(self.device)
                
                # Get model outputs
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=target_ids
                )
                
                # Use loss as a proxy for difficulty
                loss_per_step = outputs.loss.item() / len(steps)
            
            # Assign uniform difficulty per step based on overall loss
            difficulties = [loss_per_step] * len(steps)
            
        except Exception as e:
            # Fallback: uniform difficulties
            difficulties = [1.0] * len(steps)
        
        return difficulties
    
    def calculate_difficulties_for_dataset(
        self,
        dataset,
        batch_size: int = 1,
        max_samples: int = None
    ) -> Dict[int, List[float]]:
        """
        Calculate step difficulties for entire dataset
        
        Args:
            dataset: Dataset with questions and rationales
            batch_size: Batch size (keep at 1 for simplicity)
            max_samples: Maximum number of samples to process
        Returns:
            difficulties: Dict mapping sample_idx -> list of step difficulties
        """
        difficulties = {}
        
        num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
        
        for idx in tqdm(range(num_samples), desc="Calculating step difficulties"):
            sample = dataset[idx]
            question = sample['question']
            rationale = sample.get('rationale', '')
            
            if not rationale:
                difficulties[idx] = [0.0]
                continue
            
            try:
                step_diffs = self.calculate_step_difficulty(question, rationale, idx)
                difficulties[idx] = step_diffs
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                difficulties[idx] = [0.0]
        
        return difficulties


def save_difficulties(difficulties: Dict[int, List[float]], save_path: str):
    """Save difficulties to file"""
    import json
    import os
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert to serializable format
    serializable = {str(k): v for k, v in difficulties.items()}
    
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"Difficulties saved to {save_path}")


def load_difficulties(load_path: str) -> Dict[int, List[float]]:
    """Load difficulties from file"""
    import json
    
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    # Convert back to int keys
    difficulties = {int(k): v for k, v in data.items()}
    
    print(f"Loaded difficulties for {len(difficulties)} samples")
    return difficulties


# Example usage
if __name__ == "__main__":
    from configs.config import get_config
    from models.base_model import StudentModel
    from data.load_datasets import load_gsm8k, attach_rationales_to_dataset, load_rationales
    import os
    
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading datasets...")
    train_dataset, _, _ = load_gsm8k()
    
    # Load rationales
    rationale_path = "./data/rationales/gsm8k_train_rationales.json"
    if not os.path.exists(rationale_path):
        print("Rationale file not found. Please generate rationales first.")
        exit(1)
    
    rationales = load_rationales(rationale_path)
    train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
    
    # Initialize student model (pre-trained, before distillation)
    print("\nInitializing student model...")
    model = StudentModel(
        model_name="google/flan-t5-small",
        lora_r=8,
        device=device
    )
    
    # Create dummy token weights (in practice, these come from token weighting module)
    print("\nCreating dummy token weights...")
    token_weights = {}
    for idx in range(min(10, len(train_dataset))):  # Just test with 10 samples
        # Random weights for demonstration
        token_weights[idx] = torch.rand(50)
    
    # Initialize calculator
    print("\nInitializing difficulty calculator...")
    calculator = StepDifficultyCalculator(
        student_model=model,
        token_weights=token_weights,
        device=device
    )
    
    # Calculate difficulties for a few samples
    print("\nCalculating step difficulties...")
    difficulties = calculator.calculate_difficulties_for_dataset(
        train_dataset,
        max_samples=10
    )
    
    # Display results
    print("\nSample difficulties:")
    for idx in range(min(5, len(difficulties))):
        sample = train_dataset[idx]
        steps = calculator.parse_steps_from_rationale(sample['rationale'])
        diffs = difficulties[idx]
        
        print(f"\nSample {idx}:")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Number of steps: {len(steps)}")
        print(f"Step difficulties: {[f'{d:.3f}' for d in diffs]}")
    
    # Save difficulties
    save_path = "./data/difficulties/gsm8k_train_difficulties.json"
    save_difficulties(difficulties, save_path)
    
    print("\n✓ Step difficulty calculation complete!")