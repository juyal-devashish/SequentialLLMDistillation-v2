"""
Training script for token weighting module
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
from typing import Dict, List


class TokenWeightingTrainer:
    """Trainer for token weighting module"""
    
    def __init__(
        self,
        module,
        train_dataset,
        val_dataset,
        config,
        device: str = "cuda"
    ):
        self.module = module
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for debugging
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer (only train weight generator)
        self.optimizer = AdamW(
            self.module.weight_generator.parameters(),
            lr=config.training.lr_flan,
            weight_decay=config.training.weight_decay
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _prepare_batch(self, batch):
        """Prepare batch for training"""
        # Handle different dataset structures or dict types
        if isinstance(batch, dict):
            questions = batch['question']
            answers = batch['answer']
            rationales = batch['rationale'] if 'rationale' in batch else []
        else:
            # If batch is a list of items (e.g. from custom collate or raw list)
            # Or if it's a huggingface dataset batch
            try:
                questions = batch['question']
                answers = batch['answer']
                rationales = batch['rationale']
            except (KeyError, TypeError):
                # Fallback for when attributes are direct lists (like in our truncated dataset)
                questions = [item['question'] for item in batch]
                answers = [item['answer'] for item in batch]
                rationales = [item['rationale'] for item in batch]
        
        # Tokenize
        question_enc = self.module.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.data.max_length
        )
        
        rationale_enc = self.module.tokenizer(
            rationales,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.data.max_rationale_length
        )
        
        answer_enc = self.module.tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )
        
        return {
            'question_input_ids': question_enc['input_ids'].to(self.device),
            'question_attention_mask': question_enc['attention_mask'].to(self.device),
            'rationale_input_ids': rationale_enc['input_ids'].to(self.device),
            'rationale_attention_mask': rationale_enc['attention_mask'].to(self.device),
            'answer_input_ids': answer_enc['input_ids'].to(self.device)
        }
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.module.train()
        # Keep encoder and answer_predictor in eval mode (they're frozen)
        self.module.encoder.eval()
        self.module.answer_predictor.eval()
        
        total_loss = 0
        total_lp = 0
        total_lm = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            model_inputs = self._prepare_batch(batch)
            
            # DEBUG: Check inputs for first batch
            if batch_idx == 0 and epoch == 1:
                print(f"\n[DEBUG] Batch 0 Inputs:")
                print(f"  Question shape: {model_inputs['question_input_ids'].shape}")
                print(f"  Rationale shape: {model_inputs['rationale_input_ids'].shape}")
                print(f"  Answer shape: {model_inputs['answer_input_ids'].shape}")
                
                # Check if rationales are non-empty
                r_ids = model_inputs['rationale_input_ids']
                pad_id = self.module.tokenizer.pad_token_id
                non_pads = (r_ids != pad_id).sum(dim=1)
                print(f"  Rationale non-padding count: {non_pads}")
            
            # Forward pass
            loss, info = self.module(**model_inputs)
            
            # DEBUG: Check loss
            if batch_idx == 0 and epoch == 1:
                print(f"  Loss: {loss.item()}")
                print(f"  Lp: {info['lp'].item()}")
                print(f"  Lm: {info['lm'].item()}")
                if loss.item() == 0:
                    print("  ⚠️ LOSS IS ZERO!")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.module.weight_generator.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            total_lp += info['lp'].item()
            total_lm += info['lm'].item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lp': info['lp'].item(),
                'lm': info['lm'].item()
            })
            
            # Early stopping for testing
            if batch_idx >= 10 and epoch == 1:  # Just for quick testing
                break
        
        avg_loss = total_loss / min(len(self.train_loader), batch_idx + 1)
        avg_lp = total_lp / min(len(self.train_loader), batch_idx + 1)
        avg_lm = total_lm / min(len(self.train_loader), batch_idx + 1)
        
        return avg_loss, avg_lp, avg_lm
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.module.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Evaluating")):
                model_inputs = self._prepare_batch(batch)
                loss, _ = self.module(**model_inputs)
                total_loss += loss.item()
                
                # Early stopping for testing
                if batch_idx >= 5:
                    break
        
        return total_loss / min(len(self.val_loader), batch_idx + 1)
    
    def train(self, num_epochs: int = 5):
        """Full training loop"""
        print(f"Training token weighting module for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss, train_lp, train_lm = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f} (Lp: {train_lp:.4f}, Lm: {train_lm:.4f})")
            
            # Evaluate
            val_loss = self.evaluate()
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(
                    self.config.checkpoint_dir,
                    "token_weighting_best.pt"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'weight_generator_state_dict': self.module.weight_generator.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")
        
        print("\nTraining complete!")


# Example usage
if __name__ == "__main__":
    from configs.config import get_config
    from models.token_weighting import TokenWeightingModule
    from data.load_datasets import load_gsm8k, attach_rationales_to_dataset, load_rationales
    
    # Get config
    config = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, _ = load_gsm8k()
    
    # Check if rationales exist
    rationale_path = "./data/rationales/gsm8k_train_rationales.json"
    if not os.path.exists(rationale_path):
        print(f"Rationale file not found: {rationale_path}")
        print("Please generate rationales first with: python main.py --mode generate --dataset gsm8k")
        exit(1)
    
    # Load and attach rationales
    rationales = load_rationales(rationale_path)
    train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
    
    # Note: For validation, we'd need validation rationales too
    # For now, we'll use a subset of training data
    val_rationales = rationales[:len(val_dataset)]
    val_dataset = attach_rationales_to_dataset(val_dataset, val_rationales)
    
    # Initialize module
    print("Initializing token weighting module...")
    module = TokenWeightingModule(
        model_name="google/flan-t5-small",  # Use small for testing
        hidden_dim=128,
        alpha=0.5,
        device=device
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = TokenWeightingTrainer(
        module=module,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )
    
    # Train
    trainer.train(num_epochs=3)