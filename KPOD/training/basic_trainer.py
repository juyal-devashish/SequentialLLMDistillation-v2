"""
Basic training loop for student model (standard CoT distillation without KPOD components)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional, Dict
import os


class BasicTrainer:
    """Basic trainer for CoT distillation"""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        device: str = "cuda"
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self._get_learning_rate(),
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        num_training_steps = len(self.train_loader) * self._get_num_epochs()
        num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _get_learning_rate(self):
        """Get learning rate based on model type"""
        if 'llama' in self.model.model_name.lower():
            return self.config.training.lr_llama
        else:
            return self.config.training.lr_flan
    
    def _get_num_epochs(self):
        """Get number of epochs based on dataset and model"""
        dataset_name = self.config.data.dataset_name.lower()
        
        if 'llama' in self.model.model_name.lower():
            if dataset_name == 'gsm8k':
                return self.config.training.epochs_gsm8k_llama
            elif dataset_name == 'commonsenseqa':
                return self.config.training.epochs_commonsense_llama
            else:
                return self.config.training.epochs_asdiv_llama
        else:
            return self.config.training.epochs_flan
    
    def _prepare_batch(self, batch):
        """Prepare batch for training"""
        questions = batch['question']
        rationales = batch['rationale']
        answers = batch['answer']
        
        # Format: Q: <question> A: <rationale> Therefore, the answer is <answer>
        inputs = []
        targets = []
        
        for q, r, a in zip(questions, rationales, answers):
            input_text = f"Q: {q} A:"
            target_text = f"{r} Therefore, the answer is {a}"
            
            inputs.append(input_text)
            targets.append(target_text)
        
        # Tokenize
        input_encodings = self.model.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_length,
            return_tensors='pt'
        )
        
        target_encodings = self.model.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_rationale_length,
            return_tensors='pt'
        )
        
        # Move to device
        return {
            'input_ids': input_encodings.input_ids.to(self.device),
            'attention_mask': input_encodings.attention_mask.to(self.device),
            'labels': target_encodings.input_ids.to(self.device)
        }
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            model_inputs = self._prepare_batch(batch)
            
            # Forward pass
            outputs = self.model(**model_inputs)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Tracking
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                model_inputs = self._prepare_batch(batch)
                outputs = self.model(**model_inputs)
                total_loss += outputs.loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Full training loop"""
        num_epochs = self._get_num_epochs()
        print(f"Training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            # Evaluate
            val_loss = self.evaluate()
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"best_model_epoch_{epoch}"
                )
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                print(f"Saved best model to {save_path}")
            
            # Regular checkpoint
            if epoch % 5 == 0:
                save_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}"
                )
                os.makedirs(save_path, exist_ok=True)
                self.model.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")


# Example usage
if __name__ == "__main__":
    from configs.config import get_config
    from models.base_model import StudentModel
    from data.load_datasets import load_gsm8k, attach_rationales_to_dataset, load_rationales
    
    # Get config
    config = get_config()
    config.data.dataset_name = "gsm8k"
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_gsm8k()
    
    # Note: You need to generate rationales first!
    # For testing, we'll skip this step
    print("Note: This example requires pre-generated rationales")
    print("Run generate_rationales.py first to create rationale files")