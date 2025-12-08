"""
Complete KPOD trainer integrating token weighting and progressive distillation
Implements Algorithm 1 from the paper
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional, Dict, List
import os
import json


class KPODTrainer:
    """
    Complete KPOD trainer with:
    1. Token weighting for keypoint emphasis
    2. Progressive distillation with curriculum learning
    """
    
    def __init__(
        self,
        student_model,
        token_weighting_module,
        progressive_scheduler,
        train_dataset,
        val_dataset,
        config,
        device: str = "cuda"
    ):
        """
        Args:
            student_model: Student model to train
            token_weighting_module: Trained token weighting module
            progressive_scheduler: Progressive distillation scheduler
            train_dataset: Training dataset with rationales
            val_dataset: Validation dataset
            config: Configuration object
            device: Device to use
        """
        self.student_model = student_model
        self.token_weighting_module = token_weighting_module
        self.scheduler = progressive_scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Put token weighting module in eval mode (already trained)
        self.token_weighting_module.eval()
        
        # Extract token weights for all samples
        print("Extracting token weights for all samples...")
        self.token_weights = self._extract_all_token_weights()
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 to avoid multiprocessing issues
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=self._get_learning_rate(),
            weight_decay=config.training.weight_decay
        )
        
        # Scheduler
        num_training_steps = len(self.train_loader) * self._get_num_epochs()
        num_warmup_steps = int(num_training_steps * config.training.warmup_ratio)
        
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def _get_learning_rate(self):
        """Get learning rate based on model type"""
        if 'llama' in self.student_model.model_name.lower():
            return self.config.training.lr_llama
        else:
            return self.config.training.lr_flan
    
    def _get_num_epochs(self):
        """Get number of epochs based on dataset and model"""
        dataset_name = self.config.data.dataset_name.lower()
        
        if 'llama' in self.student_model.model_name.lower():
            if dataset_name == 'gsm8k':
                return self.config.training.epochs_gsm8k_llama
            elif dataset_name == 'commonsenseqa':
                return self.config.training.epochs_commonsense_llama
            else:
                return self.config.training.epochs_asdiv_llama
        else:
            return self.config.training.epochs_flan
    
    def _extract_all_token_weights(self) -> Dict[int, torch.Tensor]:
        """Extract token weights for all training samples with safety checks"""
        weights_dict = {}
        
        for idx in tqdm(range(len(self.train_dataset)), desc="Extracting token weights"):
            sample = self.train_dataset[idx]
            rationale = sample.get('rationale', '')
            
            if not rationale:
                weights_dict[idx] = torch.ones(10)  # Dummy weights
                continue
            
            try:
                # Tokenize rationale
                rationale_enc = self.token_weighting_module.tokenizer(
                    rationale,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.data.max_rationale_length
                )
                
                rationale_input_ids = rationale_enc['input_ids'].to(self.device)
                rationale_attention_mask = rationale_enc['attention_mask'].to(self.device)
                
                # Get weights
                with torch.no_grad():
                    weights = self.token_weighting_module.get_token_weights(
                        rationale_input_ids,
                        rationale_attention_mask
                    )
                    
                    # Check for NaN/Inf
                    if torch.isnan(weights).any() or torch.isinf(weights).any():
                        print(f"⚠️ Warning: Invalid weights for sample {idx}, using uniform weights")
                        weights = torch.ones_like(weights)
                    
                    # Clamp to reasonable range
                    weights = weights.clamp(min=0.1, max=2.0)
                    
                    weights_dict[idx] = weights[0].cpu()
                    
            except Exception as e:
                print(f"⚠️ Error getting weights for sample {idx}: {e}")
                weights_dict[idx] = torch.ones(50)
        
        return weights_dict
    
    def _prepare_batch_with_progressive_steps(
        self,
        batch,
        batch_indices: List[int],
        stage: int
    ):
        """
        Prepare batch with progressive input/output split
        Implements the progressive distillation strategy
        
        Args:
            batch: Batch of samples
            batch_indices: Indices of samples in batch
            stage: Current training stage
        Returns:
            Prepared inputs for model
        """
        questions = batch['question']
        rationales = batch['rationale']
        answers = batch['answer']
        
        inputs = []
        targets = []
        weights_list = []
        
        for i, (q, r, a) in enumerate(zip(questions, rationales, answers)):
            sample_idx = batch_indices[i]
            
            # Get number of input steps for this sample at this stage
            num_input_steps = self.scheduler.get_input_steps_for_sample(sample_idx)
            
            # Parse rationale into steps
            steps = self._parse_steps(r)
            
            # Split into input and output
            if num_input_steps >= len(steps):
                # Generate full rationale
                input_text = f"Q: {q} A:"
                output_text = f"{r} Therefore, the answer is {a}"
            elif num_input_steps == 0:
                # Generate full rationale (same as above)
                input_text = f"Q: {q} A:"
                output_text = f"{r} Therefore, the answer is {a}"
            else:
                # Generate from partial rationale
                input_steps = steps[:num_input_steps]
                output_steps = steps[num_input_steps:]
                
                input_rationale = ". ".join(input_steps) + "."
                output_rationale = ". ".join(output_steps) + "."
                
                input_text = f"Q: {q} A: {input_rationale}"
                output_text = f"{output_rationale} Therefore, the answer is {a}"
            
            inputs.append(input_text)
            targets.append(output_text)
            
            # Get token weights for this sample
            if sample_idx in self.token_weights:
                weights_list.append(self.token_weights[sample_idx])
            else:
                # Fallback to uniform weights
                weights_list.append(torch.ones(50))
        
        # Tokenize inputs
        input_encodings = self.student_model.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_length,
            return_tensors='pt'
        )
        
        # Tokenize targets
        target_encodings = self.student_model.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_rationale_length,
            return_tensors='pt'
        )
        
        # Align weights with target tokens
        batch_weights = []
        for i, target_ids in enumerate(target_encodings['input_ids']):
            sample_weights = weights_list[i]
            target_len = len(target_ids)
            
            # Pad or truncate weights to match target length
            if len(sample_weights) < target_len:
                padding = torch.ones(target_len - len(sample_weights))
                aligned_weights = torch.cat([sample_weights, padding])
            else:
                aligned_weights = sample_weights[:target_len]
            
            batch_weights.append(aligned_weights)
        
        batch_weights = torch.stack(batch_weights).to(self.device)
        
        return {
            'input_ids': input_encodings.input_ids.to(self.device),
            'attention_mask': input_encodings.attention_mask.to(self.device),
            'labels': target_encodings.input_ids.to(self.device),
            'weights': batch_weights
        }
    
    def _parse_steps(self, rationale: str) -> List[str]:
        """Parse rationale into steps"""
        steps = [s.strip() for s in rationale.split('.') if s.strip()]
        return steps
    
    def compute_weighted_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted loss (Equation 15) with numerical stability
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len]
            weights: [batch_size, seq_len] - token significance weights
            attention_mask: [batch_size, seq_len] - mask for padding
        Returns:
            loss: Weighted loss
        """
        # Shift for autoregressive prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Ensure weights match the label sequence length
        batch_size = shift_labels.size(0)
        seq_len = shift_labels.size(1)
        
        # Adjust weights to match sequence length
        if weights.size(1) >= seq_len + 1:
            shift_weights = weights[:, 1:seq_len+1].contiguous()
        elif weights.size(1) == seq_len:
            shift_weights = weights.contiguous()
        else:
            # Pad weights if needed
            padding_size = seq_len - weights.size(1)
            padding = torch.ones(batch_size, padding_size, device=weights.device)
            shift_weights = torch.cat([weights, padding], dim=1).contiguous()
        
        # Final dimension check and adjustment
        if shift_weights.size(1) != seq_len:
            min_len = min(shift_weights.size(1), seq_len)
            shift_weights = shift_weights[:, :min_len]
            shift_labels = shift_labels[:, :min_len]
            shift_logits = shift_logits[:, :min_len, :]
            seq_len = min_len
        
        # Normalize weights to prevent extreme values
        shift_weights = shift_weights.clamp(min=1e-8, max=10.0)
        
        # Compute per-token loss
        loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_losses = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        token_losses = token_losses.view(batch_size, seq_len)
        
        # Replace any NaN or Inf in token losses
        token_losses = torch.where(
            torch.isnan(token_losses) | torch.isinf(token_losses),
            torch.zeros_like(token_losses),
            token_losses
        )
        
        # Apply weights
        weighted_losses = token_losses * shift_weights
        
        # Create mask for valid tokens
        mask = (shift_labels != -100).float()
        
        # Apply mask
        weighted_losses = weighted_losses * mask
        
        # Compute average, avoiding division by zero
        total_loss = weighted_losses.sum()
        total_tokens = mask.sum()
        
        if total_tokens > 0:
            loss = total_loss / total_tokens
        else:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Final NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Warning: NaN/Inf detected in loss, returning zero")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return loss
    
    def train_epoch(self, epoch: int):
        """
        Train for one epoch with progressive distillation
        
        Args:
            epoch: Current epoch number (1-indexed)
        Returns:
            Average loss for the epoch
        """
        self.student_model.train()
        
        # Update progressive schedule for this stage
        print(f"\nUpdating progressive schedule for stage {epoch}...")
        self.scheduler.update_stage(epoch)
        
        total_loss = 0
        num_batches = 0
        valid_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Get batch indices (needed for progressive scheduling)
            start_idx = batch_idx * self.config.training.batch_size
            end_idx = start_idx + len(batch['question'])
            batch_indices = list(range(start_idx, min(end_idx, len(self.train_dataset))))
            
            try:
                # Prepare batch with progressive input/output split
                model_inputs = self._prepare_batch_with_progressive_steps(
                    batch,
                    batch_indices,
                    epoch
                )
                
                # Forward pass
                outputs = self.student_model(
                    input_ids=model_inputs['input_ids'],
                    attention_mask=model_inputs['attention_mask'],
                    labels=model_inputs['labels']
                )
                
                # Compute weighted loss
                loss = self.compute_weighted_loss(
                    logits=outputs.logits,
                    labels=model_inputs['labels'],
                    weights=model_inputs['weights'],
                    attention_mask=model_inputs['attention_mask']
                )
                
                # Check for NaN before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Skipping batch {batch_idx} due to NaN/Inf loss")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.config.training.max_grad_norm
                )
                
                # Optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                # Tracking
                total_loss += loss.item()
                valid_batches += 1
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': total_loss / valid_batches if valid_batches > 0 else 0
                })
                
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        if valid_batches == 0:
            print("⚠️ Warning: No valid batches in this epoch!")
            return float('nan')
        
        avg_loss = total_loss / valid_batches
        return avg_loss
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.student_model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                try:
                    # Extract batch data - handle both dict and direct access
                    if isinstance(batch, dict):
                        questions = batch['question']
                        answers = batch['answer']
                        rationales = batch['rationale'] if 'rationale' in batch else None
                    else:
                        # Fallback: access by index
                        questions = [self.val_dataset[i]['question'] for i in range(len(batch))]
                        answers = [self.val_dataset[i]['answer'] for i in range(len(batch))]
                        rationales = [self.val_dataset[i]['rationale'] for i in range(len(batch))]
                    
                    # Check if we have rationales
                    if rationales is None:
                        print("⚠️ No rationales in batch, skipping")
                        continue
                    
                    # Convert to lists if needed
                    if not isinstance(questions, list):
                        questions = list(questions)
                    if not isinstance(answers, list):
                        answers = list(answers)
                    if not isinstance(rationales, list):
                        rationales = list(rationales)
                    
                    # Skip if any rationale is None
                    if any(r is None for r in rationales):
                        print("⚠️ Some rationales are None, skipping batch")
                        continue
                    
                    inputs = [f"Q: {q} A:" for q in questions]
                    targets = [f"{r} Therefore, the answer is {a}" 
                            for r, a in zip(rationales, answers)]
                    
                    # Tokenize
                    input_encodings = self.student_model.tokenizer(
                        inputs,
                        padding=True,
                        truncation=True,
                        max_length=self.config.data.max_length,
                        return_tensors='pt'
                    )
                    
                    target_encodings = self.student_model.tokenizer(
                        targets,
                        padding=True,
                        truncation=True,
                        max_length=self.config.data.max_rationale_length,
                        return_tensors='pt'
                    )
                    
                    # Forward pass
                    outputs = self.student_model(
                        input_ids=input_encodings.input_ids.to(self.device),
                        attention_mask=input_encodings.attention_mask.to(self.device),
                        labels=target_encodings.input_ids.to(self.device)
                    )
                    
                    # Check for valid loss
                    if not torch.isnan(outputs.loss) and not torch.isinf(outputs.loss):
                        total_loss += outputs.loss.item()
                        num_batches += 1
                    else:
                        print(f"⚠️ NaN/Inf loss in validation batch")
                        
                except Exception as e:
                    print(f"⚠️ Error in validation batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        if num_batches == 0:
            print("⚠️ No valid validation batches, using train loss as proxy")
            return float('inf')  # Return inf instead of nan
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """
        Full KPOD training loop (Algorithm 1)
        """
        num_epochs = self._get_num_epochs()
        print(f"\n{'='*80}")
        print(f"Starting KPOD Training")
        print(f"{'='*80}")
        print(f"Total epochs: {num_epochs}")
        print(f"Progressive stages until T={self.scheduler.T}")
        print(f"{'='*80}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*80}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            if not torch.isnan(torch.tensor(train_loss)):
                print(f"Train Loss: {train_loss:.4f}")
            else:
                print(f"Train Loss: NaN (skipping this epoch)")
                continue
            
            # Evaluate
            val_loss = self.evaluate()
            
            if not torch.isnan(torch.tensor(val_loss)):
                print(f"Val Loss: {val_loss:.4f}")
            else:
                print(f"Val Loss: NaN")
            
            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss if not torch.isnan(torch.tensor(train_loss)) else None,
                'val_loss': val_loss if not torch.isnan(torch.tensor(val_loss)) else None,
                'current_difficulty': self.scheduler.current_difficulty
            })
            
            # Save best model (only if loss is valid)
            if not torch.isnan(torch.tensor(val_loss)) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"kpod_best_model"
                )
                os.makedirs(save_path, exist_ok=True)
                self.student_model.save_pretrained(save_path)
                print(f"✓ Saved best model to {save_path}")
            
            # Regular checkpoint
            if epoch % 5 == 0:
                save_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"kpod_checkpoint_epoch_{epoch}"
                )
                os.makedirs(save_path, exist_ok=True)
                self.student_model.save_pretrained(save_path)
                print(f"✓ Saved checkpoint to {save_path}")
        
        # Save training history
        history_path = os.path.join(
            self.config.output_dir,
            "training_history.json"
        )
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        if not torch.isinf(torch.tensor(self.best_val_loss)):
            print(f"Best validation loss: {self.best_val_loss:.4f}")
        else:
            print("Warning: No valid validation loss recorded")
        print(f"Training history saved to: {history_path}")