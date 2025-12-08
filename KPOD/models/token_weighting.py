"""
Token Weighting Module - Determines significance of tokens in rationales
Uses mask learning to identify keypoint tokens vs. reasoning-irrelevant tokens
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional


class WeightGenerator(nn.Module):
    """
    2-layer MLP that generates significance weights for tokens
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output weight in [0, 1]
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, seq_len, embedding_dim]
        Returns:
            weights: [batch_size, seq_len] - probability of NOT masking each token
        """
        weights = self.mlp(embeddings).squeeze(-1)  # [batch_size, seq_len]
        return weights


class TokenWeightingModule(nn.Module):
    """
    Complete token weighting module with mask learning
    Implements Equations 2-7 from the paper
    """
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        hidden_dim: int = 256,
        alpha: float = 0.5,
        gumbel_temperature: float = 1.0,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.alpha = alpha
        self.gumbel_temperature = gumbel_temperature
        
        # Load pre-trained model for embeddings and answer prediction
        # Using FlanT5-Large as specified in paper
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Embedding layer (from pre-trained model)
        from transformers import T5EncoderModel, T5ForConditionalGeneration
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        
        # Full model for answer prediction
        self.answer_predictor = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Freeze the pre-trained models initially
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.answer_predictor.parameters():
            param.requires_grad = False
        
        # Weight generator (trainable)
        embedding_dim = self.encoder.config.hidden_size
        self.weight_generator = WeightGenerator(embedding_dim, hidden_dim)
        
        self.to(device)
    
    def gumbel_softmax_sample(self, weights: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Gumbel-Softmax sampling for differentiable mask generation
        Implements Equation 4 from the paper
        
        Args:
            weights: [batch_size, seq_len] - probability of NOT masking
            hard: whether to use hard sampling (non-differentiable but discrete)
        Returns:
            masks: [batch_size, seq_len] - 0 = masked, 1 = not masked
        """
        # Create logits for binary decision (mask vs not mask)
        # logits[:, :, 0] = log(1 - weights) -> probability of masking
        # logits[:, :, 1] = log(weights) -> probability of not masking
        logits = torch.stack([
            torch.log(1 - weights + 1e-8),  # mask
            torch.log(weights + 1e-8)        # not mask
        ], dim=-1)  # [batch_size, seq_len, 2]
        
        # Gumbel-Softmax
        masks = F.gumbel_softmax(logits, tau=self.gumbel_temperature, hard=hard, dim=-1)
        
        # Return the "not mask" probability (index 1)
        return masks[:, :, 1]  # [batch_size, seq_len]
    
    def apply_mask_to_rationale(
        self,
        rationale_input_ids: torch.Tensor,
        masks: torch.Tensor,
        mask_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply masks to rationale tokens
        
        Args:
            rationale_input_ids: [batch_size, seq_len]
            masks: [batch_size, seq_len] - 0 = mask, 1 = keep
            mask_token_id: token ID to use for masking (default: pad_token_id)
        Returns:
            masked_input_ids: [batch_size, seq_len]
        """
        if mask_token_id is None:
            mask_token_id = self.tokenizer.pad_token_id
        
        # Convert soft masks to hard masks for token replacement
        hard_masks = (masks > 0.5).float()
        
        # Replace masked tokens with mask_token_id
        masked_input_ids = torch.where(
            hard_masks.bool(),
            rationale_input_ids,
            torch.full_like(rationale_input_ids, mask_token_id)
        )
        
        return masked_input_ids
    
    def get_token_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get token embeddings using encoder with self-attention
        Implements Equation 2 from the paper
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, seq_len, hidden_dim]
        """
        with torch.no_grad():  # Encoder is frozen
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        return embeddings
    
    def compute_answer_prediction_loss(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        rationale_input_ids: torch.Tensor,
        rationale_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute answer prediction loss Lp (Equation 5)
        Predicts answer from question + masked rationale prefixes
        
        Args:
            question_input_ids: [batch_size, q_len]
            question_attention_mask: [batch_size, q_len]
            rationale_input_ids: [batch_size, r_len]
            rationale_attention_mask: [batch_size, r_len]
            answer_input_ids: [batch_size, a_len]
            masks: [batch_size, r_len]
        Returns:
            loss: scalar
        """
        batch_size = question_input_ids.size(0)
        total_loss = 0.0
        num_prefixes = 0
        
        # Apply mask to rationale
        masked_rationale_ids = self.apply_mask_to_rationale(rationale_input_ids, masks)
        
        # Try different prefixes of the rationale (as in paper)
        # We'll sample a few prefixes for efficiency
        max_prefix_samples = min(5, rationale_input_ids.size(1))
        prefix_positions = torch.linspace(
            rationale_input_ids.size(1) // 2,
            rationale_input_ids.size(1),
            max_prefix_samples,
            dtype=torch.long
        )
        
        for k in prefix_positions:
            k = k.item()
            if k == 0:
                continue
            
            # Concatenate question + masked rationale prefix
            prefix_rationale = masked_rationale_ids[:, :k]
            prefix_attention = rationale_attention_mask[:, :k]
            
            # Combine question and rationale prefix
            combined_input_ids = torch.cat([question_input_ids, prefix_rationale], dim=1)
            combined_attention_mask = torch.cat([question_attention_mask, prefix_attention], dim=1)
            
            # Predict answer
            with torch.set_grad_enabled(True):
                outputs = self.answer_predictor(
                    input_ids=combined_input_ids,
                    attention_mask=combined_attention_mask,
                    labels=answer_input_ids
                )
                total_loss += outputs.loss
                num_prefixes += 1
        
        return total_loss / max(num_prefixes, 1)
    
    def compute_mask_ratio_loss(self, masks: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute mask ratio loss Lm (Equation 6)
        Encourages masking more tokens
        
        Args:
            masks: [batch_size, seq_len] - 1 = not masked, 0 = masked
            attention_mask: [batch_size, seq_len] - 1 = valid token
        Returns:
            loss: scalar - sum of non-masked tokens (we want to minimize this)
        """
        # Only count valid tokens (not padding)
        valid_masks = masks * attention_mask
        loss = valid_masks.sum()
        return loss
    
    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        rationale_input_ids: torch.Tensor,
        rationale_attention_mask: torch.Tensor,
        answer_input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of token weighting module
        
        Returns:
            total_loss: Combined loss (Equation 7)
            info: Dictionary with weights, masks, and individual losses
        """
        # Get token embeddings (Equation 2)
        embeddings = self.get_token_embeddings(rationale_input_ids, rationale_attention_mask)
        
        # Generate significance weights (Equation 3)
        weights = self.weight_generator(embeddings)  # [batch_size, seq_len]
        
        # Sample masks using Gumbel-Softmax (Equation 4)
        masks = self.gumbel_softmax_sample(weights)
        
        # Compute answer prediction loss (Equation 5)
        lp = self.compute_answer_prediction_loss(
            question_input_ids,
            question_attention_mask,
            rationale_input_ids,
            rationale_attention_mask,
            answer_input_ids,
            masks
        )
        
        # Compute mask ratio loss (Equation 6)
        lm = self.compute_mask_ratio_loss(masks, rationale_attention_mask)
        
        # Combined loss (Equation 7)
        total_loss = lp + self.alpha * lm
        
        info = {
            'weights': weights,
            'masks': masks,
            'lp': lp,
            'lm': lm,
            'total_loss': total_loss
        }
        
        return total_loss, info
    
    def get_token_weights(
        self,
        rationale_input_ids: torch.Tensor,
        rationale_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get token significance weights (inference mode)
        
        Args:
            rationale_input_ids: [batch_size, seq_len]
            rationale_attention_mask: [batch_size, seq_len]
        Returns:
            weights: [batch_size, seq_len] - significance weight for each token
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.get_token_embeddings(rationale_input_ids, rationale_attention_mask)
            weights = self.weight_generator(embeddings)
        return weights


# Example usage and testing
if __name__ == "__main__":
    print("Testing Token Weighting Module...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize module with small model for testing
    module = TokenWeightingModule(
        model_name="google/flan-t5-small",
        hidden_dim=128,
        alpha=0.5,
        device=device
    )
    
    # Test data
    question = "What is 2 + 3?"
    rationale = "First, we add 2 and 3. 2 + 3 = 5."
    answer = "5"
    
    # Tokenize
    question_enc = module.tokenizer(question, return_tensors="pt", padding=True)
    rationale_enc = module.tokenizer(rationale, return_tensors="pt", padding=True)
    answer_enc = module.tokenizer(answer, return_tensors="pt", padding=True)
    
    # Move to device
    question_input_ids = question_enc['input_ids'].to(device)
    question_attention_mask = question_enc['attention_mask'].to(device)
    rationale_input_ids = rationale_enc['input_ids'].to(device)
    rationale_attention_mask = rationale_enc['attention_mask'].to(device)
    answer_input_ids = answer_enc['input_ids'].to(device)
    
    print("\nForward pass...")
    loss, info = module(
        question_input_ids,
        question_attention_mask,
        rationale_input_ids,
        rationale_attention_mask,
        answer_input_ids
    )
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Answer Prediction Loss (Lp): {info['lp'].item():.4f}")
    print(f"Mask Ratio Loss (Lm): {info['lm'].item():.4f}")
    print(f"Token Weights: {info['weights'][0].cpu().numpy()}")
    print(f"Masks: {info['masks'][0].cpu().numpy()}")
    
    # Test weight extraction
    weights = module.get_token_weights(rationale_input_ids, rationale_attention_mask)
    print(f"\nExtracted weights shape: {weights.shape}")
    
    # Decode to see which tokens got what weights
    tokens = module.tokenizer.convert_ids_to_tokens(rationale_input_ids[0])
    weights_np = weights[0].cpu().numpy()
    
    print("\nToken-Weight pairs:")
    for token, weight in zip(tokens, weights_np):
        if token != '</s>':  # Skip padding
            print(f"  {token:20s}: {weight:.3f}")
    
    print("\n✓ Token Weighting Module test complete!")