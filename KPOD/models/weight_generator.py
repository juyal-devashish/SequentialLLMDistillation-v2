"""
Weight Generator for Token Significance
Implements the rationale token weighting module from Section 3.3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class WeightGenerator(nn.Module):
    """
    Generates significance weights for rationale tokens using mask learning.
    
    Architecture:
    - Input embedding layer (from pretrained model)
    - Self-attention layer
    - 2-layer MLP for weight generation
    """
    
    def __init__(
        self,
        embedding_dim: int = 1024,  # Flan-T5-Large hidden size
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Self-attention layer for encoding in-context information (Eq. 2)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(embedding_dim)
        
        # Weight generator: 2-layer MLP (Eq. 3)
        layers = []
        input_dim = embedding_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer outputs 1 dimension (weight for each token)
                layers.extend([
                    nn.Linear(input_dim, 1),
                    nn.Sigmoid()  # Output probability of NOT masking
                ])
            else:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                input_dim = hidden_dim
        
        self.weight_generator = nn.Sequential(*layers)
        
    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate significance weights for tokens.
        
        Args:
            token_embeddings: [batch_size, seq_len, embedding_dim]
            attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for padding)
            
        Returns:
            weights: [batch_size, seq_len] - probability of NOT masking each token
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Self-attention encoding (Eq. 2)
        # Convert attention_mask to format expected by MultiheadAttention
        if attention_mask is not None:
            # attention_mask: [batch_size, seq_len]
            # We need key_padding_mask where True = ignore
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        attn_output, _ = self.self_attention(
            token_embeddings,
            token_embeddings,
            token_embeddings,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection and normalization
        encoded_embeddings = self.attention_norm(token_embeddings + attn_output)
        
        # Generate weights for each token (Eq. 3)
        # weights shape: [batch_size, seq_len, 1]
        weights = self.weight_generator(encoded_embeddings)
        
        # Squeeze to [batch_size, seq_len]
        weights = weights.squeeze(-1)
        
        # Apply mask to set padding weights to 0
        if attention_mask is not None:
            weights = weights * attention_mask.float()
        
        return weights


class GumbelSoftmaxSampler:
    """
    Implements Gumbel-Softmax sampling for differentiable discrete sampling (Eq. 4)
    """
    
    @staticmethod
    def sample(
        weights: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False
    ) -> torch.Tensor:
        """
        Sample binary masks using Gumbel-Softmax.
        
        Args:
            weights: [batch_size, seq_len] - probability of NOT masking
            temperature: Gumbel-Softmax temperature
            hard: If True, return hard samples (0 or 1)
            
        Returns:
            masks: [batch_size, seq_len] - 1 = not masked, 0 = masked
        """
        batch_size, seq_len = weights.shape
        
        # Create logits for [mask, not_mask]
        # logits shape: [batch_size, seq_len, 2]
        logits = torch.stack([
            torch.log(1 - weights + 1e-10),  # Log probability of masking
            torch.log(weights + 1e-10)        # Log probability of NOT masking
        ], dim=-1)
        
        # Gumbel-Softmax sampling
        masks = F.gumbel_softmax(
            logits,
            tau=temperature,
            hard=hard,
            dim=-1
        )
        
        # Return the "not masked" probability (index 1)
        # masks shape: [batch_size, seq_len]
        return masks[..., 1]


def apply_mask_to_sequence(
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    mask_token_id: int
) -> torch.Tensor:
    """
    Apply masks to input sequence.
    
    Args:
        input_ids: [batch_size, seq_len]
        masks: [batch_size, seq_len] - 1 = keep, 0 = mask
        mask_token_id: Token ID to use for masking
        
    Returns:
        masked_input_ids: [batch_size, seq_len]
    """
    # Where mask is 0, replace with mask_token_id
    masked_input_ids = torch.where(
        masks.bool(),
        input_ids,
        torch.full_like(input_ids, mask_token_id)
    )
    
    return masked_input_ids


# Test the module
if __name__ == "__main__":
    print("Testing WeightGenerator...")
    
    # Test parameters
    batch_size = 2
    seq_len = 20
    embedding_dim = 1024
    
    # Create random embeddings
    token_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 15:] = 0  # Last 5 tokens are padding
    
    # Initialize weight generator
    weight_gen = WeightGenerator(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        num_layers=2
    )
    
    # Generate weights
    weights = weight_gen(token_embeddings, attention_mask)
    print(f"Weights shape: {weights.shape}")
    print(f"Weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
    print(f"Sample weights: {weights[0, :10]}")
    
    # Test Gumbel-Softmax sampling
    sampler = GumbelSoftmaxSampler()
    masks_soft = sampler.sample(weights, temperature=1.0, hard=False)
    masks_hard = sampler.sample(weights, temperature=1.0, hard=True)
    
    print(f"\nSoft masks shape: {masks_soft.shape}")
    print(f"Hard masks shape: {masks_hard.shape}")
    print(f"Sample soft masks: {masks_soft[0, :10]}")
    print(f"Sample hard masks: {masks_hard[0, :10]}")
    
    # Test masking
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    masked_ids = apply_mask_to_sequence(input_ids, masks_hard, mask_token_id=0)
    print(f"\nOriginal IDs: {input_ids[0, :10]}")
    print(f"Masked IDs: {masked_ids[0, :10]}")
    print(f"Masks: {masks_hard[0, :10]}")
    
    print("\n✓ WeightGenerator test passed!")