"""
Configuration file for KPOD implementation
"""
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model-related configurations"""
    # Student model options: 'llama-7b', 'flan-t5-xl', 'flan-t5-large'
    student_model_name: str = "google/flan-t5-large"
    teacher_model_name: str = "gpt-3.5-turbo"
    
    # LoRA configurations
    lora_r_llama: int = 64
    lora_r_flan: int = 128
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Weight generator
    weight_generator_hidden_dim: int = 256
    weight_generator_num_layers: int = 2


@dataclass
class TrainingConfig:
    """Training-related configurations"""
    # Learning rates
    lr_llama: float = 1e-5
    lr_flan: float = 5e-5
    
    # Batch size and epochs
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    
    # Epochs for different datasets
    epochs_gsm8k_llama: int = 20
    epochs_commonsense_llama: int = 20
    epochs_asdiv_llama: int = 40
    epochs_svamp_llama: int = 40
    epochs_flan: int = 100
    
    # Optimizer
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42


@dataclass
class TokenWeightingConfig:
    """Token weighting module configurations"""
    alpha: float = 0.5  # Balance between Lp and Lm
    gumbel_temperature: float = 1.0
    gumbel_hard: bool = False  # Use hard sampling or soft


@dataclass
class ProgressiveDistillationConfig:
    """Progressive distillation configurations"""
    p: float = 0.5  # Growth rate exponent
    c0_ratio: float = 0.3  # Initial difficulty as ratio of max
    beta: float = 12.0  # Diversity weight
    num_clusters: int = 5
    delta_s: int = 1  # Number of steps to reduce per stage
    
    # T will be set as half of total epochs dynamically


@dataclass
class DataConfig:
    """Data-related configurations"""
    dataset_name: str = "gsm8k"  # Options: gsm8k, asdiv, svamp, commonsenseqa
    max_length: int = 512
    max_rationale_length: int = 256
    cache_dir: str = "./data/cache"
    rationale_dir: str = "./data/rationales"


@dataclass
class Config:
    """Master configuration - using field(default_factory=...) for mutable defaults"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    token_weighting: TokenWeightingConfig = field(default_factory=TokenWeightingConfig)
    progressive: ProgressiveDistillationConfig = field(default_factory=ProgressiveDistillationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    
    # OpenAI API
    openai_api_key: Optional[str] = None  # Set via environment variable
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500


def get_config():
    """Get default configuration"""
    return Config()