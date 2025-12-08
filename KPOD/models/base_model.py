"""
Base student model with LoRA
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from typing import Optional, Dict


class StudentModel(nn.Module):
    """Student model wrapper with LoRA"""
    
    def __init__(
        self,
        model_name: str,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.model_name = model_name
        
        # Smart device selection
        if device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                self.device = "mps"
                print(f"Using MPS (Mac GPU) for {model_name}")
            else:
                self.device = "cpu"
                print(f"CUDA not found, using CPU for {model_name}")
        else:
            self.device = device
            
        # Determine model type
        if 'llama' in model_name.lower():
            self.model_type = 'causal'
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            task_type = TaskType.CAUSAL_LM
        elif 't5' in model_name.lower() or 'flan' in model_name.lower():
            self.model_type = 'seq2seq'
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            task_type = TaskType.SEQ_2_SEQ_LM
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Move to device if not using device_map
        if self.device in ["cpu", "mps"]:
            self.model = self.model.to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=self._get_target_modules(),
            lora_dropout=lora_dropout,
            bias="none",
            task_type=task_type
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _get_target_modules(self):
        """Get target modules for LoRA based on model type"""
        if self.model_type == 'causal':
            # LLaMA
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
        else:
            # T5/Flan-T5
            return ["q", "v"]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 256,
        **kwargs
    ):
        """Generate text"""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            **kwargs
        )
    
    def prepare_inputs(self, text, max_length=512):
        """Prepare inputs and move to correct device"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        # Move to device
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def save_pretrained(self, save_path: str):
        """Save model"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_pretrained(self, load_path: str):
        """Load model"""
        self.model = PeftModel.from_pretrained(self.model, load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)


# Example usage
if __name__ == "__main__":
    print("Testing StudentModel...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Test with Flan-T5-Small (smaller for testing)
    model = StudentModel(
        model_name="google/flan-t5-small",
        lora_r=8,
        device=device
    )
    
    # Test forward pass
    test_input = "Q: What is 2 + 2? A: Let's think step by step."
    
    print("\nInput:", test_input)
    
    # Use prepare_inputs helper
    inputs = model.prepare_inputs(test_input)
    
    # Generate
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=100
    )
    
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated:", generated_text)
    
    print("\nModel setup successful!")