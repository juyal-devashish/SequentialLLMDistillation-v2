"""
Diagnose training issues
"""
import torch
from configs.config import get_config
from data.load_datasets import load_gsm8k, ReasoningDataset, load_rationales
from models.base_model import StudentModel
from models.token_weighting import TokenWeightingModule

device = "cuda" if torch.cuda.is_available() else "cpu"
config = get_config()
config.model.student_model_name = "google/flan-t5-large"

print("="*60)
print("DIAGNOSTIC: Checking Training Components")
print("="*60)

# Load data
print("\n1. Loading data...")
full_train, _, _ = load_gsm8k()
rationales = load_rationales("./data/rationales/gsm8k_train_500_rationales.json")

val_dataset = ReasoningDataset(
    questions=[full_train[i]['question'] for i in range(50)],
    answers=[full_train[i]['answer'] for i in range(50)],
    rationales=[rationales[i]['rationale'] for i in range(50)],
    split='val'
)

print(f"✓ Val dataset: {len(val_dataset)} samples")

# Check a sample
sample = val_dataset[0]
print(f"\nSample check:")
print(f"  Question: {sample['question'][:50]}...")
print(f"  Answer: {sample['answer']}")
print(f"  Has rationale: {sample.get('rationale') is not None}")
if sample.get('rationale'):
    print(f"  Rationale length: {len(sample['rationale'])} chars")

# Load models
print("\n2. Loading models...")
student_model = StudentModel(
    model_name=config.model.student_model_name,
    lora_r=config.model.lora_r_flan,  # Fixed: model.lora_r_flan not training.lora_r_flan
    device=device
)
print("✓ Student model loaded")

token_weighting_module = TokenWeightingModule(
    model_name=config.model.student_model_name,
    hidden_dim=config.model.weight_generator_hidden_dim,
    device=device
)

checkpoint = torch.load("./checkpoints/medium_token_weighting.pt")
token_weighting_module.weight_generator.load_state_dict(
    checkpoint['weight_generator_state_dict']
)
token_weighting_module.eval()
print("✓ Token weighting module loaded")

# Test forward pass
print("\n3. Testing forward pass...")
question = val_dataset[0]['question']
rationale = val_dataset[0]['rationale']
answer = val_dataset[0]['answer']

input_text = f"Q: {question} A:"
target_text = f"{rationale} Therefore, the answer is {answer}"

# Tokenize
input_enc = student_model.tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    max_length=256
).to(device)

target_enc = student_model.tokenizer(
    target_text,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

print(f"  Input shape: {input_enc.input_ids.shape}")
print(f"  Target shape: {target_enc.input_ids.shape}")

# Forward pass
with torch.no_grad():
    outputs = student_model(
        input_ids=input_enc.input_ids,
        attention_mask=input_enc.attention_mask,
        labels=target_enc.input_ids
    )
    
    print(f"  Output loss: {outputs.loss.item():.4f}")
    print(f"  Is NaN: {torch.isnan(outputs.loss).item()}")
    print(f"  Is Inf: {torch.isinf(outputs.loss).item()}")

# Test token weight extraction
print("\n4. Testing token weight extraction...")
rationale_enc = token_weighting_module.tokenizer(
    rationale,
    return_tensors="pt",
    truncation=True,
    max_length=512
).to(device)

with torch.no_grad():
    weights = token_weighting_module.get_token_weights(
        rationale_enc.input_ids,
        rationale_enc.attention_mask
    )
    
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights range: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
    print(f"  Has NaN: {torch.isnan(weights).any().item()}")
    print(f"  Has Inf: {torch.isinf(weights).any().item()}")

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60)