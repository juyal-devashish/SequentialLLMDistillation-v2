import torch
import json
import os
import sys

# Ensure imports work whether run from root or KPOD dir
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from models.token_weighting import TokenWeightingModule
except ImportError:
    # Try relative import if running from parent
    sys.path.append(os.path.join(current_dir, '..'))
    from KPOD.models.token_weighting import TokenWeightingModule

def check_rationales():
    # Try multiple possible paths for rationales
    paths = [
        "./data/rationales/gsm8k_train_500_rationales.json",
        "../data/rationales/gsm8k_train_500_rationales.json",
        "KPOD/data/rationales/gsm8k_train_500_rationales.json"
    ]
    
    found_path = None
    for p in paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        print(f"❌ Rationales file not found in any of: {paths}")
        return False
    
    print(f"✓ Found rationales at: {found_path}")
    
    try:
        with open(found_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ JSON Decode Error: File might be corrupted or empty.")
        return False
    
    print(f"Found {len(data)} rationales.")
    
    # Check for empty rationales
    empty_count = 0
    short_count = 0
    none_count = 0
    
    # Check first 20 samples
    check_limit = min(len(data), 20)
    print(f"Checking first {check_limit} samples...")
    
    for i, item in enumerate(data[:check_limit]):
        if item is None:
             print(f"Sample {i} is None!")
             continue
             
        r = item.get('rationale', '')
        
        if r is None:
            none_count += 1
            print(f"Sample {i} rationale is None")
        elif not r:
            empty_count += 1
            print(f"Sample {i} rationale is empty string")
        elif len(r) < 10:
            short_count += 1
            print(f"Sample {i} rationale is very short: '{r}'")
            
    if empty_count > 0 or none_count > 0:
        print(f"❌ Found {empty_count} empty and {none_count} None rationales in first {check_limit} samples!")
        return False
        
    print("✓ Rationales look valid (non-empty).")
    return True

def check_model_forward():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    print(f"Testing model on {device}...")
    
    try:
        module = TokenWeightingModule(
            model_name="google/flan-t5-small", # Use small for quick test
            device=device
        )
        
        # Create a dummy batch
        q = ["What is 1+1?"]
        r = ["The answer is 2."] 
        a = ["2"]
        
        # Tokenize
        q_enc = module.tokenizer(q, return_tensors="pt", padding=True).to(device)
        r_enc = module.tokenizer(r, return_tensors="pt", padding=True).to(device)
        a_enc = module.tokenizer(a, return_tensors="pt", padding=True).to(device)
        
        loss, info = module(
            q_enc['input_ids'],
            q_enc['attention_mask'],
            r_enc['input_ids'],
            r_enc['attention_mask'],
            a_enc['input_ids']
        )
        
        print(f"Forward pass successful.")
        print(f"Total Loss: {loss.item()}")
        print(f"Lp (Prediction Loss): {info['lp'].item()}")
        print(f"Lm (Mask Loss): {info['lm'].item()}")
        
        if loss.item() == 0:
            print("⚠️ Total Loss is exactly 0! This is suspicious.")
        elif info['lp'].item() == 0:
             print("⚠️ Prediction Loss (Lp) is exactly 0! This is suspicious.")
        else:
            print("✓ Loss values seem reasonable (non-zero).")
            
    except Exception as e:
        print(f"❌ Model forward failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=== Checking Data ===")
    if check_rationales():
        print("\n=== Checking Model ===")
        check_model_forward()

