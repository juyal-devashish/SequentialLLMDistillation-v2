"""
Test basic setup without requiring API keys or heavy models
"""
import sys
import torch

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        import sklearn
        import peft
        print("✓ All required packages installed")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from configs.config import get_config
        config = get_config()
        print(f"✓ Config loaded")
        print(f"  - Student model: {config.model.student_model_name}")
        print(f"  - Device: {config.training.device}")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading"""
    print("\nTesting dataset loading...")
    try:
        from data.load_datasets import load_gsm8k
        train, val, test = load_gsm8k()
        print(f"✓ GSM8K loaded")
        print(f"  - Train size: {len(train)}")
        print(f"  - Val size: {len(val)}")
        print(f"  - Test size: {len(test)}")
        print(f"\n  Sample question: {train[0]['question'][:100]}...")
        return True
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False

def test_model_setup():
    """Test model initialization (with small model)"""
    print("\nTesting model setup (this may take a minute)...")
    try:
        from models.base_model import StudentModel
        
        # Use smallest T5 model for testing
        model = StudentModel(
            model_name="google/flan-t5-small",  # Smallest for testing
            lora_r=8,
            device="cpu"  # Force CPU to avoid MPS issues
        )
        print(f"✓ Model initialized successfully")
        
        # Test forward pass
        test_input = "Q: What is 2 + 2? A:"
        inputs = model.tokenizer(test_input, return_tensors="pt")
        
        # Ensure inputs are on CPU
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=50
            )
        
        generated = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  - Test generation: {generated}")
        return True
    except Exception as e:
        print(f"✗ Model setup error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("KPOD Basic Setup Test")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset Loading", test_dataset_loading),
        ("Model Setup", test_model_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python main.py --mode generate --dataset gsm8k")
        print("3. Run: python main.py --mode train --dataset gsm8k")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()