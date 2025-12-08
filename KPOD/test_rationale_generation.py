"""
Test rationale generation with a small sample
"""
import os
import sys
from configs.config import get_config
from data.load_datasets import load_gsm8k
from data.generate_rationales import RationaleGenerator

def main():
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)
    
    print("Testing rationale generation with 5 samples...\n")
    print(f"API Key found: {api_key[:20]}...")
    
    # Load dataset
    config = get_config()
    train_dataset, _, _ = load_gsm8k(config.data.cache_dir)
    
    # Get 5 samples
    sample_questions = train_dataset.questions[:5]
    
    print(f"\nGenerating rationales for {len(sample_questions)} questions...")
    print("This will cost approximately $0.02\n")
    
    # Initialize generator
    generator = RationaleGenerator(api_key)
    
    # Generate rationales
    rationales = generator.generate_rationales(
        questions=sample_questions,
        batch_delay=1.0  # 1 second between requests
    )
    
    # Display results
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    
    for i, result in enumerate(rationales):
        print(f"\nSample {i+1}:")
        print(f"Question: {result['question'][:100]}...")
        print(f"\nRationale: {result['rationale'][:200]}...")
        print(f"Success: {result['success']}")
        if result['error']:
            print(f"Error: {result['error']}")
        print("-" * 80)
    
    # Save to file
    os.makedirs(config.data.rationale_dir, exist_ok=True)
    save_path = os.path.join(config.data.rationale_dir, "test_rationales.json")
    
    import json
    with open(save_path, 'w') as f:
        json.dump(rationales, f, indent=2)
    
    print(f"\n✓ Test rationales saved to {save_path}")
    print(f"✓ Estimated cost: ~$0.02")
    
    success_count = sum(1 for r in rationales if r['success'])
    if success_count == len(rationales):
        print(f"\n✓ All {success_count} rationales generated successfully!")
        print("\nYou can now generate rationales for the full dataset with:")
        print("  python main.py --mode generate --dataset gsm8k")
        print("\nNote: Full dataset will:")
        print("  - Generate 7,473 rationales")
        print("  - Take ~2-3 hours with rate limiting")
        print("  - Cost approximately $15-30")
    else:
        print(f"\n⚠ Only {success_count}/{len(rationales)} rationales succeeded.")
        print("Check your API key and connection.")

if __name__ == "__main__":
    main()