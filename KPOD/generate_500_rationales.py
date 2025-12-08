"""
Generate 500 rationales quickly using parallel API calls
"""
import os
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import json
import sys


class FastRationaleGenerator:
    """Generate rationales using parallel async API calls"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_concurrent: int = 15  # Process 15 at once
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_single_rationale(
        self,
        question: str,
        idx: int,
        max_retries: int = 3
    ) -> Dict[str, str]:
        """Generate rationale for a single question with retries"""
        
        prompt = f"Q: {question}\nA: Let's think step by step."
        
        async with self.semaphore:  # Limit concurrent requests
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that solves problems step by step."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=512,
                    )
                    
                    rationale = response.choices[0].message.content.strip()
                    
                    return {
                        'idx': idx,
                        'question': question,
                        'rationale': rationale,
                        'success': True,
                        'error': None
                    }
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        return {
                            'idx': idx,
                            'question': question,
                            'rationale': '',
                            'success': False,
                            'error': str(e)
                        }
    
    async def generate_batch(
        self,
        questions: List[tuple],  # List of (idx, question)
        desc: str = "Generating"
    ) -> List[Dict]:
        """Generate rationales for a batch of questions"""
        
        tasks = [
            self.generate_single_rationale(q, i) 
            for i, q in questions
        ]
        
        # Use async tqdm for progress
        results = []
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            result = await coro
            results.append(result)
        
        return results
    
    async def generate_rationales(
        self,
        questions: List[str],
        save_path: str = None,
        checkpoint_interval: int = 100,
        start_idx: int = 0
    ) -> List[Dict]:
        """Generate rationales for multiple questions in parallel with checkpoints"""
        
        all_rationales = []
        
        # Process in chunks for checkpointing
        for chunk_start in range(0, len(questions), checkpoint_interval):
            chunk_end = min(chunk_start + checkpoint_interval, len(questions))
            chunk_questions = [
                (start_idx + i, questions[i]) 
                for i in range(chunk_start, chunk_end)
            ]
            
            print(f"\nProcessing batch {chunk_start//checkpoint_interval + 1} "
                  f"(samples {chunk_start}-{chunk_end})...")
            
            batch_results = await self.generate_batch(
                chunk_questions,
                desc=f"Batch {chunk_start//checkpoint_interval + 1}"
            )
            
            all_rationales.extend(batch_results)
            
            # Save checkpoint
            if save_path:
                checkpoint_path = save_path.replace('.json', f'_checkpoint_{start_idx + chunk_end}.json')
                # Sort by index before saving
                sorted_rationales = sorted(all_rationales, key=lambda x: x['idx'])
                with open(checkpoint_path, 'w') as f:
                    json.dump(sorted_rationales, f, indent=2)
                print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Sort by original index
        all_rationales.sort(key=lambda x: x['idx'])
        
        # Final save
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(all_rationales, f, indent=2)
            print(f"\n✓ Final rationales saved: {save_path}")
        
        # Statistics
        success_count = sum(1 for r in all_rationales if r['success'])
        print(f"\n✓ Generation complete: {success_count}/{len(all_rationales)} successful")
        
        return all_rationales


async def main():
    """Generate 500 rationales"""
    from data.load_datasets import load_gsm8k
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    print("="*80)
    print("  Fast Rationale Generation for Medium-Scale Training")
    print("="*80)
    print(f"  Target: 500 samples")
    print(f"  Parallel requests: 15")
    print(f"  Estimated time: 20-30 minutes")
    print("="*80)
    
    # Load dataset
    print("\nLoading GSM8K dataset...")
    train_dataset, _, _ = load_gsm8k()
    
    # Check if we already have some rationales
    existing_rationales = []
    rationale_path = "./data/rationales/gsm8k_train_500_rationales.json"
    
    if os.path.exists(rationale_path):
        print(f"\n⚠️  Found existing rationales at {rationale_path}")
        response = input("Resume from checkpoint or start fresh? (r/s): ")
        if response.lower() == 'r':
            with open(rationale_path, 'r') as f:
                existing_rationales = json.load(f)
            print(f"✓ Loaded {len(existing_rationales)} existing rationales")
    
    # Determine how many to generate
    start_idx = len(existing_rationales)
    num_samples = 500
    
    if start_idx >= num_samples:
        print(f"\n✓ Already have {start_idx} rationales. Nothing to generate!")
        return
    
    questions = train_dataset.questions[start_idx:num_samples]
    
    print(f"\n📝 Generating rationales for samples {start_idx}-{num_samples}...")
    print(f"   ({len(questions)} samples to generate)\n")
    
    # Initialize generator
    generator = FastRationaleGenerator(
        api_key=api_key,
        max_concurrent=15  # Adjust based on API rate limits
    )
    
    # Generate
    new_rationales = await generator.generate_rationales(
        questions=questions,
        save_path=rationale_path,
        checkpoint_interval=100,
        start_idx=start_idx
    )
    
    # Combine with existing
    all_rationales = existing_rationales + new_rationales
    all_rationales.sort(key=lambda x: x['idx'])
    
    # Save final
    with open(rationale_path, 'w') as f:
        json.dump(all_rationales, f, indent=2)
    
    print("\n" + "="*80)
    print("  ✓ Rationale Generation Complete!")
    print("="*80)
    print(f"  Total rationales: {len(all_rationales)}")
    print(f"  Success rate: {sum(1 for r in all_rationales if r['success'])}/{len(all_rationales)}")
    print(f"  Saved to: {rationale_path}")
    print("="*80)
    
    print("\n📍 Next step: Run training with these rationales")
    print("   python train_medium_scale.py")


if __name__ == "__main__":
    asyncio.run(main())