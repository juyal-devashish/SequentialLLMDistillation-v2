"""
Generate rationales for full GSM8K dataset
Uses parallel processing for speed
"""
import os
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict
from tqdm.asyncio import tqdm as async_tqdm
import json
import sys


class FastRationaleGenerator:
    """Fast parallel rationale generation"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", max_concurrent: int = 20):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_single_rationale(self, question: str, idx: int, max_retries: int = 3) -> Dict:
        prompt = f"Q: {question}\nA: Let's think step by step."
        
        async with self.semaphore:
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that solves math problems step by step."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=512,
                    )
                    
                    rationale = response.choices[0].message.content.strip()
                    return {'idx': idx, 'question': question, 'rationale': rationale, 'success': True, 'error': None}
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    else:
                        return {'idx': idx, 'question': question, 'rationale': '', 'success': False, 'error': str(e)}
    
    async def generate_batch(self, questions: List[tuple], desc: str = "Generating") -> List[Dict]:
        tasks = [self.generate_single_rationale(q, i) for i, q in questions]
        results = []
        for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            result = await coro
            results.append(result)
        return results
    
    async def generate_rationales(
        self,
        questions: List[str],
        save_path: str = None,
        checkpoint_interval: int = 500,
        start_idx: int = 0
    ) -> List[Dict]:
        all_rationales = []
        
        for chunk_start in range(0, len(questions), checkpoint_interval):
            chunk_end = min(chunk_start + checkpoint_interval, len(questions))
            chunk_questions = [(start_idx + i, questions[i]) for i in range(chunk_start, chunk_end)]
            
            print(f"\nProcessing batch {chunk_start//checkpoint_interval + 1} (samples {chunk_start}-{chunk_end})...")
            
            batch_results = await self.generate_batch(chunk_questions, desc=f"Batch {chunk_start//checkpoint_interval + 1}")
            all_rationales.extend(batch_results)
            
            # Save checkpoint
            if save_path:
                checkpoint_path = save_path.replace('.json', f'_checkpoint_{start_idx + chunk_end}.json')
                sorted_rationales = sorted(all_rationales, key=lambda x: x['idx'])
                with open(checkpoint_path, 'w') as f:
                    json.dump(sorted_rationales, f, indent=2)
                print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Sort and save final
        all_rationales.sort(key=lambda x: x['idx'])
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(all_rationales, f, indent=2)
            print(f"\n✓ Final rationales saved: {save_path}")
        
        success_count = sum(1 for r in all_rationales if r['success'])
        print(f"\n✓ Generation complete: {success_count}/{len(all_rationales)} successful")
        
        return all_rationales


async def main():
    from data.load_datasets import load_gsm8k
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    print("="*80)
    print("  Full GSM8K Rationale Generation")
    print("="*80)
    print(f"  Model: GPT-4o-mini")
    print(f"  Target: 7,473 samples")
    print(f"  Parallel requests: 20")
    print(f"  Estimated time: 2-3 hours")
    print(f"  Estimated cost: ~$3-5")
    print("="*80)
    
    # Load dataset
    print("\nLoading GSM8K dataset...")
    train_dataset, _, _ = load_gsm8k()
    questions = train_dataset.questions
    
    print(f"✓ Loaded {len(questions)} questions")
    
    # Check for existing
    save_path = "./data/rationales/gsm8k_full_train_rationales.json"
    existing_rationales = []
    
    if os.path.exists(save_path):
        print(f"\n⚠️  Found existing rationales at {save_path}")
        response = input("Resume from checkpoint or start fresh? (r/s): ")
        if response.lower() == 'r':
            with open(save_path, 'r') as f:
                existing_rationales = json.load(f)
            print(f"✓ Loaded {len(existing_rationales)} existing rationales")
    
    start_idx = len(existing_rationales)
    
    if start_idx >= len(questions):
        print(f"\n✓ Already have {start_idx} rationales. Nothing to generate!")
        return
    
    questions_to_generate = questions[start_idx:]
    
    print(f"\n📝 Generating rationales for samples {start_idx}-{len(questions)}...")
    print(f"   ({len(questions_to_generate)} samples to generate)\n")
    
    # Initialize generator
    generator = FastRationaleGenerator(
        api_key=api_key,
        model_name="gpt-4o-mini",  # Cheaper and better than GPT-3.5!
        max_concurrent=20
    )
    
    # Generate
    new_rationales = await generator.generate_rationales(
        questions=questions_to_generate,
        save_path=save_path,
        checkpoint_interval=500,
        start_idx=start_idx
    )
    
    # Combine
    all_rationales = existing_rationales + new_rationales
    all_rationales.sort(key=lambda x: x['idx'])
    
    # Save final
    with open(save_path, 'w') as f:
        json.dump(all_rationales, f, indent=2)
    
    print("\n" + "="*80)
    print("  ✓ Rationale Generation Complete!")
    print("="*80)
    print(f"  Total rationales: {len(all_rationales)}")
    print(f"  Success rate: {sum(1 for r in all_rationales if r['success'])}/{len(all_rationales)}")
    print(f"  Saved to: {save_path}")
    print("="*80)
    
    print("\n📍 Next step: Run full training")
    print("   python train_full_scale.py")


if __name__ == "__main__":
    asyncio.run(main())