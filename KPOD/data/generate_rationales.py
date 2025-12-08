"""
Generate rationales using teacher LLM (GPT-3.5-Turbo)
"""
import os
import time
from openai import OpenAI
from typing import List, Dict
from tqdm import tqdm
import json


class RationaleGenerator:
    """Generate rationales using OpenAI API"""
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Use new OpenAI client
        self.client = OpenAI(api_key=api_key)
    
    def generate_single_rationale(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, str]:
        """Generate rationale for a single question"""
        
        # Zero-shot CoT prompt
        prompt = f"Q: {question}\nA: Let's think step by step."
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that solves problems step by step."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                rationale = response.choices[0].message.content.strip()
                
                return {
                    'question': question,
                    'rationale': rationale,
                    'success': True,
                    'error': None
                }
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Error generating rationale (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay)
                else:
                    return {
                        'question': question,
                        'rationale': '',
                        'success': False,
                        'error': str(e)
                    }
        
        return {
            'question': question,
            'rationale': '',
            'success': False,
            'error': 'Max retries exceeded'
        }
    
    def generate_rationales(
        self,
        questions: List[str],
        batch_delay: float = 1.0,
        save_path: str = None,
        checkpoint_interval: int = 100
    ) -> List[Dict]:
        """Generate rationales for multiple questions"""
        
        rationales = []
        
        for i, question in enumerate(tqdm(questions, desc="Generating rationales")):
            result = self.generate_single_rationale(question)
            rationales.append(result)
            
            # Rate limiting
            time.sleep(batch_delay)
            
            # Checkpoint saving
            if save_path and (i + 1) % checkpoint_interval == 0:
                checkpoint_path = save_path.replace('.json', f'_checkpoint_{i+1}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump(rationales, f, indent=2)
                print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Final save
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(rationales, f, indent=2)
            print(f"\nFinal rationales saved: {save_path}")
        
        # Statistics
        success_count = sum(1 for r in rationales if r['success'])
        print(f"\nGeneration complete: {success_count}/{len(rationales)} successful")
        
        return rationales


def format_rationale_for_training(
    question: str,
    rationale: str,
    answer: str,
    cot_prompt: str = "Let's think step by step."
) -> str:
    """
    Format rationale in training template:
    Q: <question> A: <prompt> <rationale> Therefore, the answer is <answer>
    """
    return f"Q: {question} A: {cot_prompt} {rationale} Therefore, the answer is {answer}"


# Example usage and testing
if __name__ == "__main__":
    import sys
    from load_datasets import load_gsm8k
    
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Load dataset
    train_dataset, _, _ = load_gsm8k()
    
    # Test with a few examples
    print("Testing rationale generation with 3 examples...")
    generator = RationaleGenerator(api_key)
    
    test_questions = [train_dataset[i]['question'] for i in range(3)]
    rationales = generator.generate_rationales(
        test_questions,
        batch_delay=1.0
    )
    
    # Display results
    for i, result in enumerate(rationales):
        print(f"\n{'='*80}")
        print(f"Question {i+1}: {result['question']}")
        print(f"\nRationale: {result['rationale']}")
        print(f"Success: {result['success']}")