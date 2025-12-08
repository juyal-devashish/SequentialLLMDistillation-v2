"""
Generate rationales using a trained student model
"""
import torch
from tqdm import tqdm
import json
import time
from typing import List, Dict, Optional
from models.base_model import StudentModel

class StudentGenerator:
    """Generate rationales using a local StudentModel"""
    
    def __init__(
        self,
        model: StudentModel,
        device: str = "cuda"
    ):
        self.model = model
        self.device = device
        self.model.eval()

    def generate_single_rationale(
        self,
        question: str,
        max_length: int = 512,
        temperature: float = 0.7,
        cot_prompt: str = "Let's think step by step."
    ) -> Dict[str, str]:
        """Generate rationale for a single question using CoT prompt"""
        
        # Prepare input with CoT prompt
        input_text = f"Q: {question} A: {cot_prompt}"
        
        try:
            # Tokenize
            inputs = self.model.prepare_inputs(input_text)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9
                )
            
            # Decode
            generated_text = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the output to extract just the rationale
            # The model generates "Let's think step by step. <rationale>"
            # We want to extract <rationale>
            
            # Usually Flan-T5 output includes the input prompt if trained that way, 
            # or just the continuation. Let's handle both.
            
            rationale = generated_text
            
            # If the generated text starts with the prompt, strip it
            if rationale.startswith(input_text):
                rationale = rationale[len(input_text):].strip()
            elif rationale.startswith(cot_prompt):
                rationale = rationale[len(cot_prompt):].strip()
                
            return {
                'question': question,
                'rationale': rationale,
                'full_text': generated_text,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'question': question,
                'rationale': '',
                'full_text': '',
                'success': False,
                'error': str(e)
            }

    def generate_rationales(
        self,
        questions: List[str],
        save_path: Optional[str] = None,
        checkpoint_interval: int = 100,
        batch_size: int = 8
    ) -> List[Dict]:
        """Generate rationales for multiple questions"""
        
        rationales = []
        
        print(f"Generating rationales for {len(questions)} questions using student model...")
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating"):
            batch_questions = questions[i:i + batch_size]
            
            for question in batch_questions:
                result = self.generate_single_rationale(question)
                rationales.append(result)
            
            # Checkpoint saving
            if save_path and (i + batch_size) % checkpoint_interval == 0:
                self._save_checkpoint(rationales, save_path, i + batch_size)
                
        # Final save
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(rationales, f, indent=2)
            print(f"\nFinal rationales saved: {save_path}")
            
        return rationales
    
    def _save_checkpoint(self, rationales, save_path, index):
        """Save intermediate results"""
        checkpoint_path = save_path.replace('.json', f'_checkpoint_{index}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(rationales, f, indent=2)

