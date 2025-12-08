"""
Dataset loading utilities
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import pandas as pd


class ReasoningDataset(Dataset):
    """Generic reasoning dataset"""
    
    def __init__(
        self,
        questions: List[str],
        answers: List[str],
        rationales: Optional[List[str]] = None,
        split: str = "train"
    ):
        self.questions = questions
        self.answers = answers
        self.rationales = rationales if rationales is not None else [None] * len(questions)
        self.split = split
        
        assert len(self.questions) == len(self.answers) == len(self.rationales)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        item = {
            'question': self.questions[idx],
            'answer': self.answers[idx],
        }
        if self.rationales[idx] is not None:
            item['rationale'] = self.rationales[idx]
        return item


def load_gsm8k(cache_dir: str = "./data/cache") -> Tuple[ReasoningDataset, ReasoningDataset, ReasoningDataset]:
    """Load GSM8K dataset"""
    print("Loading GSM8K dataset...")
    
    dataset = load_dataset("gsm8k", "main", cache_dir=cache_dir)
    
    def extract_answer(answer_text: str) -> str:
        """Extract numerical answer from '#### NUMBER' format"""
        if '####' in answer_text:
            return answer_text.split('####')[1].strip()
        return answer_text.strip()
    
    # Train set
    train_questions = [item['question'] for item in dataset['train']]
    train_answers = [extract_answer(item['answer']) for item in dataset['train']]
    
    # Test set (GSM8K doesn't have official validation set, we'll split test)
    test_questions = [item['question'] for item in dataset['test']]
    test_answers = [extract_answer(item['answer']) for item in dataset['test']]
    
    # Split test into validation and test (following paper's split)
    val_size = 660
    val_questions = test_questions[:val_size]
    val_answers = test_answers[:val_size]
    test_questions = test_questions[val_size:]
    test_answers = test_answers[val_size:]
    
    train_dataset = ReasoningDataset(train_questions, train_answers, split='train')
    val_dataset = ReasoningDataset(val_questions, val_answers, split='validation')
    test_dataset = ReasoningDataset(test_questions, test_answers, split='test')
    
    print(f"GSM8K - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def load_commonsenseqa(cache_dir: str = "./data/cache") -> Tuple[ReasoningDataset, ReasoningDataset, ReasoningDataset]:
    """Load CommonsenseQA dataset"""
    print("Loading CommonsenseQA dataset...")
    
    dataset = load_dataset("commonsense_qa", cache_dir=cache_dir)
    
    def format_question(item):
        """Format question with choices"""
        question = item['question']
        choices = item['choices']
        choice_text = " ".join([f"({label}) {text}" for label, text in zip(choices['label'], choices['text'])])
        return f"{question} {choice_text}"
    
    # Train set
    train_questions = [format_question(item) for item in dataset['train']]
    train_answers = [item['answerKey'] for item in dataset['train']]
    
    # Validation set
    val_questions = [format_question(item) for item in dataset['validation']]
    val_answers = [item['answerKey'] for item in dataset['validation']]
    
    # Test set (use validation as test since test labels aren't public)
    test_questions = val_questions.copy()
    test_answers = val_answers.copy()
    
    train_dataset = ReasoningDataset(train_questions, train_answers, split='train')
    val_dataset = ReasoningDataset(val_questions, val_answers, split='validation')
    test_dataset = ReasoningDataset(test_questions, test_answers, split='test')
    
    print(f"CommonsenseQA - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def load_dataset_by_name(dataset_name: str, cache_dir: str = "./data/cache"):
    """Load dataset by name"""
    if dataset_name.lower() == 'gsm8k':
        return load_gsm8k(cache_dir)
    elif dataset_name.lower() == 'commonsenseqa':
        return load_commonsenseqa(cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def save_rationales(rationales: List[Dict], save_path: str):
    """Save generated rationales to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(rationales, f, indent=2)
    print(f"Rationales saved to {save_path}")


def load_rationales(load_path: str) -> List[Dict]:
    """Load rationales from file"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Rationale file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        rationales = json.load(f)
    print(f"Loaded {len(rationales)} rationales from {load_path}")
    return rationales


def attach_rationales_to_dataset(
    dataset: ReasoningDataset,
    rationales: List[Dict]
) -> ReasoningDataset:
    """Attach rationales to dataset"""
    assert len(dataset) == len(rationales), "Dataset and rationales length mismatch"
    
    rationale_texts = []
    for i, item in enumerate(rationales):
        rationale_texts.append(item.get('rationale', ''))
    
    return ReasoningDataset(
        questions=dataset.questions,
        answers=dataset.answers,
        rationales=rationale_texts,
        split=dataset.split
    )


# Example usage
if __name__ == "__main__":
    # Test dataset loading
    train, val, test = load_gsm8k()
    print("\nSample from GSM8K:")
    print(f"Question: {train[0]['question']}")
    print(f"Answer: {train[0]['answer']}")
    
    train, val, test = load_commonsenseqa()
    print("\nSample from CommonsenseQA:")
    print(f"Question: {train[0]['question']}")
    print(f"Answer: {train[0]['answer']}")