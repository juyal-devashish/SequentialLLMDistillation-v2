"""
Main entry point for KPOD implementation
Complete pipeline from rationale generation to training and evaluation
"""
import os
import sys
import argparse
import torch
from configs.config import get_config
from data.load_datasets import (
    load_dataset_by_name, 
    attach_rationales_to_dataset, 
    load_rationales, 
    save_rationales
)
from data.generate_rationales import RationaleGenerator
from models.base_model import StudentModel
from models.token_weighting import TokenWeightingModule
from training.train_token_weighting import TokenWeightingTrainer
from training.step_difficulty import (
    StepDifficultyCalculator,
    save_difficulties,
    load_difficulties
)
from training.question_clustering import QuestionClusterer
from training.progressive_scheduler import ProgressiveScheduler
from training.kpod_trainer import KPODTrainer
from training.sequential_trainer import SequentialDistillationManager
from evaluation.evaluator import ReasoningEvaluator


def generate_rationales(config, dataset_name):
    """Generate rationales using teacher LLM"""
    print(f"\n{'='*80}")
    print(f"Step 1: Generating Rationales for {dataset_name}")
    print(f"{'='*80}\n")
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY') or config.openai_api_key
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    
    # Load dataset
    train_dataset, _, _ = load_dataset_by_name(dataset_name, config.data.cache_dir)
    
    # Initialize generator
    generator = RationaleGenerator(api_key)
    
    # Generate rationales
    save_path = os.path.join(config.data.rationale_dir, f"{dataset_name}_train_rationales.json")
    rationales = generator.generate_rationales(
        questions=train_dataset.questions,
        save_path=save_path,
        checkpoint_interval=100
    )
    
    print(f"\n✓ Rationale generation complete!")
    return rationales


def train_token_weighting(config, dataset_name):
    """Train token weighting module"""
    print(f"\n{'='*80}")
    print(f"Step 2: Training Token Weighting Module")
    print(f"{'='*80}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load datasets
    train_dataset, val_dataset, _ = load_dataset_by_name(dataset_name, config.data.cache_dir)
    
    # Load rationales
    rationale_path = os.path.join(config.data.rationale_dir, f"{dataset_name}_train_rationales.json")
    rationales = load_rationales(rationale_path)
    train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
    
    # Use subset of training data for validation (since we don't have val rationales)
    val_rationales = rationales[:len(val_dataset)]
    val_dataset = attach_rationales_to_dataset(val_dataset, val_rationales)
    
    # Initialize module
    module = TokenWeightingModule(
        model_name=config.model.student_model_name,
        hidden_dim=config.model.weight_generator_hidden_dim,
        alpha=config.token_weighting.alpha,
        device=device
    )
    
    # Train
    trainer = TokenWeightingTrainer(
        module=module,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )
    
    trainer.train(num_epochs=5)
    
    print(f"\n✓ Token weighting module training complete!")


def calculate_step_difficulties(config, dataset_name):
    """Calculate step difficulties"""
    print(f"\n{'='*80}")
    print(f"Step 3: Calculating Step Difficulties")
    print(f"{'='*80}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset with rationales
    train_dataset, _, _ = load_dataset_by_name(dataset_name, config.data.cache_dir)
    rationale_path = os.path.join(config.data.rationale_dir, f"{dataset_name}_train_rationales.json")
    rationales = load_rationales(rationale_path)
    train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
    
    # Initialize student model (pre-trained)
    model = StudentModel(
        model_name=config.model.student_model_name,
        lora_r=config.model.lora_r_flan if 't5' in config.model.student_model_name else config.model.lora_r_llama,
        device=device
    )
    
    # Create dummy token weights for now (in full implementation, extract from module)
    token_weights = {}
    for idx in range(len(train_dataset)):
        token_weights[idx] = torch.ones(50)
    
    # Calculate difficulties
    calculator = StepDifficultyCalculator(
        student_model=model,
        token_weights=token_weights,
        device=device
    )
    
    difficulties = calculator.calculate_difficulties_for_dataset(
        train_dataset,
        max_samples=len(train_dataset)
    )
    
    # Save
    save_path = os.path.join("./data/difficulties", f"{dataset_name}_train_difficulties.json")
    save_difficulties(difficulties, save_path)
    
    print(f"\n✓ Step difficulty calculation complete!")

def cluster_questions(config, dataset_name):
    """Cluster questions for diversity"""
    print(f"\n{'='*80}")
    print(f"Step 4: Clustering Questions")
    print(f"{'='*80}\n")
    
    # Load dataset
    train_dataset, _, _ = load_dataset_by_name(dataset_name, config.data.cache_dir)
    questions = train_dataset.questions
    
    # Cluster
    clusterer = QuestionClusterer(num_clusters=config.progressive.num_clusters)
    clusterer.fit(questions, embedding_model="glove")
    
    # Save
    save_path = os.path.join("./data/clusters", f"{dataset_name}_clusters.json")
    clusterer.save(save_path)
    
    print(f"\n✓ Question clustering complete!")


def train_kpod(config, dataset_name):
    """Train complete KPOD model"""
    print(f"\n{'='*80}")
    print(f"Step 5: Training KPOD Model")
    print(f"{'='*80}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    train_dataset, val_dataset, _ = load_dataset_by_name(dataset_name, config.data.cache_dir)
    rationale_path = os.path.join(config.data.rationale_dir, f"{dataset_name}_train_rationales.json")
    rationales = load_rationales(rationale_path)
    train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
    
    # Load preprocessing artifacts
    difficulties_path = os.path.join("./data/difficulties", f"{dataset_name}_train_difficulties.json")
    difficulties = load_difficulties(difficulties_path)
    
    clusters_path = os.path.join("./data/clusters", f"{dataset_name}_clusters.json")
    clusterer = QuestionClusterer()
    clusterer.load(clusters_path)
    
    # Initialize models
    student_model = StudentModel(
        model_name=config.model.student_model_name,
        lora_r=config.model.lora_r_flan if 't5' in config.model.student_model_name else config.model.lora_r_llama,
        device=device
    )
    
    token_weighting_module = TokenWeightingModule(
        model_name=config.model.student_model_name,
        hidden_dim=config.model.weight_generator_hidden_dim,
        device=device
    )
    
    # Load trained token weighting
    token_weights_path = os.path.join(config.checkpoint_dir, "token_weighting_best.pt")
    checkpoint = torch.load(token_weights_path)
    token_weighting_module.weight_generator.load_state_dict(
        checkpoint['weight_generator_state_dict']
    )
    
    # Initialize scheduler
    num_epochs = config.training.epochs_flan if 't5' in config.model.student_model_name else config.training.epochs_gsm8k_llama
    scheduler = ProgressiveScheduler(
        step_difficulties=difficulties,
        cluster_assignments=clusterer.cluster_assignments,
        num_epochs=num_epochs,
        p=config.progressive.p,
        c0_ratio=config.progressive.c0_ratio,
        beta=config.progressive.beta,
        delta_s=config.progressive.delta_s,
        num_clusters=config.progressive.num_clusters
    )
    
    # Train
    trainer = KPODTrainer(
        student_model=student_model,
        token_weighting_module=token_weighting_module,
        progressive_scheduler=scheduler,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )
    
    trainer.train()
    
    print(f"\n✓ KPOD training complete!")


def evaluate_model(config, dataset_name, model_path):
    """Evaluate trained model"""
    print(f"\n{'='*80}")
    print(f"Evaluating Model on {dataset_name}")
    print(f"{'='*80}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = StudentModel(
        model_name=config.model.student_model_name,
        lora_r=config.model.lora_r_flan if 't5' in config.model.student_model_name else config.model.lora_r_llama,
        device=device
    )
    model.load_pretrained(model_path)
    
    # Load test set
    _, _, test_dataset = load_dataset_by_name(dataset_name, config.data.cache_dir)
    
    # Evaluate
    evaluator = ReasoningEvaluator(model, model.tokenizer, device)
    metrics = evaluator.evaluate_dataset(
        dataset=test_dataset,
        dataset_name=dataset_name,
        save_path=os.path.join(config.output_dir, f"{dataset_name}_evaluation.json")
    )
    
    evaluator.print_metrics(metrics)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="KPOD Implementation - Complete Pipeline")
    parser.add_argument('--mode', type=str, required=True,
                       choices=['generate', 'train_token_weighting', 'calculate_difficulties',
                               'cluster', 'train', 'evaluate', 'full_pipeline', 'sequential'],
                       help='Execution mode')
    parser.add_argument('--dataset', type=str, default='gsm8k',
                       choices=['gsm8k', 'commonsenseqa', 'asdiv', 'svamp'],
                       help='Dataset name')
    parser.add_argument('--model', type=str, default='google/flan-t5-large',
                       help='Student model name')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model (for evaluation)')
    parser.add_argument('--student_chain', nargs='+', default=None,
                        help='List of student models for sequential distillation')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    config.data.dataset_name = args.dataset
    config.model.student_model_name = args.model
    
    # Create directories
    os.makedirs(config.data.cache_dir, exist_ok=True)
    os.makedirs(config.data.rationale_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs("./data/difficulties", exist_ok=True)
    os.makedirs("./data/clusters", exist_ok=True)
    
    # Execute based on mode
    try:
        if args.mode == 'generate':
            generate_rationales(config, args.dataset)
        
        elif args.mode == 'train_token_weighting':
            train_token_weighting(config, args.dataset)
        
        elif args.mode == 'calculate_difficulties':
            calculate_step_difficulties(config, args.dataset)
        
        elif args.mode == 'cluster':
            cluster_questions(config, args.dataset)
        
        elif args.mode == 'train':
            train_kpod(config, args.dataset)
        
        elif args.mode == 'evaluate':
            if args.model_path is None:
                args.model_path = os.path.join(config.checkpoint_dir, "kpod_best_model")
            evaluate_model(config, args.dataset, args.model_path)
        
        elif args.mode == 'full_pipeline':
            print("Running full KPOD pipeline...")
            generate_rationales(config, args.dataset)
            train_token_weighting(config, args.dataset)
            calculate_step_difficulties(config, args.dataset)
            cluster_questions(config, args.dataset)
            train_kpod(config, args.dataset)
            
            model_path = os.path.join(config.checkpoint_dir, "kpod_best_model")
            evaluate_model(config, args.dataset, model_path)
            
        elif args.mode == 'sequential':
            print("Running Sequential Distillation Pipeline...")
            
            manager = SequentialDistillationManager(
                config=config,
                dataset_name=args.dataset,
                student_chain=args.student_chain
            )
            manager.run_pipeline()
        
        print("\n" + "="*80)
        print("✓ Execution completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()