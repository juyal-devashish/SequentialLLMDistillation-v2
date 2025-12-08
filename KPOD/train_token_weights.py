"""
Script to train the token weighting module
"""
import os
import sys
import torch
from configs.config import get_config
from data.load_datasets import load_dataset_by_name, attach_rationales_to_dataset, load_rationales
from models.token_weighting_module import TokenWeightingModule
from training.train_token_weighting import TokenWeightingTrainer, visualize_token_weights


def main():
    # Configuration
    config = get_config()
    config.data.dataset_name = "gsm8k"
    config.model.student_model_name = "google/flan-t5-large"
    
    # Check for rationales
    rationale_path = os.path.join(
        config.data.rationale_dir,
        f"{config.data.dataset_name}_train_rationales.json"
    )
    
    if not os.path.exists(rationale_path):
        print(f"Rationale file not found: {rationale_path}")
        print("Please run: python main.py --mode generate --dataset gsm8k")
        sys.exit(1)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_dataset_by_name(
        config.data.dataset_name,
        config.data.cache_dir
    )
    
    # Load and attach rationales
    print("Loading rationales...")
    rationales = load_rationales(rationale_path)
    train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
    
    # For validation, we'll need to generate rationales too (or use a subset)
    # For now, let's use a small subset of training data as validation
    val_size = 100
    val_dataset = type(train_dataset)(
        questions=train_dataset.questions[:val_size],
        answers=train_dataset.answers[:val_size],
        rationales=train_dataset.rationales[:val_size],
        split='validation'
    )
    
    # Use smaller training set for faster iteration
    train_size = 1000  # Use subset for faster training during development
    train_dataset = type(train_dataset)(
        questions=train_dataset.questions[:train_size],
        answers=train_dataset.answers[:train_size],
        rationales=train_dataset.rationales[:train_size],
        split='train'
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # Initialize token weighting module
    print("\nInitializing Token Weighting Module...")
    module = TokenWeightingModule(
        model_name=config.model.student_model_name,
        embedding_dim=1024,  # Flan-T5-Large
        alpha=config.token_weighting.alpha,
        gumbel_temperature=config.token_weighting.gumbel_temperature,
        device=config.training.device
    )
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = TokenWeightingTrainer(
        module=module,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=config.training.device
    )
    
    # Train
    print("\nStarting training...")
    num_epochs = 10  # Train for 10 epochs
    trainer.train(num_epochs=num_epochs)
    
    # Visualize some examples
    print("\nVisualizing token weights...")
    sample_questions = train_dataset.questions[:3]
    sample_rationales = train_dataset.rationales[:3]
    
    visualize_token_weights(
        module=module,
        questions=sample_questions,
        rationales=sample_rationales,
        save_path=os.path.join(config.output_dir, 'token_weight_visualization.png'),
        num_samples=3
    )
    
    print("\n✓ Token weighting module training complete!")
    print(f"Model saved to: {config.checkpoint_dir}/token_weighting_best.pt")


if __name__ == "__main__":
    main()