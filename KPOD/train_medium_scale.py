"""
Medium-scale training with 500 samples
Expected accuracy: 12-18%
Training time: ~1.5-2 hours
"""
import os
import sys
import torch
from configs.config import get_config
from data.load_datasets import load_gsm8k, ReasoningDataset, load_rationales
from models.base_model import StudentModel
from models.token_weighting import TokenWeightingModule
from training.train_token_weighting import TokenWeightingTrainer
from training.step_difficulty import StepDifficultyCalculator, save_difficulties, load_difficulties
from training.question_clustering import QuestionClusterer
from training.progressive_scheduler import ProgressiveScheduler
from training.kpod_trainer import KPODTrainer
from evaluation.evaluator import ReasoningEvaluator


def print_section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def main():
    NUM_TRAIN = 500
    NUM_VAL = 50
    NUM_TEST = 100
    
    print("="*80)
    print("  KPOD Medium-Scale Training")
    print("="*80)
    print(f"  Training samples: {NUM_TRAIN}")
    print(f"  Validation samples: {NUM_VAL}")
    print(f"  Test samples: {NUM_TEST}")
    print(f"  Expected time: 1.5-2 hours")
    print(f"  Expected accuracy: 12-18%")
    print("="*80)
    
    # Config
    config = get_config()
    config.model.student_model_name = "google/flan-t5-small" # DEBUG: Changed to small for testing
    config.data.dataset_name = "gsm8k"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n📍 Using device: {device}")
    
    # Check rationales exist
    rationale_path = "./data/rationales/gsm8k_train_500_rationales.json"
    if not os.path.exists(rationale_path):
        print(f"\n❌ Rationales not found at {rationale_path}")
        print("Please run: python generate_500_rationales.py")
        sys.exit(1)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print_section("STEP 1: Loading Data")
    
    full_train, full_val, full_test = load_gsm8k()
    rationales = load_rationales(rationale_path)
    
    # Create datasets
    train_dataset = ReasoningDataset(
        questions=[full_train[i]['question'] for i in range(NUM_TRAIN)],
        answers=[full_train[i]['answer'] for i in range(NUM_TRAIN)],
        rationales=[rationales[i]['rationale'] for i in range(NUM_TRAIN)],
        split='train'
    )
    
    val_dataset = ReasoningDataset(
    questions=[full_train[i]['question'] for i in range(NUM_VAL)],
    answers=[full_train[i]['answer'] for i in range(NUM_VAL)],
    rationales=[rationales[i]['rationale'] for i in range(NUM_VAL)],  # USE rationales, not None!
    split='val')
    
    test_dataset = ReasoningDataset(
        questions=[full_test[i]['question'] for i in range(NUM_TEST)],
        answers=[full_test[i]['answer'] for i in range(NUM_TEST)],
        rationales=[None] * NUM_TEST,
        split='test'
    )
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # ========================================================================
    # STEP 2: Train Token Weighting
    # ========================================================================
    token_weights_path = "./checkpoints/medium_token_weighting.pt"
    
    if os.path.exists(token_weights_path):
        print_section("STEP 2: Token Weighting (Loading Existing)")
        print(f"✓ Found existing weights at {token_weights_path}")
    else:
        print_section("STEP 2: Training Token Weighting Module")
        
        token_weighting_module = TokenWeightingModule(
            model_name=config.model.student_model_name,
            hidden_dim=config.model.weight_generator_hidden_dim,
            alpha=config.token_weighting.alpha,
            device=device
        )
        
        trainer = TokenWeightingTrainer(
            module=token_weighting_module,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device
        )
        
        trainer.train(num_epochs=5)
        
        # Save
        torch.save({
            'weight_generator_state_dict': token_weighting_module.weight_generator.state_dict(),
        }, token_weights_path)
        
        print(f"✓ Token weighting saved to {token_weights_path}")
    
    # ========================================================================
    # STEP 3: Calculate Difficulties
    # ========================================================================
    difficulties_path = "./data/difficulties/gsm8k_medium_difficulties.json"
    
    if os.path.exists(difficulties_path):
        print_section("STEP 3: Step Difficulties (Loading Existing)")
        difficulties = load_difficulties(difficulties_path)
        print(f"✓ Loaded difficulties for {len(difficulties)} samples")
    else:
        print_section("STEP 3: Calculating Step Difficulties")
        
        model = StudentModel(
            model_name=config.model.student_model_name,
            lora_r=config.model.lora_r_flan,
            device=device
        )
        
        token_weights = {idx: torch.ones(50) for idx in range(len(train_dataset))}
        
        calculator = StepDifficultyCalculator(
            student_model=model,
            token_weights=token_weights,
            device=device
        )
        
        difficulties = calculator.calculate_difficulties_for_dataset(
            train_dataset,
            max_samples=len(train_dataset)
        )
        
        save_difficulties(difficulties, difficulties_path)
        print(f"✓ Difficulties saved to {difficulties_path}")
    
    # ========================================================================
    # STEP 4: Cluster Questions
    # ========================================================================
    clusters_path = "./data/clusters/gsm8k_medium_clusters.json"
    
    if os.path.exists(clusters_path):
        print_section("STEP 4: Question Clustering (Loading Existing)")
        clusterer = QuestionClusterer()
        clusterer.load(clusters_path)
        print(f"✓ Loaded {clusterer.num_clusters} clusters")
    else:
        print_section("STEP 4: Clustering Questions")
        
        clusterer = QuestionClusterer(num_clusters=5)
        clusterer.fit(train_dataset.questions, embedding_model="glove")
        clusterer.save(clusters_path)
        
        print(f"✓ Clusters saved to {clusters_path}")
    
    # ========================================================================
    # STEP 5: Train KPOD
    # ========================================================================
    print_section("STEP 5: Training KPOD Model")
    
    # Initialize models
    student_model = StudentModel(
        model_name=config.model.student_model_name,
        lora_r=config.model.lora_r_flan,
        device=device
    )
    
    token_weighting_module = TokenWeightingModule(
        model_name=config.model.student_model_name,
        hidden_dim=config.model.weight_generator_hidden_dim,
        device=device
    )
    
    checkpoint = torch.load(token_weights_path)
    token_weighting_module.weight_generator.load_state_dict(
        checkpoint['weight_generator_state_dict']
    )
    
    # Initialize scheduler
    scheduler = ProgressiveScheduler(
        step_difficulties=difficulties,
        cluster_assignments=clusterer.cluster_assignments,
        num_epochs=15,  # 15 epochs for medium scale
        p=0.5,
        c0_ratio=0.3,
        beta=12.0,
        delta_s=1,
        num_clusters=5
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
    
    # Override epochs
    trainer._get_num_epochs = lambda: 15
    
    print(f"Training for 15 epochs...")
    trainer.train()
    
    # Save final model
    final_model_path = "./checkpoints/medium_kpod_model"
    os.makedirs(final_model_path, exist_ok=True)
    student_model.save_pretrained(final_model_path)
    print(f"✓ Model saved to {final_model_path}")
    
    # ========================================================================
    # STEP 6: Evaluate
    # ========================================================================
    print_section("STEP 6: Final Evaluation")
    
    evaluator = ReasoningEvaluator(
        model=student_model,
        tokenizer=student_model.tokenizer,
        device=device
    )
    
    metrics = evaluator.evaluate_dataset(
        dataset=test_dataset,
        dataset_name="gsm8k",
        max_samples=NUM_TEST,
        save_path="./outputs/medium_evaluation.json"
    )
    
    evaluator.print_metrics(metrics)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Training Complete!")
    
    print(f"📊 Results:")
    print(f"   Training samples: {NUM_TRAIN}")
    print(f"   Test accuracy: {metrics['accuracy']:.2f}%")
    print(f"   Correct: {metrics['correct']}/{metrics['total']}")
    print(f"\n📁 Saved artifacts:")
    print(f"   Token weights: {token_weights_path}")
    print(f"   Difficulties: {difficulties_path}")
    print(f"   Clusters: {clusters_path}")
    print(f"   Final model: {final_model_path}")
    print(f"   Evaluation: ./outputs/medium_evaluation.json")
    
    print(f"\n💡 Expected accuracy range: 12-18%")
    if metrics['accuracy'] >= 12:
        print(f"   ✓ Result is within expected range!")
    else:
        print(f"   ⚠️  Result is lower than expected (may need more training)")
    
    print("\n🚀 Next steps:")
    print("   1. If results look good, proceed with full training")
    print("   2. Full training command: python main.py --mode full_pipeline --dataset gsm8k")
    print("   3. Expected full training accuracy: 22-25%")


if __name__ == "__main__":
    main()