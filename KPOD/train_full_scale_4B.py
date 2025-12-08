"""
Full-scale KPOD training on complete GSM8K dataset
Based on paper specifications
Expected result: 22-25% accuracy
Total time: ~20-24 hours
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
    print("="*80)
    print("  KPOD FULL-SCALE TRAINING")
    print("="*80)
    print("  Dataset: GSM8K (7,473 training samples)")
    print("  Model: Qwen/Qwen2.5-4B")
    print("  Epochs: 100")
    print("  Expected Accuracy: 22-25%")
    print("  Estimated Time: 20-24 hours")
    print("="*80)
    
    # Configuration
    config = get_config()

    # Use Qwen 4B as the student
    config.model.student_model_name = "Qwen/Qwen2.5-4B"   # or "Qwen/Qwen1.5-4B"

    # Dataset
    config.data.dataset_name = "gsm8k"

    # Training hyperparameters for 4B model
    config.training.batch_size = 1          # Qwen-4B is much larger than Flan-T5
    config.training.gradient_accumulation = 16   # Effective batch size = 8
    config.training.lr_flan = 1e-5          # Much lower LR for 4B stability
    config.training.max_grad_norm = 0.8     # Slightly lower for large LMs
    config.training.warmup_ratio = 0.05     # Large LMs need smaller warmup
    config.training.weight_decay = 0.1      # Standard for transformer LMs
    config.training.num_epochs = 100        # Good default
    config.training.early_stopping_patience = 5  # custom field for early stopping

    # Mixed precision
    config.training.fp16 = False
    config.training.bf16 = True             # Qwen models perform best in BF16

    # Optional but recommended
    config.training.lr_scheduler = "cosine"
    config.training.logging_steps = 20

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "qwen2.5-4b-full"
    
    print(f"\n📍 Configuration:")
    print(f"   Device: {device}")
    print(f"   Model: {config.model.student_model_name}")
    print(f"   Learning Rate: {config.training.lr_flan}")
    print(f"   Batch Size: {config.training.batch_size}")
    print(f"   Gradient Clipping: {config.training.max_grad_norm}")
    
    # ========================================================================
    # STEP 1: Check/Generate Rationales
    # ========================================================================
    print_section("STEP 1: Rationales")
    
    rationale_path = "./data/rationales/gsm8k_full_train_rationales.json"
    
    if os.path.exists(rationale_path):
        print(f"✓ Found existing rationales at {rationale_path}")
        rationales = load_rationales(rationale_path)
        print(f"✓ Loaded {len(rationales)} rationales")
    else:
        print(f"❌ Rationales not found at {rationale_path}")
        print(f"\nYou need to generate rationales first.")
        print(f"This will take ~6-8 hours and cost ~$15-20 (GPT-3.5)")
        print(f"\nOptions:")
        print(f"  1. Generate with GPT-3.5: python generate_full_rationales.py")
        print(f"  2. Generate with GPT-4o-mini: ~$5 (faster, better)")
        print(f"  3. Use local Llama: Free but slower")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Load Full Dataset
    # ========================================================================
    print_section("STEP 2: Loading Full Dataset")
    
    full_train, full_val, full_test = load_gsm8k()
    
    NUM_TRAIN = len(full_train)  # 7,473
    NUM_VAL = len(full_val)      # 660
    NUM_TEST = len(full_test)    # 659
    
    print(f"✓ Loaded GSM8K:")
    print(f"   Training: {NUM_TRAIN} samples")
    print(f"   Validation: {NUM_VAL} samples")
    print(f"   Test: {NUM_TEST} samples")
    
    # Verify rationales match
    if len(rationales) < NUM_TRAIN:
        print(f"\n❌ Error: Only {len(rationales)} rationales for {NUM_TRAIN} samples")
        sys.exit(1)
    
    # Create datasets
    train_dataset = ReasoningDataset(
        questions=[full_train[i]['question'] for i in range(NUM_TRAIN)],
        answers=[full_train[i]['answer'] for i in range(NUM_TRAIN)],
        rationales=[rationales[i]['rationale'] for i in range(NUM_TRAIN)],
        split='train'
    )
    
    val_dataset = ReasoningDataset(
        questions=[full_val[i]['question'] for i in range(min(500, NUM_VAL))],
        answers=[full_val[i]['answer'] for i in range(min(500, NUM_VAL))],
        rationales=[None] * min(500, NUM_VAL),
        split='val'
    )

    test_len = min(NUM_TEST, len(full_test))
    test_dataset = ReasoningDataset(
        questions=[full_test[i]['question'] for i in range(test_len)],
        answers=[full_test[i]['answer'] for i in range(test_len)],
        rationales=[None] * test_len,
        split='test'
    )

    print(f"\n✓ Datasets created:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    print(f"   Test: {len(test_dataset)}")
    
    # ========================================================================
    # STEP 3: Train Token Weighting
    # ========================================================================
    token_weights_path = f"./checkpoints/full_token_weighting_{model_id}.pt"
    
    if os.path.exists(token_weights_path):
        print_section("STEP 3: Token Weighting (Loading Existing)")
        print(f"✓ Found existing weights at {token_weights_path}")
    else:
        print_section("STEP 3: Training Token Weighting Module")
        print("Estimated time: 2-3 hours")
        
        token_weighting_module = TokenWeightingModule(
            model_name=config.model.student_model_name,
            hidden_dim=config.model.weight_generator_hidden_dim,
            alpha=config.token_weighting.alpha,
            device=device
        )
        
        # Use subset for token weighting (faster)
        train_subset = ReasoningDataset(
            questions=train_dataset.questions[:2000],
            answers=train_dataset.answers[:2000],
            rationales=train_dataset.rationales[:2000],
            split='train'
        )
        
        val_subset = ReasoningDataset(
            questions=val_dataset.questions[:200],
            answers=val_dataset.answers[:200],
            rationales=val_dataset.rationales[:200],
            split='val'
        )
        
        print(f"Training on {len(train_subset)} samples (subset for efficiency)")
        
        trainer = TokenWeightingTrainer(
            module=token_weighting_module,
            train_dataset=train_subset,
            val_dataset=val_subset,
            config=config,
            device=device
        )
        
        trainer.train(num_epochs=8)
        
        # Save
        torch.save({
            'weight_generator_state_dict': token_weighting_module.weight_generator.state_dict(),
            'model_name': config.model.student_model_name,
        }, token_weights_path)
        
        print(f"✓ Token weighting saved to {token_weights_path}")
    
   # ========================================================================
    # STEP 4: Calculate Step Difficulties
    # ========================================================================
    difficulties_path = f"./data/difficulties/gsm8k_full_difficulties_{model_id}.json"
    token_weights_cache_path = f"./data/token_weights/gsm8k_full_token_weights_{model_id}.pt"
    os.makedirs(os.path.dirname(token_weights_cache_path), exist_ok=True)

    if os.path.exists(difficulties_path) and os.path.exists(token_weights_cache_path):
        print_section("STEP 4: Step Difficulties (Loading Existing)")
        difficulties = load_difficulties(difficulties_path)
        token_weights = torch.load(token_weights_cache_path)
        print(f"✓ Loaded difficulties for {len(difficulties)} samples")
        print(f"✓ Loaded token weights for {len(token_weights)} samples")
    else:
        print_section("STEP 4: Calculating Step Difficulties with Token Weights")
        print("Estimated time: 30-45 minutes (depending on GPU)")
        
        model = StudentModel(
            model_name=config.model.student_model_name,
            lora_r=config.model.lora_r_flan,
            device=device
        )
        
        token_weights = {}
        batch_size = 16  # adjust depending on GPU memory
        student_model_device = device
        token_weighting_module.to(device)
        token_weighting_module.eval()
        
        with torch.no_grad():
            for start_idx in range(0, len(train_dataset), batch_size):
                end_idx = min(start_idx + batch_size, len(train_dataset))
                batch_questions = [train_dataset[i]['question'] for i in range(start_idx, end_idx)]
                batch_answers = [train_dataset[i]['answer'] for i in range(start_idx, end_idx)]
                
                # Get token weights (returns list of tensors per example)
                batch_weights = token_weighting_module(batch_questions, batch_answers)
                
                for i, w in enumerate(batch_weights):
                    token_weights[start_idx + i] = w.cpu()  # move to CPU to save memory
        
        # Save token weights for future runs
        torch.save(token_weights, token_weights_cache_path)
        print(f"✓ Token weights saved to {token_weights_cache_path}")
        
        # Now calculate step difficulties using real token weights
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
    # STEP 5: Cluster Questions
    # ========================================================================
    clusters_path = "./data/clusters/gsm8k_full_clusters.json"
    
    if os.path.exists(clusters_path):
        print_section("STEP 5: Question Clustering (Loading Existing)")
        clusterer = QuestionClusterer()
        clusterer.load(clusters_path)
        print(f"✓ Loaded {clusterer.num_clusters} clusters")
    else:
        print_section("STEP 5: Clustering Questions")
        print("Estimated time: 5-10 minutes")
        
        clusterer = QuestionClusterer(num_clusters=10)  # More clusters for full dataset
        clusterer.fit(train_dataset.questions, embedding_model="glove")
        clusterer.save(clusters_path)
        
        print(f"✓ Clusters saved to {clusters_path}")
        print(f"   Cluster distribution:")
        dist = clusterer.get_cluster_distribution()
        for cluster_id, count in enumerate(dist):
            print(f"     Cluster {cluster_id}: {count} questions")
    
    # ========================================================================
    # STEP 6: Train KPOD Model
    # ========================================================================
    print_section("STEP 6: Training KPOD Model")
    print("Estimated time: 16-18 hours")
    print("This is the longest step - grab coffee! ☕")
    
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
    
    # Load trained token weighting
    checkpoint = torch.load(token_weights_path)
    token_weighting_module.weight_generator.load_state_dict(
        checkpoint['weight_generator_state_dict']
    )
    
    # Initialize scheduler
    scheduler = ProgressiveScheduler(
        step_difficulties=difficulties,
        cluster_assignments=clusterer.cluster_assignments,
        num_epochs=100,  # Full 100 epochs as per paper
        p=0.5,
        c0_ratio=0.3,
        beta=12.0,
        delta_s=1,
        num_clusters=10
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
    
    # Override to use 100 epochs
    trainer._get_num_epochs = lambda: 100
    
    print(f"Starting KPOD training for 100 epochs...")
    print(f"Training on {len(train_dataset)} samples")
    print(f"Validation on {len(val_dataset)} samples")
    
    trainer.train()
    
    # Save final model
    final_model_path = f"./checkpoints/full_kpod_model_{model_id}"
    os.makedirs(final_model_path, exist_ok=True)
    student_model.save_pretrained(final_model_path)
    print(f"✓ Model saved to {final_model_path}")
    
    # ========================================================================
    # STEP 7: Final Evaluation
    # ========================================================================
    print_section("STEP 7: Final Evaluation on Test Set")
    
    evaluator = ReasoningEvaluator(
        model=student_model,
        tokenizer=student_model.tokenizer,
        device=device
    )
    
    eval_output_path = f"./outputs/full_evaluation_{model_id}.json"
    metrics = evaluator.evaluate_dataset(
        dataset=test_dataset,
        dataset_name="gsm8k",
        max_samples=NUM_TEST,
        save_path=eval_output_path
    )
    
    evaluator.print_metrics(metrics)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print_section("FULL TRAINING COMPLETE! 🎉")
    
    print(f"📊 Final Results:")
    print(f"   Model: {model_id}")
    print(f"   Training samples: {NUM_TRAIN}")
    print(f"   Test accuracy: {metrics['accuracy']:.2f}%")
    print(f"   Correct: {metrics['correct']}/{metrics['total']}")
    
    print(f"\n📁 Saved Artifacts:")
    print(f"   Token weights: {token_weights_path}")
    print(f"   Difficulties: {difficulties_path}")
    print(f"   Clusters: {clusters_path}")
    print(f"   Final model: {final_model_path}")
    print(f"   Evaluation: {eval_output_path}")
    print(f"   Training history: ./outputs/training_history.json")
    
    print(f"\n💡 Expected Results:")
    print(f"   Paper baseline (no KPOD): ~18-20%")
    print(f"   Paper with KPOD: ~22-25%")
    
    if metrics['accuracy'] >= 22:
        print(f"   ✅ EXCELLENT! Matched or exceeded paper results!")
    elif metrics['accuracy'] >= 18:
        print(f"   ✅ GOOD! Significant improvement over baseline")
    elif metrics['accuracy'] >= 12:
        print(f"   ⚠️  MODERATE! Some learning but below target")
    else:
        print(f"   ❌ LOW! Model needs debugging")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()