"""
Test the complete KPOD pipeline on a small subset of data
This verifies all components work together end-to-end
"""
import os
import sys
import torch
from configs.config import get_config
from data.load_datasets import (
    load_dataset_by_name, 
    attach_rationales_to_dataset,
    save_rationales,
    load_rationales
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
from evaluation.evaluator import ReasoningEvaluator


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_pipeline():
    """Run complete pipeline on small subset"""
    
    print_section("KPOD PIPELINE TEST - Running on Small Subset")
    
    # Configuration
    config = get_config()
    config.data.dataset_name = "gsm8k"
    config.model.student_model_name = "google/flan-t5-small"  # Use small model for speed
    
    # Test parameters
    NUM_TRAIN_SAMPLES = 1000  # Small subset for testing
    NUM_VAL_SAMPLES = 10
    NUM_TEST_SAMPLES = 10
    NUM_EPOCHS_TOKEN_WEIGHTING = 2
    NUM_EPOCHS_KPOD = 3
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Create directories
    os.makedirs(config.data.cache_dir, exist_ok=True)
    os.makedirs(config.data.rationale_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs("./data/difficulties", exist_ok=True)
    os.makedirs("./data/clusters", exist_ok=True)
    
    # Paths for test artifacts
    test_rationale_path = "./data/rationales/test_rationales.json"
    test_difficulties_path = "./data/difficulties/test_difficulties.json"
    test_clusters_path = "./data/clusters/test_clusters.json"
    test_token_weights_path = "./checkpoints/test_token_weighting.pt"
    test_model_path = "./checkpoints/test_kpod_model"
    
    try:
        # ====================================================================
        # STEP 1: Load Dataset
        # ====================================================================
        print_section("STEP 1: Loading Dataset")
        
        full_train, full_val, full_test = load_dataset_by_name("gsm8k", config.data.cache_dir)
        
        # Create small subsets
        from torch.utils.data import Subset
        train_indices = list(range(NUM_TRAIN_SAMPLES))
        val_indices = list(range(NUM_VAL_SAMPLES))
        test_indices = list(range(NUM_TEST_SAMPLES))
        
        train_dataset = Subset(full_train, train_indices)
        val_dataset = Subset(full_val, val_indices)
        test_dataset = Subset(full_test, test_indices)
        
        print(f"✓ Loaded datasets:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        # ====================================================================
        # STEP 2: Generate Rationales
        # ====================================================================
        print_section("STEP 2: Generating Rationales")
        
        # Check if we should skip rationale generation
        skip_generation = False
        if os.path.exists(test_rationale_path):
            response = input(f"Rationales already exist at {test_rationale_path}. Skip generation? (y/n): ")
            skip_generation = response.lower() == 'y'
        
        if not skip_generation:
            api_key = os.environ.get('OPENAI_API_KEY') or config.openai_api_key
            if not api_key:
                print("⚠️  No OpenAI API key found. Using dummy rationales for testing.")
                print("   Set OPENAI_API_KEY to use real rationales.")
                
                # Create dummy rationales
                rationales = []
                for idx in train_indices:
                    sample = full_train[idx]
                    rationales.append({
                        'question': sample['question'],
                        'rationale': f"Step 1: Let's solve this. Step 2: The calculation is needed. Step 3: We get the answer.",
                        'success': True,
                        'error': None
                    })
                
                save_rationales(rationales, test_rationale_path)
            else:
                # Generate real rationales
                generator = RationaleGenerator(api_key)
                questions = [full_train[idx]['question'] for idx in train_indices]
                
                print(f"Generating rationales for {len(questions)} questions...")
                print("This may take a few minutes...")
                
                rationales = generator.generate_rationales(
                    questions=questions,
                    batch_delay=1.0,
                    save_path=test_rationale_path,
                    checkpoint_interval=10
                )
        else:
            print("Skipping rationale generation (using existing file)")
            rationales = load_rationales(test_rationale_path)
        
        print(f"✓ Rationales ready: {len(rationales)} samples")
        
        # Attach rationales to datasets
        # For subset datasets, we need to create proper dataset objects
        from data.load_datasets import ReasoningDataset
        
        train_questions = [full_train[idx]['question'] for idx in train_indices]
        train_answers = [full_train[idx]['answer'] for idx in train_indices]
        train_rationales = [r['rationale'] for r in rationales]
        
        train_dataset = ReasoningDataset(
            questions=train_questions,
            answers=train_answers,
            rationales=train_rationales,
            split='train'
        )
        
        # Val dataset (reuse some training rationales for simplicity)
        val_questions = [full_val[idx]['question'] for idx in val_indices]
        val_answers = [full_val[idx]['answer'] for idx in val_indices]
        val_rationales = train_rationales[:NUM_VAL_SAMPLES]
        
        val_dataset = ReasoningDataset(
            questions=val_questions,
            answers=val_answers,
            rationales=val_rationales,
            split='val'
        )
        
        # ====================================================================
        # STEP 3: Train Token Weighting Module
        # ====================================================================
        print_section("STEP 3: Training Token Weighting Module")
        
        token_weighting_module = TokenWeightingModule(
            model_name=config.model.student_model_name,
            hidden_dim=128,  # Smaller for testing
            alpha=config.token_weighting.alpha,
            device=device
        )
        
        print("Training token weighting module...")
        trainer = TokenWeightingTrainer(
            module=token_weighting_module,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device
        )
        
        # Override epochs for testing
        original_train_epoch = trainer.train_epoch
        def limited_train_epoch(epoch):
            result = original_train_epoch(epoch)
            # Limit batches for testing
            return result
        trainer.train_epoch = limited_train_epoch
        
        trainer.train(num_epochs=NUM_EPOCHS_TOKEN_WEIGHTING)
        
        # Save
        torch.save({
            'weight_generator_state_dict': token_weighting_module.weight_generator.state_dict(),
        }, test_token_weights_path)
        
        print(f"✓ Token weighting module trained and saved to {test_token_weights_path}")
        
        # ====================================================================
        # STEP 4: Extract Token Weights
        # ====================================================================
        print_section("STEP 4: Extracting Token Weights")
        
        token_weights = {}
        for idx in range(len(train_dataset)):
            sample = train_dataset[idx]
            rationale = sample['rationale']
            
            rationale_enc = token_weighting_module.tokenizer(
                rationale,
                return_tensors="pt",
                truncation=True,
                max_length=config.data.max_rationale_length
            )
            
            with torch.no_grad():
                weights = token_weighting_module.get_token_weights(
                    rationale_enc['input_ids'].to(device),
                    rationale_enc['attention_mask'].to(device)
                )
            
            token_weights[idx] = weights[0].cpu()
        
        print(f"✓ Extracted token weights for {len(token_weights)} samples")
        
        # Show example weights
        print("\nExample token weights for first sample:")
        sample = train_dataset[0]
        tokens = token_weighting_module.tokenizer.tokenize(sample['rationale'])[:10]
        weights_sample = token_weights[0][:10].numpy()
        for token, weight in zip(tokens, weights_sample):
            print(f"  {token:20s}: {weight:.3f}")
        
        # ====================================================================
        # STEP 5: Calculate Step Difficulties
        # ====================================================================
        print_section("STEP 5: Calculating Step Difficulties")
        
        # Initialize student model for difficulty calculation
        student_model_for_diff = StudentModel(
            model_name=config.model.student_model_name,
            lora_r=8,
            device=device
        )
        
        calculator = StepDifficultyCalculator(
            student_model=student_model_for_diff,
            token_weights=token_weights,
            device=device
        )
        
        print("Calculating difficulties...")
        difficulties = calculator.calculate_difficulties_for_dataset(
            train_dataset,
            max_samples=len(train_dataset)
        )
        
        save_difficulties(difficulties, test_difficulties_path)
        
        print(f"✓ Calculated difficulties for {len(difficulties)} samples")
        
        # Show example difficulties
        print("\nExample step difficulties for first 3 samples:")
        for idx in range(min(3, len(difficulties))):
            diffs = difficulties[idx]
            print(f"  Sample {idx}: {len(diffs)} steps, difficulties: {[f'{d:.3f}' for d in diffs]}")
        
        # ====================================================================
        # STEP 6: Cluster Questions
        # ====================================================================
        print_section("STEP 6: Clustering Questions")
        
        questions = train_dataset.questions
        
        clusterer = QuestionClusterer(num_clusters=3)  # Fewer clusters for small dataset
        print("Clustering questions...")
        clusterer.fit(questions, embedding_model="glove")
        
        clusterer.save(test_clusters_path)
        
        print(f"✓ Clustered {len(questions)} questions into {clusterer.num_clusters} clusters")
        
        # Show cluster distribution
        print("\nCluster distribution:")
        unique, counts = torch.unique(torch.tensor(clusterer.cluster_assignments), return_counts=True)
        for cluster_id, count in zip(unique.numpy(), counts.numpy()):
            print(f"  Cluster {cluster_id}: {count} questions")
        
        # ====================================================================
        # STEP 7: Initialize Progressive Scheduler
        # ====================================================================
        print_section("STEP 7: Initializing Progressive Scheduler")
        
        scheduler = ProgressiveScheduler(
            step_difficulties=difficulties,
            cluster_assignments=clusterer.cluster_assignments,
            num_epochs=NUM_EPOCHS_KPOD,
            p=config.progressive.p,
            c0_ratio=config.progressive.c0_ratio,
            beta=config.progressive.beta,
            delta_s=config.progressive.delta_s,
            num_clusters=3
        )
        
        print(f"✓ Progressive scheduler initialized")
        print(f"  Max difficulty (B): {scheduler.B:.2f}")
        print(f"  Initial difficulty (C0): {scheduler.C0:.2f}")
        print(f"  Stage T: {scheduler.T}")
        
        # ====================================================================
        # STEP 8: Train KPOD Model
        # ====================================================================
        print_section("STEP 8: Training KPOD Model")
        
        # Initialize fresh student model
        student_model = StudentModel(
            model_name=config.model.student_model_name,
            lora_r=8,
            device=device
        )
        
        # Reload token weighting module
        token_weighting_module_for_train = TokenWeightingModule(
            model_name=config.model.student_model_name,
            hidden_dim=128,
            alpha=config.token_weighting.alpha,
            device=device
        )
        checkpoint = torch.load(test_token_weights_path)
        token_weighting_module_for_train.weight_generator.load_state_dict(
            checkpoint['weight_generator_state_dict']
        )
        
        # Initialize KPOD trainer
        kpod_trainer = KPODTrainer(
            student_model=student_model,
            token_weighting_module=token_weighting_module_for_train,
            progressive_scheduler=scheduler,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device
        )
        
        # Override number of epochs
        original_get_num_epochs = kpod_trainer._get_num_epochs
        kpod_trainer._get_num_epochs = lambda: NUM_EPOCHS_KPOD
        
        print(f"Training KPOD model for {NUM_EPOCHS_KPOD} epochs...")
        kpod_trainer.train()
        
        # Save final model
        os.makedirs(test_model_path, exist_ok=True)
        student_model.save_pretrained(test_model_path)
        
        print(f"✓ KPOD model trained and saved to {test_model_path}")
        
        # ====================================================================
        # STEP 9: Evaluate Model
        # ====================================================================
        print_section("STEP 9: Evaluating Model")
        
        # Load test dataset
        test_questions = [full_test[idx]['question'] for idx in test_indices]
        test_answers = [full_test[idx]['answer'] for idx in test_indices]
        test_dataset_eval = ReasoningDataset(
            questions=test_questions,
            answers=test_answers,
            rationales=[None] * len(test_questions),
            split='test'
        )
        
        # Evaluate
        evaluator = ReasoningEvaluator(
            model=student_model,
            tokenizer=student_model.tokenizer,
            device=device
        )
        
        print(f"Evaluating on {len(test_dataset_eval)} test samples...")
        metrics = evaluator.evaluate_dataset(
            dataset=test_dataset_eval,
            dataset_name="gsm8k",
            save_path="./outputs/test_evaluation.json"
        )
        
        evaluator.print_metrics(metrics)
        
        # ====================================================================
        # STEP 10: Summary
        # ====================================================================
        print_section("PIPELINE TEST COMPLETE ✓")
        
        print("Summary of artifacts created:")
        print(f"  Rationales: {test_rationale_path}")
        print(f"  Token weights: {test_token_weights_path}")
        print(f"  Step difficulties: {test_difficulties_path}")
        print(f"  Question clusters: {test_clusters_path}")
        print(f"  Trained model: {test_model_path}")
        print(f"  Evaluation results: ./outputs/test_evaluation.json")
        
        print(f"\nFinal Test Accuracy: {metrics['accuracy']:.2f}%")
        
        print("\n" + "="*80)
        print("  All components working correctly!")
        print("  Ready to run on full dataset.")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"  ✗ ERROR: Pipeline test failed")
        print(f"{'='*80}\n")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_artifacts():
    """Clean up test artifacts"""
    import shutil
    
    print("\nCleaning up test artifacts...")
    
    artifacts = [
        "./data/rationales/test_rationales.json",
        "./data/difficulties/test_difficulties.json",
        "./data/clusters/test_clusters.json",
        "./checkpoints/test_token_weighting.pt",
        "./checkpoints/test_kpod_model",
        "./outputs/test_evaluation.json"
    ]
    
    for path in artifacts:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            print(f"  Removed: {path}")
    
    print("✓ Cleanup complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test KPOD pipeline on small subset")
    parser.add_argument('--cleanup', action='store_true', help='Clean up test artifacts before running')
    parser.add_argument('--cleanup-only', action='store_true', help='Only clean up test artifacts')
    
    args = parser.parse_args()
    
    if args.cleanup_only:
        cleanup_test_artifacts()
        sys.exit(0)
    
    if args.cleanup:
        cleanup_test_artifacts()
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      KPOD PIPELINE INTEGRATION TEST                          ║
║                                                                              ║
║  This script tests the complete KPOD pipeline on a small subset of data     ║
║  to verify all components work together correctly.                           ║
║                                                                              ║
║  Test Parameters:                                                            ║
║    - 50 training samples                                                     ║
║    - 10 validation samples                                                   ║
║    - 10 test samples                                                         ║
║    - 2 epochs for token weighting                                            ║
║    - 3 epochs for KPOD training                                              ║
║                                                                              ║
║  Estimated time: 10-20 minutes (with API) or 5-10 minutes (dummy data)      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Ask for confirmation
    response = input("Press Enter to start the test (or Ctrl+C to cancel)...")
    
    success = test_pipeline()
    
    if success:
        print("\n✓ Pipeline test passed! All components are working correctly.")
        print("\nNext steps:")
        print("1. Run on full dataset: python main.py --mode full_pipeline --dataset gsm8k")
        print("2. Or run individual steps as needed")
        sys.exit(0)
    else:
        print("\n✗ Pipeline test failed. Please check the errors above.")
        sys.exit(1)