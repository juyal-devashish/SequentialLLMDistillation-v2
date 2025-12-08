"""
Sequential Distillation Manager
Orchestrates the multi-generational student-teaching-student process
"""
import os
import torch
import json
from typing import List
from models.base_model import StudentModel
from models.token_weighting import TokenWeightingModule
from data.student_generator import StudentGenerator
from data.load_datasets import load_dataset_by_name, attach_rationales_to_dataset, load_rationales
from training.train_token_weighting import TokenWeightingTrainer
from training.step_difficulty import StepDifficultyCalculator, save_difficulties
from training.question_clustering import QuestionClusterer
from training.progressive_scheduler import ProgressiveScheduler
from training.kpod_trainer import KPODTrainer

class SequentialDistillationManager:
    """
    Manages sequential distillation where Student N teaches Student N+1
    """
    
    def __init__(
        self,
        config,
        dataset_name: str,
        student_chain: List[str] = None
    ):
        self.config = config
        self.dataset_name = dataset_name
        
        # Smart device selection
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"SequentialDistillationManager using device: {self.device}")
        
        # Default chain if not provided
        self.student_chain = student_chain or [
            "google/flan-t5-large",  # Student 1 (teaches next)
            "google/flan-t5-base",   # Student 2
            "google/flan-t5-small"   # Student 3
        ]
        
        print(f"Initialized Sequential Manager with chain: {self.student_chain}")

    def run_pipeline(self):
        """Execute the full sequential distillation pipeline"""
        
        # Initial teacher is GPT-3.5 (Generation 0)
        # We assume Generation 0 rationales already exist (from `generate_rationales.py`)
        print("\n" + "="*80)
        print("STARTING SEQUENTIAL DISTILLATION")
        print("="*80)
        
        previous_model_path = None
        
        for i, student_model_name in enumerate(self.student_chain):
            generation_idx = i + 1
            print(f"\n" + "="*60)
            print(f"GENERATION {generation_idx}: Training {student_model_name}")
            print(f"Teacher: {'GPT-3.5' if i == 0 else self.student_chain[i-1]}")
            print("="*60)
            
            # Step 1: Prepare Rationales (Generate if not Gen 0)
            rationale_path = self._prepare_rationales(generation_idx, previous_model_path)
            
            # Step 2: Train Token Weighting for this generation
            self._train_token_weighting(generation_idx, rationale_path, student_model_name)
            
            # Step 3: Calculate Difficulties
            difficulties = self._calculate_difficulties(generation_idx, rationale_path, student_model_name)
            
            # Step 4: Train Student
            previous_model_path = self._train_student(
                generation_idx, 
                student_model_name, 
                rationale_path, 
                difficulties
            )
            
        print("\n" + "="*80)
        print("SEQUENTIAL DISTILLATION COMPLETE")
        print("="*80)

    def _prepare_rationales(self, generation_idx: int, teacher_model_path: str) -> str:
        """Get rationales for current generation"""
        
        # Define path for this generation's rationales
        filename = f"{self.dataset_name}_gen{generation_idx}_rationales.json"
        save_path = os.path.join(self.config.data.rationale_dir, filename)
        
        # For Generation 1, we use the pre-generated GPT-3.5 rationales
        if generation_idx == 1:
            gpt_path = os.path.join(self.config.data.rationale_dir, f"{self.dataset_name}_train_rationales.json")
            # If the specific file doesn't exist, try the one from generate_500_rationales
            if not os.path.exists(gpt_path):
                alt_path = os.path.join(self.config.data.rationale_dir, f"{self.dataset_name}_train_500_rationales.json")
                if os.path.exists(alt_path):
                    print(f"Standard rationales not found, using subset: {alt_path}")
                    return alt_path
            
            if not os.path.exists(gpt_path):
                raise FileNotFoundError(f"Generation 1 requires GPT-3.5 rationales at {gpt_path}")
            print(f"Using existing GPT-3.5 rationales: {gpt_path}")
            return gpt_path
            
        # For subsequent generations, generate using previous student
        if os.path.exists(save_path):
            print(f"Found existing rationales for Gen {generation_idx}: {save_path}")
            return save_path
            
        print(f"Generating rationales for Gen {generation_idx} using teacher from {teacher_model_path}...")
        
        # Load teacher model
        teacher_model_name = self.student_chain[generation_idx-2] # Previous student name
        teacher = StudentModel(teacher_model_name, device=self.device)
        teacher.load_pretrained(teacher_model_path)
        
        # Load questions
        train_dataset, _, _ = load_dataset_by_name(self.dataset_name, self.config.data.cache_dir)
        questions = train_dataset.questions
        
        # Generate
        generator = StudentGenerator(teacher, device=self.device)
        generator.generate_rationales(questions, save_path=save_path)
        
        return save_path

    def _train_token_weighting(self, generation_idx: int, rationale_path: str, model_name: str):
        """Train token weighting module for this generation"""
        print(f"Training Token Weighting Module for Gen {generation_idx}...")
        
        # Load data with specific rationales
        train_dataset, val_dataset, _ = load_dataset_by_name(self.dataset_name, self.config.data.cache_dir)
        rationales = load_rationales(rationale_path)
        
        # Ensure dataset size matches rationales (handling subsets)
        if len(train_dataset) != len(rationales):
            print(f"Dataset size ({len(train_dataset)}) != Rationales ({len(rationales)}). Truncating dataset.")
            
            from data.load_datasets import ReasoningDataset
            
            new_questions = train_dataset.questions[:len(rationales)]
            new_answers = train_dataset.answers[:len(rationales)]
            
            train_dataset = ReasoningDataset(
                questions=new_questions,
                answers=new_answers,
                rationales=None,
                split='train'
            )
            
            # Also truncate val set to ensure it has rationales if we use it
            # The trainer uses val_dataset for evaluation, and attach_rationales_to_dataset expects matching lengths
            # Since we don't have validation rationales usually, we use a subset of training data as validation in standard pipeline
            # Let's mirror that logic here: create a small val set from the truncated training data
            val_size = min(len(train_dataset), 50) # Use 50 samples for validation
            val_questions = new_questions[:val_size]
            val_answers = new_answers[:val_size]
            val_dataset = ReasoningDataset(
                questions=val_questions,
                answers=val_answers,
                rationales=None,
                split='val'
            )
            val_rationales = rationales[:val_size]
            val_dataset = attach_rationales_to_dataset(val_dataset, val_rationales)
        
        train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
        
        # Initialize module
        module = TokenWeightingModule(
            model_name=model_name,
            hidden_dim=self.config.model.weight_generator_hidden_dim,
            device=self.device
        )
        
        # Train
        trainer = TokenWeightingTrainer(
            module=module,
            train_dataset=train_dataset,
            val_dataset=val_dataset, # Note: using raw val dataset
            config=self.config,
            device=self.device
        )
        trainer.train(num_epochs=3) # Reduced epochs for intermediate steps
        
        # Save
        save_path = os.path.join(self.config.checkpoint_dir, f"gen{generation_idx}_token_weighting.pt")
        torch.save({
            'weight_generator_state_dict': module.weight_generator.state_dict()
        }, save_path)
        
        return save_path

    def _calculate_difficulties(self, generation_idx: int, rationale_path: str, model_name: str):
        """Calculate step difficulties for this generation"""
        print(f"Calculating difficulties for Gen {generation_idx}...")
        
        train_dataset, _, _ = load_dataset_by_name(self.dataset_name, self.config.data.cache_dir)
        rationales = load_rationales(rationale_path)
        
        # Ensure dataset size matches rationales
        if len(train_dataset) != len(rationales):
            print(f"Dataset size ({len(train_dataset)}) != Rationales ({len(rationales)}). Truncating dataset.")
            train_dataset.questions = train_dataset.questions[:len(rationales)]
            train_dataset.answers = train_dataset.answers[:len(rationales)]
            
        train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
        
        model = StudentModel(model_name, device=self.device)
        
        # Dummy weights for calculation phase
        token_weights = {i: torch.ones(50) for i in range(len(train_dataset))}
        
        calculator = StepDifficultyCalculator(model, token_weights, device=self.device)
        difficulties = calculator.calculate_difficulties_for_dataset(train_dataset)
        
        # Save
        save_path = os.path.join("./data/difficulties", f"{self.dataset_name}_gen{generation_idx}_difficulties.json")
        save_difficulties(difficulties, save_path)
        
        return difficulties

    def _train_student(self, generation_idx: int, model_name: str, rationale_path: str, difficulties):
        """Train the student model using KPOD"""
        print(f"Training Student (Gen {generation_idx}): {model_name}...")
        
        # Load Data
        train_dataset, val_dataset, _ = load_dataset_by_name(self.dataset_name, self.config.data.cache_dir)
        rationales = load_rationales(rationale_path)
        
        # Ensure dataset size matches rationales
        if len(train_dataset) != len(rationales):
            print(f"Dataset size ({len(train_dataset)}) != Rationales ({len(rationales)}). Truncating dataset.")
            train_dataset.questions = train_dataset.questions[:len(rationales)]
            train_dataset.answers = train_dataset.answers[:len(rationales)]
            
        train_dataset = attach_rationales_to_dataset(train_dataset, rationales)
        
        # Clustering (reuse or re-cluster? Let's reuse basic clusters for simplicity)
        clusters_path = os.path.join("./data/clusters", f"{self.dataset_name}_clusters.json")
        if not os.path.exists(clusters_path):
             # Create clusters if they don't exist (e.g. first run)
             clusterer = QuestionClusterer(num_clusters=self.config.progressive.num_clusters)
             clusterer.fit(train_dataset.questions, embedding_model="glove")
             clusterer.save(clusters_path)
        else:
             clusterer = QuestionClusterer()
             clusterer.load(clusters_path)

        # Initialize Models
        student_model = StudentModel(
            model_name=model_name,
            lora_r=self.config.model.lora_r_flan,
            device=self.device
        )
        
        token_weighting_module = TokenWeightingModule(
            model_name=model_name,
            hidden_dim=self.config.model.weight_generator_hidden_dim,
            device=self.device
        )
        
        # Load weights
        weights_path = os.path.join(self.config.checkpoint_dir, f"gen{generation_idx}_token_weighting.pt")
        checkpoint = torch.load(weights_path)
        token_weighting_module.weight_generator.load_state_dict(checkpoint['weight_generator_state_dict'])
        
        # Scheduler
        scheduler = ProgressiveScheduler(
            step_difficulties=difficulties,
            cluster_assignments=clusterer.cluster_assignments,
            num_epochs=self.config.training.epochs_flan,
            p=self.config.progressive.p,
            c0_ratio=self.config.progressive.c0_ratio,
            beta=self.config.progressive.beta,
            delta_s=self.config.progressive.delta_s
        )
        
        # Trainer
        trainer = KPODTrainer(
            student_model=student_model,
            token_weighting_module=token_weighting_module,
            progressive_scheduler=scheduler,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=self.config,
            device=self.device
        )
        
        trainer.train()
        
        # Save Final Model
        final_path = os.path.join(self.config.checkpoint_dir, f"kpod_gen{generation_idx}_{model_name.split('/')[-1]}")
        student_model.save_pretrained(final_path)
        
        return final_path

