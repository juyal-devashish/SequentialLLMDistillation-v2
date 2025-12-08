"""
Progressive distillation scheduler
Implements the value function and question selection (Equations 9-13)
"""
import torch
import numpy as np
from typing import List, Dict, Set, Tuple
import json


class ProgressiveScheduler:
    """
    Schedules progressive distillation with diversity-aware question selection
    Implements value function F(S(t)) from Equation 12
    """
    
    def __init__(
        self,
        step_difficulties: Dict[int, List[float]],
        cluster_assignments: np.ndarray,
        num_epochs: int,
        p: float = 0.5,
        c0_ratio: float = 0.3,
        beta: float = 12.0,
        delta_s: int = 1,
        num_clusters: int = 5
    ):
        """
        Args:
            step_difficulties: Dict mapping sample_idx -> list of step difficulties
            cluster_assignments: Array of cluster IDs for each question
            num_epochs: Total number of training epochs
            p: Growth rate exponent (default: 0.5)
            c0_ratio: Initial difficulty ratio (default: 0.3 = 30%)
            beta: Diversity weight (default: 12.0)
            delta_s: Number of steps to reduce per stage (default: 1)
            num_clusters: Number of clusters
        """
        self.step_difficulties = step_difficulties
        self.cluster_assignments = cluster_assignments
        self.num_epochs = num_epochs
        self.p = p
        self.c0_ratio = c0_ratio
        self.beta = beta
        self.delta_s = delta_s
        self.num_clusters = num_clusters
        
        # Calculate T (stage to reach maximum difficulty)
        self.T = num_epochs // 2
        
        # Calculate maximum difficulty B (sum of all step difficulties)
        self.B = self._calculate_max_difficulty()
        
        # Calculate initial difficulty C0
        self.C0 = self.c0_ratio * self.B
        
        # Calculate growth parameter u (Equation 10)
        self.u = (self.B - self.C0) * (self.p + 1) / (self.T ** (self.p + 1))
        
        # Initialize current state
        self.current_input_steps = {idx: self._get_num_steps(idx) - 1 for idx in step_difficulties.keys()}
        self.current_difficulty = self._calculate_current_difficulty()
        
        print(f"Progressive Scheduler initialized:")
        print(f"  Total samples: {len(step_difficulties)}")
        print(f"  Max difficulty B: {self.B:.2f}")
        print(f"  Initial difficulty C0: {self.C0:.2f}")
        print(f"  Growth parameter u: {self.u:.4f}")
        print(f"  Stage T (max difficulty): {self.T}")
    
    def _get_num_steps(self, sample_idx: int) -> int:
        """Get number of steps for a sample"""
        return len(self.step_difficulties.get(sample_idx, [1]))
    
    def _calculate_max_difficulty(self) -> float:
        """Calculate maximum difficulty B (Equation 10)"""
        total = 0.0
        for sample_idx, diffs in self.step_difficulties.items():
            total += sum(diffs)
        return total
    
    def _calculate_current_difficulty(self) -> float:
        """Calculate current total difficulty"""
        total = 0.0
        for sample_idx, num_input_steps in self.current_input_steps.items():
            diffs = self.step_difficulties[sample_idx]
            # Difficulty is sum of output steps (after num_input_steps)
            output_diffs = diffs[num_input_steps + 1:] if num_input_steps + 1 < len(diffs) else []
            total += sum(output_diffs)
        return total
    
    def get_target_difficulty(self, stage: int) -> float:
        """
        Calculate target difficulty D(t) for a stage
        Implements Equation 10
        
        Args:
            stage: Current stage (epoch)
        Returns:
            target_difficulty: Target total difficulty for this stage
        """
        if stage >= self.T:
            return self.B
        
        D_t = (self.u * (stage ** (self.p + 1))) / (self.p + 1) + self.C0
        return min(D_t, self.B)
    
    def calculate_value_function(
        self,
        selected_samples: Set[int],
        delta_H: float,
        delta_D: float
    ) -> float:
        """
        Calculate value function F(S(t)) from Equation 12
        
        Args:
            selected_samples: Set of selected sample indices
            delta_H: Increased difficulty from selection
            delta_D: Target difficulty increase
        Returns:
            value: Value function score
        """
        # First term: negative distance from target
        distance_term = -(delta_D - delta_H)
        
        # Second term: diversity (sqrt of cluster intersection sizes)
        diversity_term = 0.0
        for cluster_id in range(self.num_clusters):
            # Count how many selected samples are in this cluster
            count = sum(1 for idx in selected_samples 
                       if self.cluster_assignments[idx] == cluster_id)
            diversity_term += np.sqrt(count)
        
        diversity_term *= self.beta
        
        value = distance_term + diversity_term
        return value
    
    def calculate_sample_difficulty_increase(self, sample_idx: int) -> float:
        """
        Calculate difficulty increase if we reduce input steps for a sample
        Implements h_i(S(t)) from Equation 9
        
        Args:
            sample_idx: Sample index
        Returns:
            difficulty_increase: Difficulty added by reducing delta_s input steps
        """
        current_input_steps = self.current_input_steps[sample_idx]
        new_input_steps = max(0, current_input_steps - self.delta_s)
        
        diffs = self.step_difficulties[sample_idx]
        
        # Current output steps difficulty
        current_output_diffs = diffs[current_input_steps + 1:] if current_input_steps + 1 < len(diffs) else []
        current_difficulty = sum(current_output_diffs)
        
        # New output steps difficulty (more steps to generate)
        new_output_diffs = diffs[new_input_steps + 1:] if new_input_steps + 1 < len(diffs) else []
        new_difficulty = sum(new_output_diffs)
        
        increase = new_difficulty - current_difficulty
        return max(0.0, increase)  # Should be positive
    
    def select_samples_for_stage(
        self,
        stage: int,
        use_greedy: bool = True
    ) -> Set[int]:
        """
        Select samples to increase difficulty for this stage
        Solves Equation 13 (approximately using greedy algorithm)
        
        Args:
            stage: Current stage (epoch)
            use_greedy: Whether to use greedy approximation (vs brute force)
        Returns:
            selected_samples: Set of sample indices to increase difficulty
        """
        target_difficulty = self.get_target_difficulty(stage)
        delta_D = target_difficulty - self.current_difficulty
        
        if delta_D <= 0:
            # Already at or above target
            return set()
        
        print(f"\nStage {stage}: Target difficulty increase: {delta_D:.2f}")
        
        # Greedy selection (approximates submodular optimization)
        selected = set()
        current_delta_H = 0.0
        
        # Get candidate samples (those that can still be made harder)
        candidates = [
            idx for idx, num_input_steps in self.current_input_steps.items()
            if num_input_steps > 0
        ]
        
        # Calculate difficulty increase for each candidate
        candidate_difficulties = {
            idx: self.calculate_sample_difficulty_increase(idx)
            for idx in candidates
        }
        
        # Greedy selection
        while current_delta_H < delta_D and candidates:
            best_value = float('-inf')
            best_candidate = None
            
            for candidate in candidates:
                # Try adding this candidate
                test_selected = selected.copy()
                test_selected.add(candidate)
                
                test_delta_H = current_delta_H + candidate_difficulties[candidate]
                
                # Check constraint
                if test_delta_H > delta_D:
                    continue
                
                # Calculate value function
                value = self.calculate_value_function(test_selected, test_delta_H, delta_D)
                
                if value > best_value:
                    best_value = value
                    best_candidate = candidate
            
            if best_candidate is None:
                break
            
            # Add best candidate
            selected.add(best_candidate)
            current_delta_H += candidate_difficulties[best_candidate]
            candidates.remove(best_candidate)
        
        print(f"  Selected {len(selected)} samples")
        print(f"  Actual difficulty increase: {current_delta_H:.2f}")
        
        # Show cluster diversity
        cluster_dist = {}
        for idx in selected:
            cluster_id = self.cluster_assignments[idx]
            cluster_dist[cluster_id] = cluster_dist.get(cluster_id, 0) + 1
        print(f"  Cluster distribution: {cluster_dist}")
        
        return selected
    
    def update_stage(self, stage: int):
        """
        Update input steps for selected samples at this stage
        Implements Equation 11
        
        Args:
            stage: Current stage (epoch)
        """
        if stage > self.T:
            # After T, train on full rationales (no input steps)
            for idx in self.current_input_steps.keys():
                self.current_input_steps[idx] = 0
            self.current_difficulty = self.B
            return
        
        # Select samples to increase difficulty
        selected_samples = self.select_samples_for_stage(stage)
        
        # Update input steps for selected samples (Equation 11)
        for idx in selected_samples:
            self.current_input_steps[idx] = max(
                0,
                self.current_input_steps[idx] - self.delta_s
            )
        
        # Recalculate current difficulty
        self.current_difficulty = self._calculate_current_difficulty()
        
        print(f"  New total difficulty: {self.current_difficulty:.2f}")
    
    def get_input_steps_for_sample(self, sample_idx: int) -> int:
        """Get current number of input steps for a sample"""
        return self.current_input_steps.get(sample_idx, 0)
    
    def save_schedule(self, save_path: str):
        """Save current schedule state"""
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'current_input_steps': {str(k): v for k, v in self.current_input_steps.items()},
            'current_difficulty': self.current_difficulty,
            'hyperparameters': {
                'p': self.p,
                'c0_ratio': self.c0_ratio,
                'beta': self.beta,
                'delta_s': self.delta_s,
                'T': self.T,
                'B': self.B,
                'C0': self.C0,
                'u': self.u
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Schedule saved to {save_path}")


# Example usage
if __name__ == "__main__":
    from training.step_difficulty import load_difficulties
    from training.question_clustering import QuestionClusterer
    import os
    
    print("Loading step difficulties...")
    difficulties_path = "./data/difficulties/gsm8k_train_difficulties.json"
    if not os.path.exists(difficulties_path):
        print("Difficulties file not found. Please calculate difficulties first.")
        exit(1)
    
    difficulties = load_difficulties(difficulties_path)
    
    print("\nLoading cluster assignments...")
    clusters_path = "./data/clusters/gsm8k_clusters.json"
    if not os.path.exists(clusters_path):
        print("Clusters file not found. Please run clustering first.")
        exit(1)
    
    clusterer = QuestionClusterer()
    clusterer.load(clusters_path)
    
    # Initialize scheduler
    print("\nInitializing progressive scheduler...")
    scheduler = ProgressiveScheduler(
        step_difficulties=difficulties,
        cluster_assignments=clusterer.cluster_assignments,
        num_epochs=20,
        p=0.5,
        c0_ratio=0.3,
        beta=12.0,
        delta_s=1,
        num_clusters=5
    )
    
    # Simulate progression through stages
    print("\nSimulating progressive stages:")
    for stage in range(1, 11):  # First 10 stages
        print(f"\n{'='*60}")
        print(f"Stage {stage}")
        print(f"{'='*60}")
        
        target_diff = scheduler.get_target_difficulty(stage)
        print(f"Target difficulty: {target_diff:.2f}")
        print(f"Current difficulty: {scheduler.current_difficulty:.2f}")
        
        scheduler.update_stage(stage)
        
        # Show some sample input step counts
        sample_indices = list(difficulties.keys())[:5]
        print(f"\nSample input steps:")
        for idx in sample_indices:
            num_steps = scheduler._get_num_steps(idx)
            input_steps = scheduler.get_input_steps_for_sample(idx)
            output_steps = num_steps - input_steps - 1
            print(f"  Sample {idx}: {input_steps} input, {output_steps} output (of {num_steps} total)")
    
    # Save schedule
    save_path = "./data/schedules/gsm8k_schedule.json"
    scheduler.save_schedule(save_path)
    
    print("\n✓ Progressive scheduler test complete!")