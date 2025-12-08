"""
Question clustering for diversity in progressive distillation
Uses K-means clustering on question embeddings
"""
import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict
import json
import os


class QuestionClusterer:
    """Cluster questions for diversity-based selection"""
    
    def __init__(self, num_clusters: int = 5):
        self.num_clusters = num_clusters
        self.kmeans = None
        self.cluster_assignments = None
    
    def get_question_embeddings(
        self,
        questions: List[str],
        embedding_model: str = "glove",
        device: str = "cpu"
    ) -> np.ndarray:
        """
        Get embeddings for questions
        Paper uses GloVe word embeddings + averaging
        
        Args:
            questions: List of question texts
            embedding_model: Type of embedding ("glove" or "transformer")
            device: Device to use
        Returns:
            embeddings: [num_questions, embedding_dim]
        """
        if embedding_model == "glove":
            # Use simple approach: load pre-trained GloVe or use TF-IDF as proxy
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            vectorizer = TfidfVectorizer(max_features=300)  # 300-dim like GloVe
            embeddings = vectorizer.fit_transform(questions).toarray()
            
        elif embedding_model == "transformer":
            # Alternative: use sentence transformers
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(questions, show_progress_bar=True)
            
        else:
            raise ValueError(f"Unknown embedding model: {embedding_model}")
        
        return embeddings
    
    def fit(self, questions: List[str], embedding_model: str = "glove"):
        """
        Fit K-means clustering on questions
        
        Args:
            questions: List of question texts
            embedding_model: Type of embedding to use
        """
        print(f"Getting embeddings for {len(questions)} questions...")
        embeddings = self.get_question_embeddings(questions, embedding_model)
        
        print(f"Fitting K-means with {self.num_clusters} clusters...")
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.cluster_assignments = self.kmeans.fit_predict(embeddings)
        
        # Print cluster sizes
        unique, counts = np.unique(self.cluster_assignments, return_counts=True)
        print("Cluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} questions")
        
        return self.cluster_assignments
    
    def get_cluster_for_question(self, question_idx: int) -> int:
        """Get cluster assignment for a question"""
        if self.cluster_assignments is None:
            raise ValueError("Must call fit() first")
        return self.cluster_assignments[question_idx]
    
    def get_questions_in_cluster(self, cluster_id: int) -> List[int]:
        """Get all question indices in a cluster"""
        if self.cluster_assignments is None:
            raise ValueError("Must call fit() first")
        return np.where(self.cluster_assignments == cluster_id)[0].tolist()
    
    def get_cluster_distribution(self, question_indices: List[int]) -> Dict[int, int]:
        """
        Get distribution of clusters in a set of questions
        
        Args:
            question_indices: List of question indices
        Returns:
            distribution: Dict mapping cluster_id -> count
        """
        if self.cluster_assignments is None:
            raise ValueError("Must call fit() first")
        
        distribution = {}
        for idx in question_indices:
            cluster_id = self.cluster_assignments[idx]
            distribution[cluster_id] = distribution.get(cluster_id, 0) + 1
        
        return distribution
    
    def save(self, save_path: str):
        """Save cluster assignments"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        data = {
            'num_clusters': self.num_clusters,
            'cluster_assignments': self.cluster_assignments.tolist()
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Cluster assignments saved to {save_path}")
    
    def load(self, load_path: str):
        """Load cluster assignments"""
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        self.num_clusters = data['num_clusters']
        self.cluster_assignments = np.array(data['cluster_assignments'])
        
        print(f"Loaded cluster assignments for {len(self.cluster_assignments)} questions")


# Example usage
if __name__ == "__main__":
    from data.load_datasets import load_gsm8k
    
    print("Loading dataset...")
    train_dataset, _, _ = load_gsm8k()
    
    # Get questions
    questions = [sample['question'] for sample in train_dataset]
    
    print(f"\nClustering {len(questions)} questions...")
    clusterer = QuestionClusterer(num_clusters=5)
    
    # Fit (using first 1000 for speed)
    cluster_assignments = clusterer.fit(questions[:1000], embedding_model="glove")
    
    # Show some examples from each cluster
    print("\nSample questions from each cluster:")
    for cluster_id in range(5):
        indices = clusterer.get_questions_in_cluster(cluster_id)
        print(f"\nCluster {cluster_id} ({len(indices)} questions):")
        for idx in indices[:3]:  # Show 3 examples
            print(f"  - {questions[idx][:100]}...")
    
    # Save
    save_path = "./data/clusters/gsm8k_clusters.json"
    clusterer.save(save_path)
    
    print("\n✓ Question clustering complete!")