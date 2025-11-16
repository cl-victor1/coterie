"""
Realism evaluator for assessing persona behavior authenticity.
Uses data-grounded approach with real user review embeddings.

Criterion 1 (40% weight): Journey-to-Matched-Reviews Similarity
- Finds Top 100 most similar reviews for each persona
- Compares journey embedding against average of matched reviews
- Ranks personas by similarity to real user experiences

Criterion 2 (40% weight): Persona-Journey Embedding Consistency
- Computes cosine similarity between persona and journey embeddings
- Evaluates consistency between persona definition and actual behavior
- Higher scores indicate persona's actions align with their characteristics

Criterion 3 (20% weight): Persona-Journey to Cluster Centroids Similarity
- Computes average of persona and journey embeddings
- Compares against 3 real user cluster centroids
- Takes maximum similarity to measure alignment with user segments
"""

import json
import os
import re
from typing import Dict, Any, List, Tuple, Optional, Set
from datetime import datetime
import numpy as np


class RealismEvaluator:
    """
    Evaluates the realism of persona behaviors using multiple data-grounded criteria.
    Implements comprehensive evaluation against 8804 real user reviews.

    Evaluation Criteria:
    - Criterion 1 (40% weight): Journey-to-Matched-Reviews Similarity
      Uses embedding similarity to match persona journeys with real user reviews
    
    - Criterion 2 (40% weight): Persona-Journey Embedding Consistency
      Evaluates consistency between persona definition and actual journey behavior
      Computes cosine similarity between persona and journey embeddings
    
    - Criterion 3 (20% weight): Persona-Journey to Cluster Centroids Similarity
      Compares average of persona and journey embeddings against real user cluster centroids
      Takes maximum similarity across 3 clusters to measure alignment with user segments
    
    Total Maximum Score: 100% (from all implemented criteria)
    """

    def __init__(
        self,
        persona_embeddings_path: Optional[str] = None,
        review_embeddings_path: Optional[str] = None,
        journey_embeddings_path: Optional[str] = None,
        cluster_results_path: Optional[str] = None
    ):
        """
        Initialize realism evaluator with embedding file paths.

        Args:
            persona_embeddings_path: Path to persona embeddings JSON
            review_embeddings_path: Path to review embeddings JSON (8804 reviews)
            journey_embeddings_path: Path to journey embeddings JSON
            cluster_results_path: Path to cluster results JSON (for Criterion 3)
        """
        # Default paths to latest embeddings
        eval_dir = "/Users/victor_official/AI personas/evaluation"

        self.persona_embeddings_path = persona_embeddings_path or os.path.join(
            eval_dir, "persona_embeddings_20251115_175110.json"
        )
        self.review_embeddings_path = review_embeddings_path or os.path.join(
            eval_dir, "review_embeddings_20251115_181818.json"
        )
        self.journey_embeddings_path = journey_embeddings_path
        self.cluster_results_path = cluster_results_path or os.path.join(
            eval_dir, "cluster_results_20251115_195944.json"
        )

        # Cache for loaded embeddings and data
        self._persona_embeddings = None
        self._review_embeddings = None
        self._journey_embeddings = None
        self._cluster_centroids = None

        # Criterion weights
        self.criterion_weights = {
            "journey_to_reviews": 0.40,  # Criterion 1
            "persona_journey_consistency": 0.40,  # Criterion 2
            "persona_journey_to_clusters": 0.20,  # Criterion 3
        }

    def _load_embeddings(self, file_path: str, embedding_key: str = "embeddings") -> Dict[str, Any]:
        """
        Load embeddings from JSON file.

        Args:
            file_path: Path to embeddings JSON file
            embedding_key: Key to access embeddings list in JSON

        Returns:
            Dictionary with metadata and embeddings

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON structure is invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate structure
            if embedding_key not in data:
                raise ValueError(f"Missing '{embedding_key}' key in {file_path}")

            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {str(e)}")

    def _compute_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec_a: First vector (n-dimensional)
            vec_b: Second vector (n-dimensional)

        Returns:
            Cosine similarity in range [0, 1] (normalized from [-1, 1])
        """
        # Ensure numpy arrays
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)

        # Compute cosine similarity
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        cosine_sim = dot_product / (norm_a * norm_b)

        # Normalize from [-1, 1] to [0, 1]
        normalized_sim = (cosine_sim + 1) / 2

        return float(normalized_sim)

    def _find_top_k_similar_reviews(
        self,
        persona_embedding: np.ndarray,
        review_embeddings: List[Dict[str, Any]],
        k: int = 100
    ) -> Tuple[List[int], List[float], List[Dict[str, Any]]]:
        """
        Find top K most similar reviews to a persona embedding.

        Args:
            persona_embedding: Persona embedding vector (3072-dim)
            review_embeddings: List of review embedding dictionaries
            k: Number of top reviews to return

        Returns:
            Tuple of (indices, similarities, review_metadata)
        """
        similarities = []

        # Compute similarity for all reviews
        for idx, review in enumerate(review_embeddings):
            review_emb = np.array(review['embedding'])
            sim = self._compute_cosine_similarity(persona_embedding, review_emb)
            similarities.append((idx, sim, review))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Extract top K
        top_k = similarities[:k]
        indices = [item[0] for item in top_k]
        sims = [item[1] for item in top_k]
        metadata = [
            {
                "id": item[2]["id"],
                "title": item[2].get("title", ""),
                "score": item[2].get("score", 0),
                "similarity": item[1]
            }
            for item in top_k
        ]

        return indices, sims, metadata

    def _compute_cluster_centroids(self) -> Dict[int, np.ndarray]:
        """
        Compute cluster centroids from cluster results and review embeddings.

        Returns:
            Dictionary mapping cluster_id to centroid embedding (np.ndarray)

        Raises:
            FileNotFoundError: If cluster results or review embeddings not found
            ValueError: If data structure is invalid
        """
        try:
            # Load cluster results
            if not os.path.exists(self.cluster_results_path):
                raise FileNotFoundError(f"Cluster results file not found: {self.cluster_results_path}")

            with open(self.cluster_results_path, 'r', encoding='utf-8') as f:
                cluster_data = json.load(f)

            # Load review embeddings if not cached
            if self._review_embeddings is None:
                review_data_file = self._load_embeddings(
                    self.review_embeddings_path,
                    embedding_key="embeddings"
                )
                self._review_embeddings = review_data_file["embeddings"]

            # Build mapping from review ID to embedding
            review_id_to_embedding = {
                emb['id']: np.array(emb['embedding'])
                for emb in self._review_embeddings
                if emb['embedding'] is not None
            }

            # Group embeddings by cluster
            cluster_embeddings = {}
            for review in cluster_data.get('reviews', []):
                review_id = review['id']
                cluster_id = review['cluster']
                
                # Skip noise points (cluster == -1)
                if cluster_id == -1:
                    continue
                
                if review_id in review_id_to_embedding:
                    if cluster_id not in cluster_embeddings:
                        cluster_embeddings[cluster_id] = []
                    cluster_embeddings[cluster_id].append(review_id_to_embedding[review_id])

            # Compute centroids
            centroids = {}
            for cluster_id, embeddings in cluster_embeddings.items():
                if embeddings:
                    centroids[cluster_id] = np.mean(embeddings, axis=0)

            return centroids

        except Exception as e:
            raise ValueError(f"Failed to compute cluster centroids: {str(e)}")

    def _compute_criterion1_score(
        self,
        persona_key: str,
        persona_embedding: np.ndarray,
        journey_embedding: np.ndarray,
        review_embeddings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute Criterion 1: Journey-to-Matched-Reviews Similarity.

        Steps:
        1. Find Top 100 reviews most similar to persona
        2. Compute average embedding of these 100 reviews
        3. Compute similarity between journey and average review embedding

        Args:
            persona_key: Persona identifier (e.g., 'sarah_kim')
            persona_embedding: Persona embedding vector
            journey_embedding: Journey embedding vector
            review_embeddings: All review embeddings

        Returns:
            Dictionary with Criterion 1 scores and metadata
        """
        # Step 1: Find Top 100 reviews similar to persona
        indices, similarities, metadata = self._find_top_k_similar_reviews(
            persona_embedding,
            review_embeddings,
            k=100
        )

        # Step 2: Compute average embedding of Top 100 reviews
        top_100_embeddings = [
            np.array(review_embeddings[idx]['embedding'])
            for idx in indices
        ]
        avg_review_embedding = np.mean(top_100_embeddings, axis=0)

        # Step 3: Compute similarity between journey and average review
        journey_to_reviews_similarity = self._compute_cosine_similarity(
            journey_embedding,
            avg_review_embedding
        )

        # Compute average similarity of persona to matched reviews
        avg_persona_to_reviews_sim = np.mean(similarities)

        # Prepare result
        result = {
            "persona_key": persona_key,
            "score": journey_to_reviews_similarity,
            "weighted_score": journey_to_reviews_similarity * self.criterion_weights["journey_to_reviews"],
            "matched_reviews_count": len(indices),
            "avg_persona_to_reviews_similarity": float(avg_persona_to_reviews_sim),
            "avg_journey_to_reviews_similarity": journey_to_reviews_similarity,
            "top_10_matched_review_ids": [m["id"] for m in metadata[:10]],
            "top_10_matched_reviews_metadata": metadata[:10]
        }

        return result

    def evaluate_persona_realism(
        self,
        persona_data: Dict[str, Any],
        action_log: List[Dict],
        journey_embeddings_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate realism for a single persona using multiple criteria.

        Criterion 1 (40%): Journey-to-Matched-Reviews Similarity
        Criterion 2 (40%): Persona-Journey Embedding Consistency
        Criterion 3 (20%): Persona-Journey to Cluster Centroids Similarity

        Args:
            persona_data: Persona definition and characteristics
            action_log: List of actions taken during testing
            journey_embeddings_path: Path to journey embeddings (optional override)

        Returns:
            Realism scores and analysis for this persona
        """
        try:
            # Load embeddings
            if self._persona_embeddings is None:
                persona_data_file = self._load_embeddings(
                    self.persona_embeddings_path,
                    embedding_key="personas"
                )
                self._persona_embeddings = persona_data_file["personas"]

            if self._review_embeddings is None:
                review_data_file = self._load_embeddings(
                    self.review_embeddings_path,
                    embedding_key="embeddings"
                )
                self._review_embeddings = review_data_file["embeddings"]

            # Load journey embeddings
            journey_path = journey_embeddings_path or self.journey_embeddings_path
            if not journey_path:
                raise ValueError("Journey embeddings path not provided")

            journey_data = self._load_embeddings(journey_path, embedding_key="journeys")

            # Find persona embedding
            persona_key = persona_data.get("key") or persona_data.get("persona_key")
            persona_name = persona_data.get("name")

            persona_emb = None
            for p in self._persona_embeddings:
                if p["key"] == persona_key or p["name"] == persona_name:
                    persona_emb = np.array(p["embedding"])
                    break

            if persona_emb is None:
                raise ValueError(f"Persona embedding not found for {persona_key or persona_name}")

            # Find journey embedding
            journey_emb = None
            for j in journey_data["journeys"]:
                if j["persona_name"] == persona_name:
                    journey_emb = np.array(j["embedding"])
                    break

            if journey_emb is None:
                raise ValueError(f"Journey embedding not found for {persona_name}")

            # Compute Criterion 1 score
            criterion1_result = self._compute_criterion1_score(
                persona_key,
                persona_emb,
                journey_emb,
                self._review_embeddings
            )

            # Compute Criterion 2 score (Persona-Journey Embedding Consistency)
            criterion2_result = self._compute_criterion2_score(
                persona_key,
                persona_emb,
                journey_emb
            )

            # Compute Criterion 3 score (Persona-Journey to Cluster Centroids)
            # Load cluster centroids if not cached
            if self._cluster_centroids is None:
                self._cluster_centroids = self._compute_cluster_centroids()
            
            criterion3_result = self._compute_criterion3_score(
                persona_key,
                persona_emb,
                journey_emb,
                self._cluster_centroids
            )

            # Calculate overall realism score
            overall_score = (criterion1_result["weighted_score"] + 
                           criterion2_result["weighted_score"] + 
                           criterion3_result["weighted_score"])

            # Prepare final result
            result = {
                "persona_key": persona_key,
                "persona_name": persona_name,
                "criterion_1_journey_to_reviews": criterion1_result,
                "criterion_2_persona_journey_consistency": criterion2_result,
                "criterion_3_persona_journey_to_clusters": criterion3_result,
                "overall_realism_score": overall_score,
                "methodology": "Multi-criteria realism evaluation (Criterion 1: 40%, Criterion 2: 40%, Criterion 3: 20%)"
            }

            return result

        except Exception as e:
            # Return error result instead of crashing
            return {
                "error": str(e),
                "persona_key": persona_data.get("key", "unknown"),
                "persona_name": persona_data.get("name", "unknown"),
                "overall_realism_score": 0.0,
                "methodology": "Failed to compute - see error"
            }

    def compare_personas(
        self,
        all_persona_results: List[Dict[str, Any]],
        journey_embeddings_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare realism across all personas and rank them.

        Args:
            all_persona_results: List of persona test results
            journey_embeddings_path: Path to journey embeddings

        Returns:
            Comparative analysis with rankings
        """
        try:
            # Load all embeddings
            if self._persona_embeddings is None:
                persona_data_file = self._load_embeddings(
                    self.persona_embeddings_path,
                    embedding_key="personas"
                )
                self._persona_embeddings = persona_data_file["personas"]

            if self._review_embeddings is None:
                review_data_file = self._load_embeddings(
                    self.review_embeddings_path,
                    embedding_key="embeddings"
                )
                self._review_embeddings = review_data_file["embeddings"]

            journey_path = journey_embeddings_path or self.journey_embeddings_path
            if not journey_path:
                raise ValueError("Journey embeddings path not provided")

            journey_data = self._load_embeddings(journey_path, embedding_key="journeys")

            # Load cluster centroids if not cached
            if self._cluster_centroids is None:
                self._cluster_centroids = self._compute_cluster_centroids()

            # Compute scores for all personas
            persona_scores = []

            for persona_result in all_persona_results:
                persona_data = persona_result.get("persona", {})
                persona_name = persona_data.get("name")
                persona_key = None

                # Find persona key
                for p in self._persona_embeddings:
                    if p["name"] == persona_name:
                        persona_key = p["key"]
                        persona_emb = np.array(p["embedding"])
                        break

                if not persona_key:
                    continue

                # Find journey embedding
                journey_emb = None
                for j in journey_data["journeys"]:
                    if j["persona_name"] == persona_name:
                        journey_emb = np.array(j["embedding"])
                        break

                if journey_emb is None:
                    continue

                # Compute Criterion 1
                criterion1_result = self._compute_criterion1_score(
                    persona_key,
                    persona_emb,
                    journey_emb,
                    self._review_embeddings
                )

                # Compute Criterion 2
                criterion2_result = self._compute_criterion2_score(
                    persona_key,
                    persona_emb,
                    journey_emb
                )

                # Compute Criterion 3
                criterion3_result = self._compute_criterion3_score(
                    persona_key,
                    persona_emb,
                    journey_emb,
                    self._cluster_centroids
                )

                # Calculate overall score
                criterion2_score = criterion2_result["score"]
                criterion2_weighted = criterion2_result["weighted_score"]
                criterion3_score = criterion3_result["score"]
                criterion3_weighted = criterion3_result["weighted_score"]
                
                overall_weighted = (criterion1_result["weighted_score"] + 
                                  criterion2_weighted + 
                                  criterion3_weighted)

                persona_scores.append({
                    "persona_key": persona_key,
                    "persona_name": persona_name,
                    "criterion1_score": criterion1_result["score"],
                    "criterion1_weighted": criterion1_result["weighted_score"],
                    "criterion2_score": criterion2_score,
                    "criterion2_weighted": criterion2_weighted,
                    "criterion3_score": criterion3_score,
                    "criterion3_weighted": criterion3_weighted,
                    "weighted_score": overall_weighted,
                    "full_criterion1_result": criterion1_result,
                    "full_criterion2_result": criterion2_result,
                    "full_criterion3_result": criterion3_result
                })

            # Sort by weighted score (descending)
            persona_scores.sort(key=lambda x: x["weighted_score"], reverse=True)

            # Add rankings
            for rank, persona in enumerate(persona_scores, 1):
                persona["rank"] = rank

            # Prepare comparison table
            comparison_table = {
                persona["persona_name"]: {
                    "rank": persona["rank"],
                    "criterion1_score": round(persona["criterion1_score"], 4),
                    "criterion2_score": round(persona["criterion2_score"], 4),
                    "criterion3_score": round(persona["criterion3_score"], 4),
                    "overall_score": round(persona["weighted_score"], 4),
                    "criterion1_weighted": round(persona["criterion1_weighted"], 4),
                    "criterion2_weighted": round(persona["criterion2_weighted"], 4),
                    "criterion3_weighted": round(persona["criterion3_weighted"], 4)
                }
                for persona in persona_scores
            }

            result = {
                "methodology": "Multi-criteria evaluation: Criterion 1 (40%), Criterion 2 (40%), Criterion 3 (20%)",
                "total_personas_evaluated": len(persona_scores),
                "most_realistic": persona_scores[0]["persona_name"] if persona_scores else "N/A",
                "least_realistic": persona_scores[-1]["persona_name"] if persona_scores else "N/A",
                "comparison_table": comparison_table,
                "detailed_scores": persona_scores,
                "statistics": {
                    "mean_criterion1_score": float(np.mean([p["criterion1_score"] for p in persona_scores])),
                    "std_criterion1_score": float(np.std([p["criterion1_score"] for p in persona_scores])),
                    "min_criterion1_score": float(np.min([p["criterion1_score"] for p in persona_scores])),
                    "max_criterion1_score": float(np.max([p["criterion1_score"] for p in persona_scores])),
                    "mean_criterion2_score": float(np.mean([p["criterion2_score"] for p in persona_scores])),
                    "std_criterion2_score": float(np.std([p["criterion2_score"] for p in persona_scores])),
                    "min_criterion2_score": float(np.min([p["criterion2_score"] for p in persona_scores])),
                    "max_criterion2_score": float(np.max([p["criterion2_score"] for p in persona_scores])),
                    "mean_criterion3_score": float(np.mean([p["criterion3_score"] for p in persona_scores])),
                    "std_criterion3_score": float(np.std([p["criterion3_score"] for p in persona_scores])),
                    "min_criterion3_score": float(np.min([p["criterion3_score"] for p in persona_scores])),
                    "max_criterion3_score": float(np.max([p["criterion3_score"] for p in persona_scores])),
                    "mean_overall_score": float(np.mean([p["weighted_score"] for p in persona_scores])),
                    "std_overall_score": float(np.std([p["weighted_score"] for p in persona_scores]))
                }
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "methodology": "Failed to compare personas - see error",
                "comparison_table": {},
                "most_realistic": "N/A",
                "least_realistic": "N/A"
            }

    def _compute_criterion2_score(
        self,
        persona_key: str,
        persona_embedding: np.ndarray,
        journey_embedding: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute Criterion 2: Persona-Journey Embedding Consistency.

        Evaluates consistency between persona definition and actual journey behavior
        by computing cosine similarity between their embeddings.

        Args:
            persona_key: Persona identifier
            persona_embedding: Persona embedding vector
            journey_embedding: Journey embedding vector

        Returns:
            Criterion 2 score and analysis
        """
        try:
            # Compute cosine similarity
            similarity = self._compute_cosine_similarity(
                persona_embedding,
                journey_embedding
            )
            
            # Prepare result
            result = {
                'persona_key': persona_key,
                'score': similarity,
                'weighted_score': similarity * self.criterion_weights['persona_journey_consistency'],
                'method': 'persona_journey_embedding_similarity'
            }
            
            return result
            
        except Exception as e:
            return {
                'persona_key': persona_key,
                'score': 0.0,
                'weighted_score': 0.0,
                'error': str(e)
            }

    def save_evaluation_results(
        self,
        results: Dict[str, Any],
        filename_prefix: str = "realism_evaluation"
    ) -> str:
        """
        Save evaluation results to JSON file in evaluation directory.

        Args:
            results: Evaluation results dictionary
            filename_prefix: Prefix for output filename

        Returns:
            Path to saved file
        """
        try:
            # Ensure evaluation directory exists
            eval_dir = "/Users/victor_official/AI personas/evaluation"
            os.makedirs(eval_dir, exist_ok=True)

            # Generate timestamp-based filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.json"
            filepath = os.path.join(eval_dir, filename)

            # Save results
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"✅ Evaluation results saved to: {filepath}")
            return filepath

        except Exception as e:
            print(f"❌ Failed to save evaluation results: {str(e)}")
            return ""

    def _compute_criterion3_score(
        self,
        persona_key: str,
        persona_embedding: np.ndarray,
        journey_embedding: np.ndarray,
        cluster_centroids: Dict[int, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compute Criterion 3: Persona-Journey to Cluster Centroids Similarity.

        Args:
            persona_key: Persona identifier
            persona_embedding: Persona embedding vector
            journey_embedding: Journey embedding vector
            cluster_centroids: Dictionary of cluster centroids

        Returns:
            Criterion 3 score and analysis
        """
        try:
            # Step 1: Compute average of persona and journey embeddings
            avg_embedding = (persona_embedding + journey_embedding) / 2.0

            # Step 2: Compute similarity to each cluster centroid
            cluster_similarities = {}
            for cluster_id, centroid in cluster_centroids.items():
                similarity = self._compute_cosine_similarity(avg_embedding, centroid)
                cluster_similarities[cluster_id] = float(similarity)

            # Step 3: Take maximum similarity as the score
            if cluster_similarities:
                best_cluster_id = max(cluster_similarities, key=cluster_similarities.get)
                max_similarity = cluster_similarities[best_cluster_id]
            else:
                best_cluster_id = -1
                max_similarity = 0.0

            # Prepare result
            result = {
                'persona_key': persona_key,
                'score': max_similarity,
                'weighted_score': max_similarity * self.criterion_weights['persona_journey_to_clusters'],
                'best_cluster_id': int(best_cluster_id),
                'all_cluster_similarities': cluster_similarities,
                'num_clusters_compared': len(cluster_centroids)
            }

            return result

        except Exception as e:
            return {
                'persona_key': persona_key,
                'score': 0.0,
                'weighted_score': 0.0,
                'error': str(e)
            }


def main():
    """
    Main function: demonstrates how to use RealismEvaluator for realism assessment.
    """
    print("=" * 80)
    print("🎯 Persona Realism Evaluation System")
    print("=" * 80)
    print()
    
    # Initialize evaluator
    print("📊 Initializing evaluator...")
    evaluator = RealismEvaluator()
    
    # Check if required files exist
    required_files = [
        evaluator.persona_embeddings_path,
        evaluator.review_embeddings_path,
        evaluator.cluster_results_path
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("\n❌ Missing required data files:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nPlease run the following scripts to generate required embedding files:")
        print("   1. python scripts/generate_persona_embeddings.py")
        print("   2. python scripts/generate_review_embeddings.py")
        print("   3. python scripts/cluster_review_embeddings.py")
        return
    
    print("✅ All required files found")
    print()
    
    # Find available journey embeddings files
    eval_dir = "/Users/victor_official/AI personas/evaluation"
    journey_files = [
        f for f in os.listdir(eval_dir) 
        if f.startswith("journey_embeddings_") and f.endswith(".json")
    ]
    
    if not journey_files:
        print("❌ Journey embeddings file not found")
        print("Please run tests to generate journey data, then run:")
        print("   python scripts/generate_journey_embeddings.py")
        return
    
    # Use the latest journey embeddings file
    latest_journey_file = sorted(journey_files)[-1]
    journey_path = os.path.join(eval_dir, latest_journey_file)
    print(f"📂 Using Journey Embeddings: {latest_journey_file}")
    print()
    
    # Load journey data
    try:
        with open(journey_path, 'r', encoding='utf-8') as f:
            journey_data = json.load(f)
        journeys = journey_data.get('journeys', [])
        print(f"✅ Successfully loaded {len(journeys)} journey records")
    except Exception as e:
        print(f"❌ Failed to load journey data: {str(e)}")
        return
    
    if not journeys:
        print("❌ No available journey data")
        return
    
    print()
    print("=" * 80)
    print("🔍 Starting Persona Realism Evaluation")
    print("=" * 80)
    print()
    
    # Evaluate each persona
    all_results = []
    
    for idx, journey in enumerate(journeys, 1):
        persona_name = journey.get('persona_name', 'Unknown')
        print(f"[{idx}/{len(journeys)}] Evaluating Persona: {persona_name}")
        print("-" * 60)
        
        # Build persona_data
        persona_data = {
            'name': persona_name,
            'key': journey.get('persona_key', persona_name.lower().replace(' ', '_'))
        }
        
        # Evaluate realism
        result = evaluator.evaluate_persona_realism(
            persona_data=persona_data,
            action_log=[],  # Can pass in actual action log
            journey_embeddings_path=journey_path
        )
        
        # Display results
        if 'error' not in result:
            print(f"   ✅ Overall Realism Score: {result['overall_realism_score']:.4f}")
            
            # Criterion 1
            c1 = result.get('criterion_1_journey_to_reviews', {})
            print(f"   📊 Criterion 1 (Journey-Review Similarity): {c1.get('score', 0):.4f} (weighted: {c1.get('weighted_score', 0):.4f})")
            
            # Criterion 2
            c2 = result.get('criterion_2_persona_journey_consistency', {})
            if c2:
                print(f"   📊 Criterion 2 (Persona-Journey Consistency): {c2.get('score', 0):.4f} (weighted: {c2.get('weighted_score', 0):.4f})")
            else:
                print(f"   ⚠️  Criterion 2: Insufficient data, cannot calculate")
            
            # Criterion 3
            c3 = result.get('criterion_3_persona_journey_to_clusters', {})
            print(f"   📊 Criterion 3 (Cluster Match): {c3.get('score', 0):.4f} (weighted: {c3.get('weighted_score', 0):.4f})")
            print(f"   🎯 Best Matching Cluster: Cluster {c3.get('best_cluster_id', -1)}")
        else:
            print(f"   ❌ Evaluation failed: {result['error']}")
        
        all_results.append(result)
        print()
    
    # Compare all personas
    if len(all_results) > 1:
        print("=" * 80)
        print("📊 Persona Realism Rankings")
        print("=" * 80)
        print()
        
        # Prepare data structure for comparison
        comparison_data = [
            {
                'persona': {
                    'name': r.get('persona_name', 'Unknown'),
                    'key': r.get('persona_key', 'unknown')
                }
            }
            for r in all_results if 'error' not in r
        ]
        
        if comparison_data:
            comparison = evaluator.compare_personas(
                all_persona_results=comparison_data,
                journey_embeddings_path=journey_path
            )
            
            if 'error' not in comparison:
                print(f"🏆 Most Realistic Persona: {comparison['most_realistic']}")
                print(f"📉 Least Realistic Persona: {comparison['least_realistic']}")
                print()
                print("Detailed Rankings:")
                print("-" * 80)
                
                for persona_name, scores in comparison['comparison_table'].items():
                    rank = scores['rank']
                    overall = scores['overall_score']
                    c1 = scores['criterion1_score']
                    c2 = scores['criterion2_score']
                    c3 = scores['criterion3_score']
                    
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
                    print(f"{medal} {persona_name}")
                    print(f"    Overall Score: {overall:.4f}")
                    print(f"    C1: {c1:.4f} | C2: {c2:.4f} | C3: {c3:.4f}")
                    print()
                
                # Save comparison results
                print("=" * 80)
                print("💾 Saving Evaluation Results")
                print("=" * 80)
                saved_path = evaluator.save_evaluation_results(
                    comparison,
                    filename_prefix="persona_comparison"
                )
                
                if saved_path:
                    print(f"✅ Comparison results saved")
                    print()
                    
                    # Display statistics
                    stats = comparison.get('statistics', {})
                    print("📈 Statistics:")
                    print(f"   Criterion 1 - Mean: {stats.get('mean_criterion1_score', 0):.4f}, "
                          f"Std Dev: {stats.get('std_criterion1_score', 0):.4f}")
                    print(f"   Criterion 2 - Mean: {stats.get('mean_criterion2_score', 0):.4f}, "
                          f"Std Dev: {stats.get('std_criterion2_score', 0):.4f}")
                    print(f"   Criterion 3 - Mean: {stats.get('mean_criterion3_score', 0):.4f}, "
                          f"Std Dev: {stats.get('std_criterion3_score', 0):.4f}")
            else:
                print(f"❌ Comparison failed: {comparison['error']}")
    
    print()
    print("=" * 80)
    print("✨ Evaluation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
