"""
HDBSCAN clustering analysis of review embeddings
Read embeddings file and perform topic clustering
"""

import json
import numpy as np
import pandas as pd
import hdbscan
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_embeddings(json_path):
    """Load embeddings data"""
    print(f"Loading embeddings: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Metadata: {data['metadata']}")
    return data


def prepare_data(data, score_weight=0.3):
    """Prepare clustering data, combining embedding and scores
    
    Args:
        data: embeddings data
        score_weight: score weight (0-1), default 0.3 means scores account for 30%
    """
    embeddings_list = []
    reviews_list = []
    scores_list = []
    
    print("\nPreparing data...")
    for item in data['embeddings']:
        if item['embedding'] is not None:
            embeddings_list.append(item['embedding'])
            scores_list.append(item['score'])
            reviews_list.append({
                'id': item['id'],
                'title': item['title'],
                'content': item['content'],
                'score': item['score'],
                'sentiment': item['sentiment'],
                'verified_buyer': item['metadata']['verified_buyer'],
                'size': item['metadata']['size'],
                'age': item['metadata']['age'],
            })
    
    embeddings_array = np.array(embeddings_list)
    scores_array = np.array(scores_list).reshape(-1, 1)
    reviews_df = pd.DataFrame(reviews_list)
    
    # Normalize scores to [0,1] range
    scores_normalized = (scores_array - 1) / 4  # Map 1-5 to 0-1
    
    # Expand score features to match embedding dimension influence
    # Adjust scale by multiplying by square root of embedding dimension
    embedding_dim = embeddings_array.shape[1]
    score_feature_scaled = scores_normalized * np.sqrt(embedding_dim)
    
    # Merge embedding and score features
    # embedding has (1-score_weight) weight, score has score_weight weight
    embeddings_weighted = embeddings_array * (1 - score_weight)
    scores_weighted = score_feature_scaled * score_weight
    
    combined_features = np.hstack([embeddings_weighted, scores_weighted])
    
    print(f"Valid data: {len(reviews_df)} items")
    print(f"Original embedding dimension: {embeddings_array.shape[1]}")
    print(f"Combined dimension: {combined_features.shape[1]}")
    print(f"Score weight: {score_weight*100:.0f}%, Embedding weight: {(1-score_weight)*100:.0f}%")
    print(f"Score distribution: min={scores_array.min()}, max={scores_array.max()}, mean={scores_array.mean():.2f}")
    
    return combined_features, reviews_df


def reduce_dimensions(embeddings, n_components=5, n_neighbors=15):
    """Reduce dimensions using UMAP"""
    print(f"\nReducing dimensions to {n_components}D using UMAP...")
    print(f"Parameters: n_neighbors={n_neighbors}")
    
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        verbose=True
    )
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"Reduced shape: {reduced_embeddings.shape}")
    
    return reduced_embeddings


def perform_clustering(reduced_embeddings, min_cluster_size=50, min_samples=10):
    """Perform HDBSCAN clustering"""
    print(f"\nPerforming HDBSCAN clustering...")
    print(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )
    
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    # Statistics
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\nClustering results:")
    print(f"  - Found {n_clusters} clusters")
    print(f"  - Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
    
    return cluster_labels, clusterer


def analyze_clusters(reviews_df, cluster_labels):
    """Analyze clustering results"""
    reviews_df['cluster'] = cluster_labels
    
    print("\n" + "="*80)
    print("Cluster Analysis")
    print("="*80)
    
    # Statistics for each cluster
    cluster_counts = Counter(cluster_labels)
    
    for cluster_id in sorted([c for c in cluster_counts.keys() if c != -1]):
        cluster_reviews = reviews_df[reviews_df['cluster'] == cluster_id]
        
        print(f"\n【Cluster #{cluster_id}】 - {len(cluster_reviews)} reviews")
        print(f"Average score: {cluster_reviews['score'].mean():.2f}")
        print(f"Average sentiment: {cluster_reviews['sentiment'].mean():.3f}")
        print(f"Verified buyer ratio: {cluster_reviews['verified_buyer'].sum()/len(cluster_reviews)*100:.1f}%")
        
        # Extract keywords (using TF-IDF)
        texts = (cluster_reviews['title'] + '. ' + cluster_reviews['content']).tolist()
        
        if len(texts) >= 5:  # Analyze only if at least 5 reviews
            try:
                vectorizer = TfidfVectorizer(
                    max_features=10,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                
                print(f"Keywords: {', '.join(feature_names[:8])}")
            except:
                pass
        
        # Display sample reviews
        print("\nSample reviews:")
        for idx, (_, review) in enumerate(cluster_reviews.head(3).iterrows()):
            content_preview = review['content'][:100] + "..." if len(review['content']) > 100 else review['content']
            print(f"  {idx+1}. [{review['score']}⭐] {content_preview}")
    
    # Noise points statistics
    if -1 in cluster_counts:
        noise_reviews = reviews_df[reviews_df['cluster'] == -1]
        print(f"\n【Noise Points】 - {len(noise_reviews)} reviews")
        print(f"Average score: {noise_reviews['score'].mean():.2f}")
    
    return reviews_df


def visualize_clusters(reduced_embeddings, cluster_labels, reviews_df, output_dir):
    """Visualize clustering results"""
    print("\nGenerating visualization charts...")
    
    # If dimension > 2, reduce to 2D for visualization
    if reduced_embeddings.shape[1] > 2:
        print("Reducing to 2D for visualization...")
        vis_embeddings = UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        ).fit_transform(reduced_embeddings)
    else:
        vis_embeddings = reduced_embeddings
    
    # Create charts
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Cluster distribution plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(
        vis_embeddings[:, 0],
        vis_embeddings[:, 1],
        c=cluster_labels,
        cmap='Spectral',
        s=10,
        alpha=0.6
    )
    ax1.set_title('HDBSCAN Clustering Results', fontsize=14, fontweight='bold')
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 2. Colored by rating
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(
        vis_embeddings[:, 0],
        vis_embeddings[:, 1],
        c=reviews_df['score'],
        cmap='RdYlGn',
        s=10,
        alpha=0.6,
        vmin=1,
        vmax=5
    )
    ax2.set_title('Reviews colored by Rating', fontsize=14, fontweight='bold')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    plt.colorbar(scatter2, ax=ax2, label='Rating (1-5)')
    
    # 3. Cluster size distribution
    ax3 = axes[1, 0]
    cluster_counts = Counter([c for c in cluster_labels if c != -1])
    if cluster_counts:
        clusters = sorted(cluster_counts.keys())
        counts = [cluster_counts[c] for c in clusters]
        ax3.bar(clusters, counts, color='steelblue', alpha=0.7)
        ax3.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Reviews')
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Average rating by cluster
    ax4 = axes[1, 1]
    cluster_stats = reviews_df.groupby('cluster')['score'].mean().sort_index()
    cluster_stats = cluster_stats[cluster_stats.index != -1]
    if len(cluster_stats) > 0:
        colors = ['green' if score >= 4 else 'orange' if score >= 3 else 'red' 
                  for score in cluster_stats.values]
        ax4.bar(cluster_stats.index, cluster_stats.values, color=colors, alpha=0.7)
        ax4.axhline(y=reviews_df['score'].mean(), color='red', linestyle='--', 
                    label=f'Overall avg: {reviews_df["score"].mean():.2f}')
        ax4.set_title('Average Rating by Cluster', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Average Rating')
        ax4.set_ylim(0, 5)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{output_dir}/cluster_visualization_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved: {plot_path}")
    
    plt.close()


def save_results(reviews_df, cluster_labels, output_dir):
    """Save clustering results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/cluster_results_{timestamp}.json"
    
    # Prepare output data
    output_data = {
        "metadata": {
            "total_reviews": len(reviews_df),
            "n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            "n_noise": list(cluster_labels).count(-1),
            "clustered_at": datetime.now().isoformat(),
        },
        "cluster_summary": [],
        "reviews": []
    }
    
    # Cluster summary
    for cluster_id in sorted([c for c in set(cluster_labels) if c != -1]):
        cluster_reviews = reviews_df[reviews_df['cluster'] == cluster_id]
        output_data["cluster_summary"].append({
            "cluster_id": int(cluster_id),
            "size": len(cluster_reviews),
            "avg_score": float(cluster_reviews['score'].mean()),
            "avg_sentiment": float(cluster_reviews['sentiment'].mean()),
        })
    
    # Review details
    for _, review in reviews_df.iterrows():
        output_data["reviews"].append({
            "id": int(review['id']),
            "cluster": int(review['cluster']),
            "title": review['title'],
            "content": review['content'],
            "score": int(review['score']),
            "sentiment": float(review['sentiment']),
        })
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved: {output_path}")


def main():
    # Configuration parameters
    EMBEDDINGS_FILE = "/Users/victor_official/AI personas/evaluation/review_embeddings_20251115_181818.json"
    OUTPUT_DIR = "/Users/victor_official/AI personas/evaluation"
    
    # Feature weight parameters
    SCORE_WEIGHT = 0.3  # Score weight 30%, embedding weight 70%
    
    # UMAP parameters
    UMAP_N_COMPONENTS = 5
    UMAP_N_NEIGHBORS = 15
    
    # HDBSCAN parameters
    MIN_CLUSTER_SIZE = 40  # Minimum cluster size
    MIN_SAMPLES = 10       # Minimum neighbors for core points
    
    print("="*80)
    print("Review HDBSCAN Clustering Analysis (Combined with Scores)")
    print("="*80)
    
    # 1. Load data
    data = load_embeddings(EMBEDDINGS_FILE)
    embeddings, reviews_df = prepare_data(data, score_weight=SCORE_WEIGHT)
    
    # 2. UMAP dimensionality reduction
    reduced_embeddings = reduce_dimensions(
        embeddings,
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS
    )
    
    # 3. HDBSCAN clustering
    cluster_labels, clusterer = perform_clustering(
        reduced_embeddings,
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES
    )
    
    # 4. Analyze results
    reviews_df = analyze_clusters(reviews_df, cluster_labels)
    
    # 5. Visualize
    visualize_clusters(reduced_embeddings, cluster_labels, reviews_df, OUTPUT_DIR)
    
    # 6. Save results
    save_results(reviews_df, cluster_labels, OUTPUT_DIR)
    
    print("\n" + "="*80)
    print("Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

