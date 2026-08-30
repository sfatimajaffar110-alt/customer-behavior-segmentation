# src/clustering.py
"""
Clustering functions.
Covers: scaling, K-Means, Hierarchical, evaluation metrics,
        PCA visualization, cluster profiling with business names,
        saving and loading models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from scipy.cluster.hierarchy import linkage, fcluster

FEATURES = [
    'Days_Since_Last_Purchase',
    'Total_Transactions',
    'Total_Products_Purchased',
    'Total_Spend',
    'Average_Transaction_Value'
]


# ─────────────────────────────────────────────
# SCALING
# ─────────────────────────────────────────────

def scale_features(customer_data: pd.DataFrame,
                   models_dir: str = 'outputs/models') -> tuple:
    """
    Fit StandardScaler on RFM features.
    Returns (X_scaled_df, scaler) and saves scaler to disk.
    """
    X = customer_data[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, f'{models_dir}/scaler.joblib')
    print(f"Scaler saved to {models_dir}/scaler.joblib")
    return X_scaled_df, scaler


def load_scaler(models_dir: str = 'outputs/models') -> StandardScaler:
    """Load a previously saved scaler."""
    return joblib.load(f'{models_dir}/scaler.joblib')


# ─────────────────────────────────────────────
# K-MEANS
# ─────────────────────────────────────────────

def plot_optimal_k(X_scaled_df: pd.DataFrame, k_max: int = 10,
                   save_path: str = None) -> None:
    """Plot elbow curve and silhouette scores to find optimal K."""
    inertia, sil_scores = [], []
    K_range = range(2, k_max + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled_df)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled_df, km.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(K_range, inertia, marker='o', color='steelblue')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True)

    ax2.plot(K_range, sil_scores, marker='o', color='seagreen')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score per K')
    ax2.grid(True)

    plt.suptitle('Finding Optimal Number of Clusters', fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def train_kmeans(X_scaled_df: pd.DataFrame, n_clusters: int = 3,
                 models_dir: str = 'outputs/models') -> tuple:
    """
    Train K-Means and save model to disk.
    Returns (kmeans_model, labels).
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled_df)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(kmeans, f'{models_dir}/kmeans_model.joblib')
    print(f"K-Means model saved to {models_dir}/kmeans_model.joblib")
    return kmeans, labels


def load_kmeans(models_dir: str = 'outputs/models') -> KMeans:
    """Load a previously saved KMeans model."""
    return joblib.load(f'{models_dir}/kmeans_model.joblib')


def predict_cluster(customer_features: dict,
                    models_dir: str = 'outputs/models') -> int:
    """
    Predict cluster for a single new customer.
    customer_features: dict with keys matching FEATURES list.
    Returns cluster label as int.
    """
    scaler = load_scaler(models_dir)
    kmeans = load_kmeans(models_dir)
    X = pd.DataFrame([customer_features])[FEATURES]
    X_scaled = scaler.transform(X)
    return int(kmeans.predict(X_scaled)[0])


# ─────────────────────────────────────────────
# HIERARCHICAL CLUSTERING
# ─────────────────────────────────────────────

def train_hierarchical(X_scaled_df: pd.DataFrame,
                       n_clusters: int = 3) -> tuple:
    """
    Fit Hierarchical Clustering using Ward linkage.
    Returns (linked_matrix, labels).
    """
    linked = linkage(X_scaled_df, method='ward')
    labels = fcluster(linked, n_clusters, criterion='maxclust')
    print(f"Hierarchical clustering complete: {n_clusters} clusters")
    return linked, labels


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_clustering(X_scaled_df: pd.DataFrame,
                        kmeans_labels, hierarchical_labels) -> pd.DataFrame:
    """
    Print and return a comparison table of clustering metrics
    for both K-Means and Hierarchical.
    """
    metrics = {
        'Silhouette (higher=better)': [
            silhouette_score(X_scaled_df, kmeans_labels),
            silhouette_score(X_scaled_df, hierarchical_labels)
        ],
        'Davies-Bouldin (lower=better)': [
            davies_bouldin_score(X_scaled_df, kmeans_labels),
            davies_bouldin_score(X_scaled_df, hierarchical_labels)
        ],
        'Calinski-Harabasz (higher=better)': [
            calinski_harabasz_score(X_scaled_df, kmeans_labels),
            calinski_harabasz_score(X_scaled_df, hierarchical_labels)
        ]
    }
    results = pd.DataFrame(metrics, index=['K-Means', 'Hierarchical']).T.round(3)

    print("\n" + "=" * 55)
    print("  CLUSTERING EVALUATION METRICS")
    print("=" * 55)
    print(results.to_string())
    print("=" * 55)
    return results


# ─────────────────────────────────────────────
# PCA VISUALIZATION
# ─────────────────────────────────────────────

def plot_pca_clusters(X_scaled_df: pd.DataFrame,
                      kmeans_labels, hierarchical_labels,
                      save_path: str = None) -> None:
    """
    Reduce to 2D via PCA and plot both clustering results side by side.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled_df)
    explained = pca.explained_variance_ratio_ * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=kmeans_labels, cmap='Set1', alpha=0.6, s=30)
    ax1.set_title('K-Means Clusters (PCA)')
    ax1.set_xlabel(f'PC1 ({explained[0]:.1f}% variance)')
    ax1.set_ylabel(f'PC2 ({explained[1]:.1f}% variance)')
    plt.colorbar(sc1, ax=ax1, label='Cluster')

    sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=hierarchical_labels, cmap='Set2', alpha=0.6, s=30)
    ax2.set_title('Hierarchical Clusters (PCA)')
    ax2.set_xlabel(f'PC1 ({explained[0]:.1f}% variance)')
    ax2.set_ylabel(f'PC2 ({explained[1]:.1f}% variance)')
    plt.colorbar(sc2, ax=ax2, label='Cluster')

    plt.suptitle(
        f'Cluster Visualization via PCA '
        f'({explained[0] + explained[1]:.1f}% variance explained)',
        fontweight='bold'
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


# ─────────────────────────────────────────────
# CLUSTER PROFILING
# ─────────────────────────────────────────────

def profile_clusters(customer_data: pd.DataFrame,
                     kmeans_labels,
                     save_path: str = None) -> tuple:
    """
    Compute mean RFM values per cluster and auto-assign
    business segment names based on spend and recency rank.
    Returns (profile_df, segment_names dict, customer_data with labels).
    """
    customer_data = customer_data.copy()
    customer_data['KMeans_Cluster'] = kmeans_labels

    profile = customer_data.groupby('KMeans_Cluster')[FEATURES].mean().round(2)

    spend_rank = profile['Total_Spend'].rank()
    recency_rank = profile['Days_Since_Last_Purchase'].rank()

    segment_names = {}
    for cluster in profile.index:
        if spend_rank[cluster] == spend_rank.max():
            segment_names[cluster] = 'Champions'
        elif recency_rank[cluster] == recency_rank.min():
            segment_names[cluster] = 'At-Risk Customers'
        else:
            segment_names[cluster] = 'Occasional Buyers'

    customer_data['Segment_Name'] = customer_data['KMeans_Cluster'].map(segment_names)

    print("\nSegment Mapping:")
    for k, v in segment_names.items():
        count = (customer_data['KMeans_Cluster'] == k).sum()
        print(f"  Cluster {k} → {v} ({count} customers)")

    # Normalized profile bar chart
    profile_named = profile.copy()
    profile_named.index = [segment_names[i] for i in profile_named.index]
    profile_norm = (
        (profile_named - profile_named.min()) /
        (profile_named.max() - profile_named.min())
    )

    profile_norm.T.plot(kind='bar', figsize=(12, 6), colormap='Set1', edgecolor='white')
    plt.title('Cluster Profiles — Normalized Feature Averages', fontweight='bold', fontsize=14)
    plt.ylabel('Normalized Mean Value (0–1)')
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

    return profile, segment_names, customer_data