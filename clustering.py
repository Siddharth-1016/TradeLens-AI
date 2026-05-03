"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 4+5 — STOCK CLUSTERING ENGINE + INTERPRETATION        ║
║  KMeans, Hierarchical, auto-label → BUY / MAYBE BUY / NOT  ║
╚══════════════════════════════════════════════════════════════╝

DESIGN RATIONALE:
- We use UNSUPERVISED clustering (no labels needed) — pure data-driven
- KMeans: fast, deterministic with seed, good for spherical clusters
- Hierarchical: reveals tree structure of stock relationships
- Silhouette / Davies-Bouldin / Elbow → objective cluster quality scoring
- Cluster label mapping uses domain heuristics on centroid values
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics  import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings, os
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER QUALITY EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_k_range(
    X: np.ndarray,
    k_range: range = range(2, 9),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluates KMeans for k in [2..8].
    Metrics:
      - Inertia       (elbow method — lower is better)
      - Silhouette    (0→1, higher is better)
      - Davies-Bouldin (lower is better)

    The OPTIMAL k is chosen as the one maximising Silhouette
    while Davies-Bouldin is below the median.
    """
    results = []
    for k in k_range:
        km  = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X)
        results.append({
            "k":              k,
            "inertia":        km.inertia_,
            "silhouette":     silhouette_score(X, lbl),
            "davies_bouldin": davies_bouldin_score(X, lbl),
        })

    df = pd.DataFrame(results)

    # Auto-select k: best silhouette with low DB
    db_thresh = df["davies_bouldin"].median()
    good      = df[df["davies_bouldin"] <= db_thresh]
    if good.empty:
        good = df
    optimal_k = int(good.loc[good["silhouette"].idxmax(), "k"])

    print(f"\n[Clustering] Evaluation table:\n{df.to_string(index=False)}")
    print(f"[Clustering] Auto-selected optimal k = {optimal_k}")

    return df, optimal_k


# ══════════════════════════════════════════════════════════════════════════════
# KMEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans(
    X: np.ndarray,
    k: int = 3,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray]:
    """Fit KMeans, return (model, labels)."""
    km  = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    lbl = km.fit_predict(X)
    sil = silhouette_score(X, lbl)
    db  = davies_bouldin_score(X, lbl)
    print(f"[KMeans] k={k}  Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")
    return km, lbl


# ══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def run_hierarchical(
    X: np.ndarray,
    k: int = 3,
    linkage_method: str = "ward",
) -> tuple[AgglomerativeClustering, np.ndarray]:
    """
    Ward linkage minimises within-cluster variance — best for compact clusters.
    """
    hc  = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    lbl = hc.fit_predict(X)
    sil = silhouette_score(X, lbl)
    db  = davies_bouldin_score(X, lbl)
    print(f"[Hierarchical] k={k}  Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}")
    return hc, lbl


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER → INVESTMENT LABEL MAPPING
# ══════════════════════════════════════════════════════════════════════════════

# Key features that define quality of each cluster centroid
SCORING_FEATURES = [
    # Positive score contributors (higher = better)
    "roe", "roa", "momentum_3m", "momentum_6m",
    "rsi",                      # moderate momentum
    "price_vs_sma50",           # above mid-term trend
    "revenue_growth", "profit_margin",
    "pct_from_52w_low",         # close to high means strength
    # Negative score contributors (higher raw = worse)
    "debt_to_equity",           # high debt = risk
    "volatility_20d",           # high vol = uncertainty
    "pct_from_52w_high",        # very negative = far below 52w high
]

POSITIVE_FEATURES = [
    "roe", "roa", "momentum_3m", "momentum_6m",
    "revenue_growth", "profit_margin", "price_vs_sma50",
    "pct_from_52w_low",
]
NEGATIVE_FEATURES = ["debt_to_equity", "volatility_20d"]


def score_cluster(centroid: pd.Series) -> float:
    """
    Compute a scalar quality score for a cluster centroid.
    Higher → stronger fundamentals + momentum.
    """
    score = 0.0
    for f in POSITIVE_FEATURES:
        if f in centroid.index and not np.isnan(centroid[f]):
            score += centroid[f]

    for f in NEGATIVE_FEATURES:
        if f in centroid.index and not np.isnan(centroid[f]):
            score -= centroid[f]

    # RSI: penalise extremes (> 70 overbought, < 30 oversold)
    if "rsi" in centroid.index and not np.isnan(centroid["rsi"]):
        rsi = centroid["rsi"]
        if 45 <= rsi <= 65:
            score += 1.0
        elif rsi > 75 or rsi < 25:
            score -= 1.0

    # pct_from_52w_high: 0 means at 52w high (best); very negative is bad
    if "pct_from_52w_high" in centroid.index:
        score += centroid["pct_from_52w_high"] * (-1)  # closer to 0 → higher score

    return score


LABEL_MAP = {
    0: "🟢 BUY",          # best cluster
    1: "🟡 MAYBE BUY",    # average cluster
    2: "🔴 NOT BUY",      # worst cluster
}
LABEL_COLOR = {
    "🟢 BUY":      "#00C851",
    "🟡 MAYBE BUY": "#FFBB33",
    "🔴 NOT BUY":  "#FF4444",
}


def assign_investment_labels(
    df_clean:  pd.DataFrame,
    df_scaled: pd.DataFrame,
    labels:    np.ndarray,
    features:  list,
    km_model:  KMeans = None,
) -> pd.DataFrame:
    """
    Maps raw cluster IDs (0,1,2) → BUY / MAYBE BUY / NOT BUY
    using centroid scoring.

    Strategy:
    1. Compute mean of each cluster on UNSCALED features (interpretable)
    2. Score each centroid
    3. Rank clusters by score → assign BUY / MAYBE / NOT
    """
    df_result = df_clean.copy()
    df_result["cluster_raw"] = labels

    # Compute centroids on unscaled features for interpretability
    centroids = (
        df_result
        .groupby("cluster_raw")[features]
        .mean()
    )

    # Score each centroid
    scores = {c: score_cluster(centroids.loc[c]) for c in centroids.index}
    ranked = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

    rank_to_label = {
        ranked[0]: "🟢 BUY",
        ranked[1]: "🟡 MAYBE BUY",
        ranked[2]: "🔴 NOT BUY",
    }

    # Handle k ≠ 3 gracefully
    n_clusters = len(ranked)
    if n_clusters == 2:
        rank_to_label = {ranked[0]: "🟢 BUY", ranked[1]: "🔴 NOT BUY"}
    elif n_clusters > 3:
        # Map top→BUY, bottom→NOT BUY, rest→MAYBE BUY
        rank_to_label = {ranked[0]: "🟢 BUY", ranked[-1]: "🔴 NOT BUY"}
        for c in ranked[1:-1]:
            rank_to_label[c] = "🟡 MAYBE BUY"

    df_result["investment_label"] = df_result["cluster_raw"].map(rank_to_label)
    df_result["cluster_score"]    = df_result["cluster_raw"].map(scores)

    # ── Summary Table ─────────────────────────────────────────────────────────
    print("\n[ClusterLabels] Centroid Scores:")
    for c in ranked:
        label = rank_to_label[c]
        print(f"  Cluster {c} → {label}  (score={scores[c]:.4f})")

    print("\n[ClusterLabels] Label Distribution:")
    print(df_result["investment_label"].value_counts().to_string())

    return df_result, rank_to_label


# ══════════════════════════════════════════════════════════════════════════════
# PCA VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def pca_scatter(
    df_scaled: pd.DataFrame,
    labels:    np.ndarray,
    label_map: dict,
    save_path: str = None,
) -> tuple[np.ndarray, PCA]:
    """
    Reduces scaled features to 2D via PCA and returns the array.
    Rendering is done in the Streamlit app via Plotly.
    """
    pca = PCA(n_components=2, random_state=42)
    X2  = pca.fit_transform(df_scaled.values)

    print(f"[PCA] Explained variance: PC1={pca.explained_variance_ratio_[0]:.2%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.2%}")
    return X2, pca


# ══════════════════════════════════════════════════════════════════════════════
# FULL CLUSTERING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering_pipeline(
    df_clean:  pd.DataFrame,
    df_scaled: pd.DataFrame,
    features:  list,
    k:         int = None,
) -> tuple[pd.DataFrame, KMeans, np.ndarray, PCA, np.ndarray]:
    """
    Master function:
    1. Auto-select k (or use provided)
    2. Run KMeans + Hierarchical, compare
    3. Assign investment labels
    4. PCA for visualisation

    Returns
    -------
    df_labelled : feature df with cluster + label columns
    km_model    : fitted KMeans
    labels      : raw cluster array
    pca         : fitted PCA
    X_pca       : 2D projection
    """
    X = df_scaled[features].values

    # ── 1. Find optimal k ─────────────────────────────────────────────────────
    if k is None:
        _, k = evaluate_k_range(X, k_range=range(2, 7))
        k    = max(k, 3)  # enforce minimum 3 for our 3-label system

    # ── 2. KMeans ─────────────────────────────────────────────────────────────
    km, km_labels = run_kmeans(X, k=k)

    # ── 3. Hierarchical (comparison) ─────────────────────────────────────────
    _, hc_labels  = run_hierarchical(X, k=k)

    # Agreement score between methods
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(km_labels, hc_labels)
    print(f"[Clustering] Adjusted Rand Index (KMeans vs Hierarchical): {ari:.4f}")

    # We proceed with KMeans labels (slightly more stable)
    # ── 4. Label assignment ───────────────────────────────────────────────────
    df_labelled, label_map = assign_investment_labels(
        df_clean, df_scaled, km_labels, features, km
    )

    # ── 5. PCA ────────────────────────────────────────────────────────────────
    X_pca, pca = pca_scatter(df_scaled[features], km_labels, label_map)

    return df_labelled, km, km_labels, pca, X_pca


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT SINGLE TICKER CLUSTER  (Streamlit app)
# ══════════════════════════════════════════════════════════════════════════════

def predict_single(
    ticker_features: pd.DataFrame,
    km_model:        KMeans,
    scaler,
    features:        list,
    label_map:       dict,
) -> str:
    """
    Given a new ticker's feature row, predict its investment label.
    """
    X = ticker_features[features].fillna(0).values
    X_scaled = scaler.transform(X)
    cluster  = int(km_model.predict(X_scaled)[0])
    label    = label_map.get(cluster, "🟡 MAYBE BUY")
    return label, cluster


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from feature_engineering import ALL_CLUSTER_FEATURES, clean_and_scale

    base = os.path.join(os.path.dirname(__file__), "..")
    path = os.path.join(base, "data/processed/stocks.csv")

    df_clean = pd.read_csv(path, index_col="ticker")
    df_clean_feats = df_clean[ALL_CLUSTER_FEATURES].copy()
    _, df_scaled, scaler = clean_and_scale(df_clean_feats)

    df_labelled, km, labels, pca, X_pca = run_clustering_pipeline(
        df_clean_feats, df_scaled, ALL_CLUSTER_FEATURES
    )
    print("\nSample results:")
    print(df_labelled[["investment_label", "cluster_score"]].head(10))
