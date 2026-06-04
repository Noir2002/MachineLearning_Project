from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import ensure_output_dirs
from .train_models import NON_FEATURE_COLUMNS


def _load_preprocessed() -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    if not config.TRAIN_FEATURES_CSV.exists():
        raise FileNotFoundError("Run python -m src.build_features --split train before clustering.")
    df = pd.read_csv(config.TRAIN_FEATURES_CSV)
    feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return df, preprocessing.fit_transform(X), df[config.TARGET_COLUMN].astype(str)


def _cluster_metrics(name: str, labels_true: pd.Series, labels_pred: np.ndarray, X: np.ndarray) -> dict:
    cluster_labels = np.asarray(labels_pred)
    unique_clusters = set(cluster_labels.tolist())
    non_noise_clusters = unique_clusters - {-1}
    if len(non_noise_clusters) >= 2:
        silhouette = silhouette_score(X, cluster_labels)
    else:
        silhouette = np.nan
    return {
        "method": name,
        "cluster_count": int(len(non_noise_clusters)),
        "noise_count": int(np.sum(cluster_labels == -1)),
        "silhouette": silhouette,
        "adjusted_rand_index": adjusted_rand_score(labels_true, cluster_labels),
        "normalized_mutual_information": normalized_mutual_info_score(labels_true, cluster_labels),
    }


def _plot_clusters(points: np.ndarray, cluster_labels: np.ndarray, path, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "cluster": cluster_labels.astype(str)})
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="cluster",
        palette="tab10",
        s=55,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def run_clustering() -> pd.DataFrame:
    ensure_output_dirs()
    df, X, y = _load_preprocessed()
    n_clusters = int(y.nunique())
    pca_points = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X)

    methods = {
        "kmeans": KMeans(n_clusters=n_clusters, n_init=30, random_state=config.RANDOM_STATE),
        "agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
        "dbscan": DBSCAN(eps=1.8, min_samples=5),
    }

    rows = []
    figure_map = {}
    for name, estimator in methods.items():
        print(f"Running clustering: {name}")
        labels = estimator.fit_predict(X)
        rows.append(_cluster_metrics(name, y, labels, X))
        if name in {"kmeans", "agglomerative"}:
            figure_map[name] = _plot_clusters(
                pca_points,
                labels,
                config.FIGURES_DIR / f"clustering_{name}.png",
                f"{name.title()} clusters on PCA projection",
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(config.CLUSTERING_METRICS_CSV, index=False)
    audit_path = config.RESULTS_DIR / "clustering_audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "input_rows": int(len(df)),
                "target_used_for_external_metrics": config.TARGET_COLUMN,
                "species_used_as_target": False,
                "figures": figure_map,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustering on handcrafted features.")
    parser.parse_args()
    run_clustering()


if __name__ == "__main__":
    main()

