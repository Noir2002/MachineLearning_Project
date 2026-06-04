from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loader import ensure_output_dirs
from .train_models import NON_FEATURE_COLUMNS


def _load_features() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if not config.TRAIN_FEATURES_CSV.exists():
        raise FileNotFoundError("Run python -m src.build_features --split train before visualization.")
    df = pd.read_csv(config.TRAIN_FEATURES_CSV)
    feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return df, preprocessing.fit_transform(X), feature_columns


def _save_countplot(df: pd.DataFrame, x: str, path, title: str, *, hue: str | None = None) -> str:
    fig_width = 11 if x == config.SPECIES_COLUMN else 8
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    order = df[x].value_counts().index
    sns.countplot(data=df, y=x, order=order, hue=hue, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel(x)
    if hue:
        ax.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _projection_plot(points: np.ndarray, labels: pd.Series, path, title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], config.TARGET_COLUMN: labels.values})
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue=config.TARGET_COLUMN,
        s=55,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title=config.TARGET_COLUMN, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def safe_tsne_perplexity(n_samples: int) -> int:
    if n_samples <= 3:
        return 1
    return int(max(2, min(30, (n_samples - 1) // 3)))


def generate_visualizations() -> dict:
    ensure_output_dirs()
    df, X, _ = _load_features()
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_rows": int(len(df)),
        "target_for_projection_colors": config.TARGET_COLUMN,
        "figures": {},
    }

    audit["figures"]["bug_type_distribution"] = _save_countplot(
        df,
        config.TARGET_COLUMN,
        config.FIGURES_DIR / "bug_type_distribution.png",
        "Bug type distribution",
    )
    audit["figures"]["species_distribution"] = _save_countplot(
        df,
        config.SPECIES_COLUMN,
        config.FIGURES_DIR / "species_distribution.png",
        "Species distribution",
    )
    audit["figures"]["species_by_bug_type"] = _save_countplot(
        df,
        config.SPECIES_COLUMN,
        config.FIGURES_DIR / "species_by_bug_type.png",
        "Species grouped by bug type",
        hue=config.TARGET_COLUMN,
    )

    pca_points = PCA(n_components=2, random_state=config.RANDOM_STATE).fit_transform(X)
    audit["figures"]["pca_2d"] = _projection_plot(
        pca_points,
        df[config.TARGET_COLUMN],
        config.FIGURES_DIR / "pca_2d.png",
        "PCA projection of handcrafted features",
    )

    perplexity = safe_tsne_perplexity(len(df))
    tsne_points = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=config.RANDOM_STATE,
    ).fit_transform(X)
    audit["tsne_perplexity"] = perplexity
    audit["figures"]["tsne_2d"] = _projection_plot(
        tsne_points,
        df[config.TARGET_COLUMN],
        config.FIGURES_DIR / "tsne_2d.png",
        "t-SNE projection of handcrafted features",
    )

    nonlinear_method = "isomap"
    nonlinear_path = config.FIGURES_DIR / "isomap_2d.png"
    try:
        n_neighbors = max(2, min(10, len(df) - 1))
        nonlinear_points = Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(X)
        audit["isomap_n_neighbors"] = n_neighbors
    except Exception as exc:
        nonlinear_method = "mds"
        nonlinear_path = config.FIGURES_DIR / "mds_2d.png"
        nonlinear_points = MDS(
            n_components=2,
            random_state=config.RANDOM_STATE,
            normalized_stress="auto",
            n_init=4,
        ).fit_transform(X)
        audit["isomap_error"] = str(exc)

    audit["nonlinear_projection_method"] = nonlinear_method
    audit["figures"][f"{nonlinear_method}_2d"] = _projection_plot(
        nonlinear_points,
        df[config.TARGET_COLUMN],
        nonlinear_path,
        f"{nonlinear_method.upper()} projection of handcrafted features",
    )

    config.VISUALIZATION_AUDIT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Wrote visualization audit to {config.VISUALIZATION_AUDIT_JSON}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training feature visualizations.")
    parser.parse_args()
    generate_visualizations()


if __name__ == "__main__":
    main()

