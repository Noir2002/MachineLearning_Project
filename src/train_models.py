from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from . import config
from .data_loader import ensure_output_dirs, load_labels

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")


NON_FEATURE_COLUMNS = {config.ID_COLUMN, config.TARGET_COLUMN, config.SPECIES_COLUMN}


def load_training_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if not config.TRAIN_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Training features not found: {config.TRAIN_FEATURES_CSV}. "
            "Run python -m src.build_features --split train first."
        )
    df = pd.read_csv(config.TRAIN_FEATURES_CSV)
    if len(df) != 249:
        raise ValueError(f"Expected 249 training rows, got {len(df)}")
    if 154 in set(df[config.ID_COLUMN].astype(int)):
        raise ValueError("ID 154 must not appear in training features.")
    feature_columns = [col for col in df.columns if col not in NON_FEATURE_COLUMNS]
    X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = df[config.TARGET_COLUMN].astype(str)
    return df, X, y


def _scaled_pipeline(classifier, scaler="standard") -> Pipeline:
    scaler_step = StandardScaler() if scaler == "standard" else RobustScaler()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_step),
            ("classifier", classifier),
        ]
    )


def _tree_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def model_registry() -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": _tree_pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic_regression": _scaled_pipeline(
            LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
        ),
        "svm_rbf": _scaled_pipeline(
            SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
        ),
        "knn": _scaled_pipeline(KNeighborsClassifier(n_neighbors=5), scaler="standard"),
        "random_forest": _tree_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                random_state=config.RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
        "extra_trees": _tree_pipeline(
            ExtraTreesClassifier(
                n_estimators=400,
                random_state=config.RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )
        ),
        "gradient_boosting": _tree_pipeline(
            GradientBoostingClassifier(random_state=config.RANDOM_STATE)
        ),
        "mlp": _scaled_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=600,
                early_stopping=False,
                random_state=config.RANDOM_STATE,
            )
        ),
    }


def frequent_class_subset(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict]:
    counts = y.value_counts()
    frequent_classes = sorted(counts[counts >= 2].index.tolist())
    mask = y.isin(frequent_classes)
    X_cv = X.loc[mask].reset_index(drop=True)
    y_cv = y.loc[mask].reset_index(drop=True)
    folds = min(5, int(y_cv.value_counts().min()))
    if folds < 2:
        raise ValueError("Not enough frequent classes for stratified cross-validation.")
    info = {
        "frequent_classes": frequent_classes,
        "excluded_from_cv_classes": sorted(counts[counts < 2].index.tolist()),
        "excluded_from_cv_sample_count": int((~mask).sum()),
        "cv_folds": folds,
        "class_counts_all": counts.sort_index().to_dict(),
        "class_counts_cv": y_cv.value_counts().sort_index().to_dict(),
    }
    return X_cv, y_cv, info


def metric_row(model_name: str, y_true: pd.Series, y_pred: np.ndarray, extra: dict | None = None) -> dict:
    row = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if extra:
        row.update(extra)
    return row


def save_confusion_matrix(model_name: str, y_true: pd.Series, y_pred: np.ndarray) -> str:
    labels = sorted(y_true.unique())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion matrix: {model_name}")
    ax.set_xlabel("Predicted bug type")
    ax.set_ylabel("True bug type")
    fig.tight_layout()
    path = config.FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _feature_importance(final_model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    classifier = final_model.named_steps["classifier"]
    feature_names = list(X.columns)
    if hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_)
        method = "model_feature_importances"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result = permutation_importance(
                final_model,
                X,
                y,
                scoring="f1_macro",
                n_repeats=5,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        importances = result.importances_mean
        method = "permutation_importance_training_descriptive"

    order = np.argsort(importances)[::-1]
    top = order[:20]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(x=importances[top], y=[feature_names[i] for i in top], ax=ax, color="#4c78a8")
    ax.set_title("Top feature importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    path = config.FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "method": method,
        "figure": str(path),
        "top_features": [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top
        ],
    }


def train_models() -> dict:
    ensure_output_dirs()
    df, X, y = load_training_features()
    X_cv, y_cv, cv_info = frequent_class_subset(X, y)
    cv = StratifiedKFold(n_splits=cv_info["cv_folds"], shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    reports: dict = {
        "target": config.TARGET_COLUMN,
        "species_used_as_target": False,
        "cv_policy": cv_info,
        "cv_reports": {},
        "full_data_descriptive_reports": {},
    }
    confusion_figures: dict[str, str] = {}

    for model_name, pipeline in model_registry().items():
        print(f"Evaluating {model_name}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                y_pred = cross_val_predict(pipeline, X_cv, y_cv, cv=cv, n_jobs=None)
            rows.append(metric_row(model_name, y_cv, y_pred, {"status": "ok", "error": ""}))
            reports["cv_reports"][model_name] = classification_report(
                y_cv, y_pred, output_dict=True, zero_division=0
            )
            confusion_figures[model_name] = save_confusion_matrix(model_name, y_cv, y_pred)
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "accuracy": np.nan,
                    "macro_precision": np.nan,
                    "macro_recall": np.nan,
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            reports["cv_reports"][model_name] = {"error": str(exc)}

    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    successful = comparison[comparison["status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("All model evaluations failed.")
    successful = successful.sort_values(["macro_f1", "weighted_f1", "accuracy"], ascending=False)
    best_model_name = str(successful.iloc[0]["model"])
    best_pipeline = clone(model_registry()[best_model_name])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        best_pipeline.fit(X, y)
    train_pred = best_pipeline.predict(X)
    full_training_metrics = metric_row(best_model_name, y, train_pred)
    reports["full_data_descriptive_reports"][best_model_name] = classification_report(
        y, train_pred, output_dict=True, zero_division=0
    )

    feature_importance = _feature_importance(best_pipeline, X, y)
    joblib.dump(best_pipeline, config.BEST_MODEL_PATH)

    known_labels = sorted(load_labels()[config.TARGET_COLUMN].astype(str).unique().tolist())
    info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_model": best_model_name,
        "selection_metric": "frequent_class_cv_macro_f1",
        "target": config.TARGET_COLUMN,
        "species_used_as_target": False,
        "train_feature_rows": int(len(df)),
        "excluded_ids": sorted(config.EXCLUDED_TRAIN_IDS),
        "cv_policy": cv_info,
        "best_validation_metrics": successful.iloc[0].to_dict(),
        "full_data_descriptive_training_metrics": full_training_metrics,
        "model_path": str(config.BEST_MODEL_PATH),
        "known_bug_type_labels": known_labels,
        "confusion_matrix_figures": confusion_figures,
        "feature_importance": feature_importance,
    }
    config.BEST_MODEL_INFO_JSON.write_text(json.dumps(info, indent=2), encoding="utf-8")
    config.CLASSIFICATION_REPORTS_JSON.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Best model: {best_model_name}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Train supervised feature-based models.")
    parser.parse_args()
    train_models()


if __name__ == "__main__":
    main()
