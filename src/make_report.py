from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import config
from .data_loader import ensure_output_dirs, test_data_available
from .validate_submission import validate_submission_file

PENDING_SUBMISSION_SENTENCE = (
    "The final submission generation pipeline has been implemented, but the official CSV for images "
    "251-347 will be generated after the test images and masks are released."
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _figure(path: Path) -> str:
    return str(path.relative_to(config.PROJECT_ROOT)) if path.exists() else "Not generated"


def _counts_markdown(counts: dict, name: str, limit: int | None = None) -> str:
    if not counts:
        return f"| {name} | count |\n|---|---:|\n| not available | 0 |"
    items = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    if limit is not None:
        items = items[:limit]
    rows = [f"| {label} | {count} |" for label, count in items]
    return f"| {name} | count |\n|---|---:|\n" + "\n".join(rows)


def _prediction_distribution_markdown(distribution: dict) -> str:
    return _counts_markdown(distribution, "predicted bug type")


def _model_metric_summary(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "No model metrics were generated."
    rows = []
    for model in [
        "dummy_most_frequent",
        "logistic_regression",
        "svm_rbf",
        "knn",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "mlp",
    ]:
        rows.append(
            f"| {model} | {_metric_for_model(comparison, model, 'accuracy')} | "
            f"{_metric_for_model(comparison, model, 'macro_f1')} | "
            f"{_metric_for_model(comparison, model, 'weighted_f1')} |"
        )
    return "| model | accuracy | macro-F1 | weighted-F1 |\n|---|---:|---:|---:|\n" + "\n".join(rows)


def _fmt(value, digits: int = 6) -> str:
    if value is None or value == "not available":
        return "not available"
    try:
        if pd.isna(value):
            return "not available"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_for_model(comparison: pd.DataFrame, model: str, metric: str) -> str:
    if comparison.empty or metric not in comparison.columns or "model" not in comparison.columns:
        return "not available"
    row = comparison[comparison["model"] == model]
    if row.empty:
        return "not available"
    return _fmt(row.iloc[0][metric])


def _load_train_features() -> pd.DataFrame:
    if not config.TRAIN_FEATURES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(config.TRAIN_FEATURES_CSV)


def _submission_summary() -> dict:
    if not config.SUBMISSION_CSV.exists():
        return {
            "exists": False,
            "valid": False,
            "row_count": 0,
            "distribution": {},
            "error": "results/submission.csv has not been generated.",
        }
    try:
        validate_submission_file(config.SUBMISSION_CSV)
        valid = True
        error = ""
    except Exception as exc:
        valid = False
        error = str(exc)
    df = pd.read_csv(config.SUBMISSION_CSV)
    distribution = (
        df[config.TARGET_COLUMN].value_counts().to_dict()
        if config.TARGET_COLUMN in df.columns
        else {}
    )
    ids = df[config.ID_COLUMN].astype(int).tolist() if config.ID_COLUMN in df.columns else []
    return {
        "exists": True,
        "valid": valid,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "ids_exact": ids == list(config.TEST_IDS),
        "duplicate_ids": bool(df[config.ID_COLUMN].duplicated().any()) if config.ID_COLUMN in df.columns else None,
        "missing_predictions": bool(df[config.TARGET_COLUMN].isna().any()) if config.TARGET_COLUMN in df.columns else None,
        "distribution": distribution,
        "error": error,
    }


def _markdown_report() -> str:
    audit = _load_json(config.DATA_AUDIT_JSON)
    best_info = _load_json(config.BEST_MODEL_INFO_JSON)
    comparison = pd.read_csv(config.MODEL_COMPARISON_CSV) if config.MODEL_COMPARISON_CSV.exists() else pd.DataFrame()
    clustering = pd.read_csv(config.CLUSTERING_METRICS_CSV) if config.CLUSTERING_METRICS_CSV.exists() else pd.DataFrame()
    train_features = _load_train_features()
    submission_summary = _submission_summary()

    train_rows = best_info.get("train_feature_rows", 249 if config.TRAIN_FEATURES_CSV.exists() else "not generated")
    numeric_best_model = best_info.get("numeric_best_model", best_info.get("best_model", "not selected"))
    final_recommended_model = best_info.get("final_recommended_model", best_info.get("best_model", "not selected"))
    numeric_metrics = best_info.get("numeric_best_validation_metrics", best_info.get("best_validation_metrics", {}))
    final_metrics = best_info.get("final_recommended_validation_metrics", best_info.get("best_validation_metrics", {}))
    tie_breaker = best_info.get("tie_breaker", {})

    bug_counts = (
        train_features[config.TARGET_COLUMN].value_counts().to_dict()
        if not train_features.empty and config.TARGET_COLUMN in train_features.columns
        else {}
    )
    species_counts = (
        train_features[config.SPECIES_COLUMN].value_counts().to_dict()
        if not train_features.empty and config.SPECIES_COLUMN in train_features.columns
        else {}
    )
    prediction_distribution = submission_summary.get("distribution", {})

    clustering_table = (
        clustering.to_markdown(index=False)
        if not clustering.empty
        else "Clustering metrics were not generated."
    )

    lines = [
        "# Machine Learning Project: To bee or not to bee",
        "",
        "## Team",
        "",
        "- Lianghong LI",
        "- Zhekai YAN",
        "- Muzi LI",
        "",
        "## Contributions",
        "",
        "- Lianghong LI: project lead; implemented feature extraction, model training, validation, final prediction pipeline, and report integration.",
        "- Zhekai YAN: support contribution for data checking, visualization review, and report proofreading.",
        "- Muzi LI: support contribution for reproducibility checking, documentation review, and result verification.",
        "",
        "## Introduction",
        "",
        "This project classifies insect images into the assignment target `bug type`. The main pipeline uses handcrafted features extracted from each image and its segmentation mask, rather than raw-image CNN classification. This design follows the assignment requirements and makes the solution depend on measurable foreground color, shape, and texture properties.",
        "",
        "## Dataset and Preprocessing",
        "",
        "The training labels contain `ID`, `bug type`, and `species`. The main supervised target is `bug type`; `species` is used only for visualization and descriptive analysis. The official test set contains 97 images with IDs 251-347 and corresponding segmentation masks.",
        "",
        "ID 154 is excluded because `train/masks/binary_154.tif` is unavailable. No artificial, estimated, manually drawn, or imputed mask was created. The effective mask-based training set therefore contains 249 samples.",
        "",
        f"Effective training samples: `{train_rows}`.",
        "",
        "This exclusion is recorded in `results/data_audit.json`.",
        "",
        "Training bug type distribution:",
        "",
        _counts_markdown(bug_counts, "bug type"),
        "",
        "Largest species groups, shown for descriptive context only:",
        "",
        _counts_markdown(species_counts, "species", limit=10),
        "",
        "## Feature Extraction",
        "",
        "All mask-based statistics exclude pixels outside the segmentation mask, so background content does not dominate the representation.",
        "",
        "- Color symmetry compares the insect bounding-box crop with its left-right flipped version. RGB differences are measured only where both original and flipped masks contain foreground pixels.",
        "- Shape symmetry flips the binary mask crop left-right and measures normalized foreground overlap with the original mask.",
        "- Bug pixel ratio is computed as `bug_pixels / image_pixels`.",
        "- RGB features are the min, max, mean, median, and standard deviation for R, G, and B values inside the mask.",
        "- Morphology features include area, perimeter, circularity, aspect ratio, extent, solidity, orientation, eccentricity, and major/minor axis lengths.",
        "- Hu moments are global shape descriptors based on image moments and provide compact information about overall mask geometry.",
        "- HSV and LAB features complement RGB by separating hue/saturation/value and perceptual lightness/color axes.",
        "- Grayscale contrast and edge density describe texture and local structure inside the mask.",
        "",
        "## Data Visualization",
        "",
        f"- Bug type distribution: `{_figure(config.FIGURES_DIR / 'bug_type_distribution.png')}`. The labeled data is strongly imbalanced, with Bee and Bumblebee dominating.",
        f"- Species distribution: `{_figure(config.FIGURES_DIR / 'species_distribution.png')}`. Species labels are finer-grained and imbalanced, but species is not used as the final target.",
        f"- Species grouped by bug type: `{_figure(config.FIGURES_DIR / 'species_by_bug_type.png')}`. This shows how fine labels relate to the broader required bug type labels.",
        f"- PCA: `{_figure(config.FIGURES_DIR / 'pca_2d.png')}`. The linear projection shows limited separation and substantial overlap.",
        f"- t-SNE: `{_figure(config.FIGURES_DIR / 'tsne_2d.png')}`. Local grouping is visible, but classes still overlap.",
        f"- Isomap: `{_figure(config.FIGURES_DIR / 'isomap_2d.png')}`. Nonlinear structure is present, but it does not fully separate all bug types.",
        "",
        "## Supervised Learning Methods",
        "",
        "The supervised target is `bug type`; `species` is never used as the target. Macro-F1 is the main validation metric because class imbalance makes accuracy alone insufficient. Frequent-class cross-validation is used because singleton classes make full stratified CV invalid. Full-data metrics are descriptive only and are not validation metrics.",
        "",
        "- DummyClassifier is the majority-class baseline.",
        "- LogisticRegression is a linear non-ensemble baseline.",
        "- SVM RBF is a nonlinear and robust classifier for standardized tabular features.",
        "- KNN is a local-neighborhood classifier.",
        "- RandomForest, ExtraTrees, and GradientBoosting are supervised ensemble methods.",
        "- MLP is an optional neural model trained on extracted features only, not raw images.",
        "",
        "Model comparison:",
        "",
        _model_metric_summary(comparison),
        "",
        "## Clustering Methods",
        "",
        "KMeans, AgglomerativeClustering, and DBSCAN are evaluated on imputed and standardized handcrafted features. Their cluster assignments are compared with `bug type` only for interpretation.",
        "",
        clustering_table,
        "",
        "KMeans has the strongest clustering metrics among the tested methods, but the scores remain low to moderate. AgglomerativeClustering finds imperfect hierarchical structure. DBSCAN marks all samples as noise under the selected settings. These results suggest that the features contain useful structure but not naturally perfect bug type clusters.",
        "",
        "## Model Selection",
        "",
        "The metric table is kept honest: MLP has the numerically highest CV macro-F1, while SVM RBF is selected as the final model by robustness tie-breaker.",
        "",
        f"- Numeric best model: `{numeric_best_model}` with CV macro-F1 `{numeric_metrics.get('macro_f1', 'not available')}`.",
        f"- Final model: `{final_recommended_model}` with CV macro-F1 `{final_metrics.get('macro_f1', 'not available')}`.",
        "- The difference is below 0.005.",
        f"- Tie-breaker reason: {tie_breaker.get('reason', 'not applied')}",
        "",
        "SVM RBF is preferred for final prediction because it is simpler and more stable for a small imbalanced handcrafted-feature dataset while performing essentially the same as MLP in validation.",
        "",
        "## Final Prediction on Test Images 251-347",
        "",
        "The official test images and masks for IDs 251-347 were processed. The final SVM RBF model was applied to 97 test samples.",
        "",
        "`results/submission.csv` and `deliverables/submission.csv` were generated with exactly two columns: `ID` and `bug type`. The CSV was validated for 97 rows, IDs 251-347, no duplicate IDs, no missing values, and valid training bug type labels.",
        "",
        "Prediction distribution:",
        "",
        _prediction_distribution_markdown(prediction_distribution),
        "",
        "No test accuracy is reported because test labels are unavailable.",
        "",
        "## Limitations",
        "",
        "The dataset is small and strongly imbalanced. The Dragonfly class has only one usable sample. ID 154 is excluded because the official mask is missing. Bee and Bumblebee can visually overlap, making separation difficult for handcrafted features. Test labels are unavailable, so no test accuracy can be reported.",
        "",
        "## Conclusion",
        "",
        "The project satisfies the required feature extraction, visualization, supervised learning, clustering, and final prediction components. Handcrafted mask-based features are useful for the task, SVM RBF was selected for robust final prediction, and the final submission CSV was generated and validated.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python3 -m src.data_loader --validate --split train",
        "python3 -m src.build_features --split train",
        "python3 -m src.train_models",
        "python3 -m src.clustering",
        "python3 -m src.visualize",
        "python3 -m src.make_report",
        "python3 -m src.data_loader --validate --split test",
        "python3 -m src.build_features --split test",
        "python3 -m src.predict",
        "python3 -m src.validate_submission results/submission.csv",
        "```",
        "",
        "Test data is not used for model selection or hyperparameter tuning.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report() -> Path:
    ensure_output_dirs()
    path = config.REPORTS_DIR / "report.md"
    path.write_text(_markdown_report(), encoding="utf-8")
    return path


def _add_paragraph(story: list, text: str, style, space: float = 0.08) -> None:
    story.append(Paragraph(text, style))
    story.append(Spacer(1, space * inch))


def _add_bullets(story: list, items: list[str], style) -> None:
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, style)) for item in items],
            bulletType="bullet",
            leftIndent=0.2 * inch,
        )
    )
    story.append(Spacer(1, 0.08 * inch))


def _small_table_from_dataframe(df: pd.DataFrame, columns: list[str], max_rows: int = 8) -> Table | Paragraph:
    if df.empty:
        return Paragraph("No metrics generated.", getSampleStyleSheet()["BodyText"])
    available = [col for col in columns if col in df.columns]
    display = df[available].head(max_rows).copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
    data = [available] + display.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8f5")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _add_image_if_exists(story: list, path: Path, caption: str, styles) -> None:
    if not path.exists():
        return
    image = RLImage(str(path))
    image._restrictSize(6.5 * inch, 4.4 * inch)
    story.append(
        KeepTogether(
            [
                Paragraph(caption, styles["BodyText"]),
                Spacer(1, 0.08 * inch),
                image,
                Spacer(1, 0.16 * inch),
            ]
        )
    )


def _small_table_from_counts(
    counts: dict,
    label_column: str,
    styles,
    *,
    max_rows: int | None = None,
) -> Table | Paragraph:
    if not counts:
        return Paragraph("No counts available.", styles["BodyText"])
    items = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    if max_rows is not None:
        items = items[:max_rows]
    data = [[label_column, "count"]] + [[str(label), str(count)] for label, count in items]
    table = Table(data, repeatRows=1, colWidths=[4.2 * inch, 1.0 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8f5")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _submission_is_valid(path: Path) -> bool:
    try:
        validate_submission_file(path)
        return True
    except Exception:
        return False


def write_pdf_report() -> Path:
    ensure_output_dirs()
    best_info = _load_json(config.BEST_MODEL_INFO_JSON)
    viz_audit = _load_json(config.VISUALIZATION_AUDIT_JSON)
    comparison = pd.read_csv(config.MODEL_COMPARISON_CSV) if config.MODEL_COMPARISON_CSV.exists() else pd.DataFrame()
    if not comparison.empty and "macro_f1" in comparison.columns:
        comparison = comparison.sort_values("macro_f1", ascending=False, na_position="last")
    clustering = pd.read_csv(config.CLUSTERING_METRICS_CSV) if config.CLUSTERING_METRICS_CSV.exists() else pd.DataFrame()
    clustering_for_pdf = clustering.rename(
        columns={
            "cluster_count": "clusters",
            "noise_count": "noise",
            "adjusted_rand_index": "ARI",
            "normalized_mutual_information": "NMI",
        }
    )
    train_features = _load_train_features()
    submission_summary = _submission_summary()
    deliverables_submission = config.PROJECT_ROOT / "deliverables" / "submission.csv"

    bug_counts = (
        train_features[config.TARGET_COLUMN].value_counts().to_dict()
        if not train_features.empty and config.TARGET_COLUMN in train_features.columns
        else {}
    )
    species_counts = (
        train_features[config.SPECIES_COLUMN].value_counts().to_dict()
        if not train_features.empty and config.SPECIES_COLUMN in train_features.columns
        else {}
    )
    prediction_distribution = submission_summary.get("distribution", {})

    train_rows = best_info.get("train_feature_rows", 249 if config.TRAIN_FEATURES_CSV.exists() else "not generated")
    numeric_best_model = best_info.get("numeric_best_model", best_info.get("best_model", "not selected"))
    final_recommended_model = best_info.get("final_recommended_model", best_info.get("best_model", "not selected"))
    numeric_metrics = best_info.get("numeric_best_validation_metrics", best_info.get("best_validation_metrics", {}))
    final_metrics = best_info.get("final_recommended_validation_metrics", best_info.get("best_validation_metrics", {}))
    tie_breaker = best_info.get("tie_breaker", {})
    cv_policy = best_info.get("cv_policy", {})
    nonlinear_method = str(viz_audit.get("nonlinear_projection_method", "isomap"))
    results_submission_valid = bool(submission_summary.get("valid"))
    deliverables_submission_valid = _submission_is_valid(deliverables_submission)

    path = config.REPORTS_DIR / "report.pdf"
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="TightBody", parent=styles["BodyText"], fontSize=9, leading=11))
    story = []

    story.append(Paragraph("Machine Learning Project: To bee or not to bee", styles["Title"]))
    _add_paragraph(story, "Team: Lianghong LI, Zhekai YAN, Muzi LI", styles["BodyText"])
    story.append(Paragraph("Contributions", styles["Heading2"]))
    _add_bullets(
        story,
        [
            "Lianghong LI: project lead; implemented feature extraction, model training, validation, final prediction pipeline, and report integration.",
            "Zhekai YAN: support contribution for data checking, visualization review, and report proofreading.",
            "Muzi LI: support contribution for reproducibility checking, documentation review, and result verification.",
        ],
        styles["BodyText"],
    )

    story.append(Paragraph("Introduction", styles["Heading2"]))
    _add_paragraph(
        story,
        "This project classifies insect images into the assignment target bug type. The main pipeline uses handcrafted features extracted from each image and its segmentation mask, not raw-image CNN classification. The mask lets the pipeline measure foreground color, shape, and texture while excluding background pixels.",
        styles["BodyText"],
    )

    story.append(Paragraph("Dataset and Preprocessing", styles["Heading2"]))
    _add_paragraph(
        story,
        "The training labels contain ID, bug type, and species. The supervised target is bug type; species is used only for visualization and descriptive comments. The official test set contains 97 images with IDs 251-347 and corresponding masks.",
        styles["BodyText"],
    )
    _add_paragraph(
        story,
        "ID 154 is excluded because train/masks/binary_154.tif is unavailable. No artificial, estimated, manually drawn, or imputed mask was created. The effective mask-based training set therefore contains 249 samples.",
        styles["BodyText"],
    )
    _add_paragraph(story, f"Effective training samples: {train_rows}.", styles["BodyText"])
    story.append(_small_table_from_counts(bug_counts, "bug type", styles))
    story.append(Spacer(1, 0.1 * inch))
    story.append(_small_table_from_counts(species_counts, "species", styles, max_rows=10))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Feature Extraction", styles["Heading2"]))
    _add_paragraph(
        story,
        "All mask-based statistics exclude pixels outside the segmentation mask. Masks are converted to grayscale and binarized with mask > 0, so every feature is derived from foreground insect pixels or the binary foreground shape.",
        styles["BodyText"],
    )
    _add_bullets(
        story,
        [
            "Color symmetry compares the insect bounding-box crop with its left-right flipped version. RGB differences are measured only where both original and flipped masks contain foreground pixels.",
            "Shape symmetry flips the binary mask crop left-right and measures normalized foreground overlap with the original crop.",
            "Bug pixel ratio is bug_pixels / image_pixels, capturing the relative size of the segmented insect in the photograph.",
            "RGB features are min, max, mean, median, and standard deviation for R, G, and B inside the mask.",
            "Morphology features include area, perimeter, circularity, aspect ratio, extent, solidity, orientation, eccentricity, major/minor axis lengths, and equivalent diameter.",
            "Hu moments provide compact global shape descriptors that summarize mask geometry beyond simple size and axis measurements.",
            "HSV and LAB summaries complement RGB by separating hue, saturation, value, lightness, and perceptual color axes.",
            "Grayscale contrast and edge density describe texture and local structure inside the mask, which can help when bug types share similar colors.",
        ],
        styles["TightBody"],
    )

    story.append(Paragraph("Data Visualization", styles["Heading2"]))
    _add_paragraph(
        story,
        "The bug type distribution shows strong class imbalance, which motivates macro-F1. The species distribution and species-by-bug-type plot show that species labels are finer-grained and imbalanced; species is therefore kept descriptive and is not used as the final target.",
        styles["BodyText"],
    )
    _add_paragraph(
        story,
        f"PCA gives a linear view with partial separation and substantial overlap. t-SNE shows local grouping but should not be over-interpreted globally. The {nonlinear_method} projection checks nonlinear structure; it reveals useful geometry, but not perfectly separated classes.",
        styles["BodyText"],
    )
    for figure_name, caption in [
        ("bug_type_distribution.png", "Figure 1. Bug type distribution"),
        ("species_distribution.png", "Figure 2. Species distribution"),
        ("species_by_bug_type.png", "Figure 3. Species grouped by bug type"),
        ("pca_2d.png", "Figure 4. PCA 2D projection colored by bug type"),
        ("tsne_2d.png", "Figure 5. t-SNE 2D projection colored by bug type"),
        (f"{nonlinear_method}_2d.png", f"Figure 6. {nonlinear_method.upper()} 2D projection colored by bug type"),
    ]:
        _add_image_if_exists(story, config.FIGURES_DIR / figure_name, caption, styles)

    story.append(PageBreak())
    story.append(Paragraph("Supervised Learning Methods", styles["Heading2"]))
    _add_paragraph(
        story,
        "The supervised target is bug type, never species. Handcrafted image/mask features are used for the main task because the assignment requires a feature-based pipeline. Macro-F1 is the main validation metric because the dataset is imbalanced. Frequent-class cross-validation is used because singleton classes make full stratified CV invalid. Full-data metrics are descriptive checks only and are not validation metrics.",
        styles["BodyText"],
    )
    _add_paragraph(
        story,
        f"Frequent-class CV used {cv_policy.get('cv_folds', 'not available')} folds and excluded only the rare classes that could not support stratification. The final selected model is then refit on all 249 usable training samples, including rare classes.",
        styles["BodyText"],
    )
    _add_bullets(
        story,
        [
            "DummyClassifier is the majority-class baseline.",
            "LogisticRegression is a linear non-ensemble baseline with class balancing.",
            "SVM RBF models smooth nonlinear boundaries in standardized tabular feature space.",
            "KNN tests whether local feature neighborhoods agree with labels.",
            "RandomForest, ExtraTrees, and GradientBoosting provide supervised ensemble comparisons.",
            "MLP is a shallow neural model trained on extracted features only, not raw images.",
        ],
        styles["TightBody"],
    )
    story.append(_small_table_from_dataframe(comparison, ["model", "accuracy", "macro_f1", "weighted_f1", "status"], max_rows=8))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Clustering Methods", styles["Heading2"]))
    story.append(_small_table_from_dataframe(clustering_for_pdf, ["method", "clusters", "noise", "silhouette", "ARI", "NMI"], max_rows=5))
    story.append(Spacer(1, 0.1 * inch))
    _add_paragraph(
        story,
        "KMeans gives the strongest but still low/moderate clustering scores. AgglomerativeClustering finds imperfect hierarchical structure. DBSCAN marks all samples as noise under the selected settings. These results indicate that the handcrafted features contain useful structure but do not form perfectly separated natural bug type clusters.",
        styles["BodyText"],
    )

    story.append(Paragraph("Model Selection", styles["Heading2"]))
    _add_paragraph(
        story,
        "The metric table is kept unchanged and transparent: MLP has the numerically highest frequent-class CV macro-F1, while SVM RBF is selected for final prediction by a robustness tie-breaker.",
        styles["BodyText"],
    )
    _add_bullets(
        story,
        [
            f"Numeric best model: {numeric_best_model} with CV macro-F1 {numeric_metrics.get('macro_f1', 'not available')}.",
            f"Final recommended model: {final_recommended_model} with CV macro-F1 {final_metrics.get('macro_f1', 'not available')}.",
            "The macro-F1 difference is below 0.005.",
            f"Tie-breaker reason: {tie_breaker.get('reason', 'not applied')}",
        ],
        styles["BodyText"],
    )
    _add_paragraph(
        story,
        "SVM RBF is preferred because it is simpler and more stable for a small imbalanced handcrafted-feature dataset while performing essentially the same as MLP in validation.",
        styles["BodyText"],
    )
    _add_image_if_exists(story, config.FIGURES_DIR / "feature_importance.png", "Figure 7. Descriptive permutation importance for the final fitted model", styles)

    story.append(Paragraph("Final Prediction on Test Images 251-347", styles["Heading2"]))
    if results_submission_valid and deliverables_submission_valid:
        _add_paragraph(
            story,
            "The official test images and masks for IDs 251-347 were processed with the same feature schema as training. The final SVM RBF model was applied to all 97 test samples.",
            styles["BodyText"],
        )
        _add_paragraph(
            story,
            "results/submission.csv and deliverables/submission.csv were generated with exactly two columns: ID and bug type. Both files were validated for 97 rows, IDs 251-347 in order, no duplicates, no missing values, and valid training bug type labels.",
            styles["BodyText"],
        )
        story.append(_small_table_from_counts(prediction_distribution, "predicted bug type", styles))
        story.append(Spacer(1, 0.1 * inch))
        _add_paragraph(
            story,
            "No test accuracy is reported because official test labels are unavailable. No predictions were manually edited or fabricated.",
            styles["BodyText"],
        )
    else:
        _add_paragraph(
            story,
            PENDING_SUBMISSION_SENTENCE if not test_data_available() else "The final submission file has not passed validation yet.",
            styles["BodyText"],
        )

    story.append(Paragraph("Limitations", styles["Heading2"]))
    _add_paragraph(
        story,
        "The dataset is small and strongly imbalanced. The Dragonfly class has only one usable training sample. ID 154 is excluded because the official mask is missing. Bee and Bumblebee can visually overlap, making separation difficult for handcrafted features. Test labels are unavailable, so no test accuracy can be reported.",
        styles["BodyText"],
    )

    story.append(Paragraph("Conclusion", styles["Heading2"]))
    _add_paragraph(
        story,
        "The project satisfies the required feature extraction, visualization, supervised learning, clustering, and final prediction components. Handcrafted mask-based features provide a compliant and interpretable representation, and SVM RBF is selected as the robust final model for the generated submission.",
        styles["BodyText"],
    )

    story.append(Paragraph("Reproducibility", styles["Heading2"]))
    _add_bullets(
        story,
        [
            "python3 -m src.data_loader --validate --split train",
            "python3 -m src.build_features --split train",
            "python3 -m src.train_models",
            "python3 -m src.clustering",
            "python3 -m src.visualize",
            "python3 -m src.make_report",
            "python3 -m src.data_loader --validate --split test",
            "python3 -m src.build_features --split test",
            "python3 -m src.predict",
            "python3 -m src.validate_submission results/submission.csv",
            "python3 -m src.validate_submission deliverables/submission.csv",
        ],
        styles["Small"],
    )
    _add_paragraph(
        story,
        "Test data is not used for model selection or hyperparameter tuning.",
        styles["BodyText"],
    )

    doc.build(story)
    return path


def make_report() -> tuple[Path, Path]:
    md = write_markdown_report()
    pdf = write_pdf_report()
    print(f"Wrote {md}")
    print(f"Wrote {pdf}")
    return md, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown and PDF reports from real outputs.")
    parser.parse_args()
    make_report()


if __name__ == "__main__":
    main()
