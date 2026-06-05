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


def _top_model_table(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "Model comparison metrics were not generated."
    columns = ["model", "accuracy", "macro_f1", "weighted_f1", "status"]
    available = [col for col in columns if col in comparison.columns]
    top = comparison.sort_values("macro_f1", ascending=False, na_position="last")[available].head(8)
    return top.to_markdown(index=False)


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


def _markdown_report() -> str:
    audit = _load_json(config.DATA_AUDIT_JSON)
    best_info = _load_json(config.BEST_MODEL_INFO_JSON)
    viz_audit = _load_json(config.VISUALIZATION_AUDIT_JSON)

    comparison = pd.read_csv(config.MODEL_COMPARISON_CSV) if config.MODEL_COMPARISON_CSV.exists() else pd.DataFrame()
    clustering = pd.read_csv(config.CLUSTERING_METRICS_CSV) if config.CLUSTERING_METRICS_CSV.exists() else pd.DataFrame()
    train_features = _load_train_features()
    train_rows = best_info.get("train_feature_rows", 249 if config.TRAIN_FEATURES_CSV.exists() else "not generated")
    numeric_best_model = best_info.get("numeric_best_model", best_info.get("best_model", "not selected"))
    final_recommended_model = best_info.get("final_recommended_model", best_info.get("best_model", "not selected"))
    numeric_metrics = best_info.get("numeric_best_validation_metrics", best_info.get("best_validation_metrics", {}))
    final_metrics = best_info.get("final_recommended_validation_metrics", best_info.get("best_validation_metrics", {}))
    tie_breaker = best_info.get("tie_breaker", {})
    full_metrics = best_info.get("full_data_descriptive_training_metrics", {})
    test_available = test_data_available()
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
    largest_bug_type = max(bug_counts, key=bug_counts.get) if bug_counts else "not available"
    smallest_bug_count = min(bug_counts.values()) if bug_counts else "not available"
    largest_species = max(species_counts, key=species_counts.get) if species_counts else "not available"

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
        "- Zekai YAN",
        "- Muzi LI",
        "",
        "## Contributions",
        "",
        "- Lianghong LI: project lead; implemented feature extraction, model training, validation, final prediction pipeline, and report integration.",
        "- Zekai YAN: support contribution for data checking / visualization review / report proofreading.",
        "- Muzi LI: support contribution for reproducibility checking / documentation review / result verification.",
        "",
        "## 1. Introduction",
        "",
        "The objective is to classify each insect image into the assignment target `bug type`. The main pipeline uses handcrafted features extracted from each image and its segmentation mask. This follows the assignment requirement that the main solution should not be a raw-image convolutional neural network. The mask is used to restrict measurements to the insect foreground rather than the full background-heavy photograph.",
        "",
        "## 2. Dataset",
        "",
        "Training labels are read from `train/classif.xlsx` and normalized to `ID`, `bug type`, and `species`. The supervised target is `bug type`, not `species`. The `species` column is retained only for distribution plots and descriptive analysis because species labels are more detailed than the required final task and could distract model selection from the requested output label.",
        "",
        f"Effective feature-based training samples: `{train_rows}`.",
        "",
        f"Observed bug type counts after the ID 154 exclusion: `{bug_counts}`.",
        "",
        "## 3. Data issue: missing mask for ID 154",
        "",
        "Image ID 154 was excluded because its segmentation mask `train/masks/binary_154.tif` is not available. No artificial mask was created, estimated, manually drawn, or imputed. Since the required feature extraction depends on the segmentation mask, ID 154 is not included in `data/processed/train_features.csv`.",
        "",
        "The exclusion is recorded in `results/data_audit.json`.",
        "",
        "## 4. Feature extraction",
        "",
        "All features are computed from the image and its segmentation mask. Pixels outside the mask are excluded from color and texture statistics so that the background does not dominate the representation.",
        "",
        "- `color_symmetry`: the insect bounding-box crop is compared with its left-right flipped version. Only pixels that are foreground in both the original and flipped mask are compared. The mean absolute RGB difference is normalized by 255 and converted to a similarity score, so more symmetric left/right color patterns receive higher values.",
        "- `shape_symmetry`: the binary mask crop is flipped left-right and overlapped with the original mask crop. The score is the normalized overlap between the original and flipped foreground regions, which measures whether the segmented insect shape is approximately bilaterally symmetric.",
        "- `bug_pixel_ratio`: this is `bug_pixels / image_pixels`, where `bug_pixels` is the number of foreground mask pixels and `image_pixels` is image height times width. It captures the scale of the insect in the photograph.",
        "- RGB summaries: for each R, G, and B channel, the pipeline computes min, max, mean, median, and standard deviation using foreground pixels only. These features summarize brightness and color variation inside the insect body.",
        "- Morphology features: area, perimeter, circularity, bounding-box width and height, aspect ratio, extent, solidity, orientation, eccentricity, major/minor axis lengths, equivalent diameter, component count, and foreground bounding-box fill ratio describe size, elongation, compactness, and shape complexity.",
        "- Hu moments: seven moment invariants provide compact shape descriptors that are relatively stable under translation, scale, and rotation. They complement simpler morphology features by encoding global mask geometry.",
        "- HSV and LAB summaries: HSV separates hue, saturation, and value, while LAB separates lightness from opponent color axes. These color spaces complement RGB because visually similar RGB values can correspond to different perceptual color relationships.",
        "- Grayscale and edge density: grayscale mean/std/contrast and edge density capture texture and local structural variation inside the mask. This can help separate bug types with similar colors but different body or wing texture.",
        "",
        "Masks are converted to grayscale and binarized with `mask > 0`. Image-mask dimension mismatches fail clearly unless an explicit nearest-neighbor resize flag is added in code.",
        "",
        "## 5. Data visualization",
        "",
        "The visualization stage uses only training features and labels. Projection plots color samples by `bug type`; `species` is used only for descriptive distribution graphics.",
        "",
        f"- Bug type distribution (`{_figure(config.FIGURES_DIR / 'bug_type_distribution.png')}`): the classes are strongly imbalanced. The largest class is `{largest_bug_type}`, and the smallest class count is `{smallest_bug_count}`. This motivates macro-F1 rather than accuracy alone.",
        f"- Species distribution (`{_figure(config.FIGURES_DIR / 'species_distribution.png')}`): species are also imbalanced; the largest species group is `{largest_species}`. Species is more granular than the required target, so it is visualized but not used as the supervised target.",
        f"- Species by bug type (`{_figure(config.FIGURES_DIR / 'species_by_bug_type.png')}`): this plot shows how several species map into broader bug type categories. It helps explain why the final classifier predicts `bug type` rather than species.",
        f"- PCA 2D projection (`{_figure(config.FIGURES_DIR / 'pca_2d.png')}`): PCA gives a linear summary of the feature space. The projection shows partial structure but also overlap, which is expected because visually similar insects share color and shape properties.",
        f"- t-SNE 2D projection (`{_figure(config.FIGURES_DIR / 'tsne_2d.png')}`): t-SNE emphasizes local neighborhoods. It is useful for seeing local grouping, but distances between far-away groups should not be over-interpreted.",
        f"- Nonlinear projection (`{viz_audit.get('nonlinear_projection_method', 'not generated')}`): the generated figure is `{_figure(config.FIGURES_DIR / (str(viz_audit.get('nonlinear_projection_method', 'isomap')) + '_2d.png'))}`. This projection checks whether nonlinear geometry reveals class structure beyond PCA. The observed structure is informative but still overlapping, so supervised models are needed.",
        "",
        "## 6. Supervised models",
        "",
        "The supervised target is `bug type`; `species` is not used as a target. The main task therefore matches the required submission column exactly. Handcrafted features are used instead of a raw-image CNN because the assignment requests a feature-based pipeline using images and segmentation masks.",
        "",
        "Macro-F1 is the primary model-selection metric because the label distribution is imbalanced. Accuracy can be dominated by Bee and Bumblebee, while macro-F1 gives each class equal weight. For validation, singleton classes make full stratified cross-validation invalid, so the model-comparison CV excludes only classes with too few samples for valid stratification. The selected final model is then refit on all 249 usable samples, including rare classes. Full-data training metrics are reported as descriptive checks only and are not validation metrics.",
        "",
        "Models evaluated:",
        "",
        f"- DummyClassifier: a most-frequent baseline used to show the minimum expected performance from class imbalance alone. Macro-F1: `{_metric_for_model(comparison, 'dummy_most_frequent', 'macro_f1')}`.",
        f"- LogisticRegression: a linear probabilistic classifier with class balancing, useful as an interpretable non-ensemble baseline. Macro-F1: `{_metric_for_model(comparison, 'logistic_regression', 'macro_f1')}`.",
        f"- SVM RBF: a nonlinear margin-based classifier that can model smooth boundaries in the standardized handcrafted-feature space. Macro-F1: `{_metric_for_model(comparison, 'svm_rbf', 'macro_f1')}`.",
        f"- KNN: a distance-based non-parametric classifier, included to test whether local feature neighborhoods match labels. Macro-F1: `{_metric_for_model(comparison, 'knn', 'macro_f1')}`.",
        f"- RandomForest: an ensemble of decision trees with class balancing, useful for nonlinear interactions and feature robustness. Macro-F1: `{_metric_for_model(comparison, 'random_forest', 'macro_f1')}`.",
        f"- ExtraTrees: a more randomized tree ensemble that can reduce variance and provide a strong ensemble comparison. Macro-F1: `{_metric_for_model(comparison, 'extra_trees', 'macro_f1')}`.",
        f"- GradientBoosting: a sequential tree ensemble that builds additive corrections to previous trees. Macro-F1: `{_metric_for_model(comparison, 'gradient_boosting', 'macro_f1')}`.",
        f"- MLPClassifier: a shallow neural network trained only on extracted features, not raw images. It tests whether a nonlinear learned representation over handcrafted features improves performance. Macro-F1: `{_metric_for_model(comparison, 'mlp', 'macro_f1')}`.",
        "",
        "Model comparison:",
        "",
        _top_model_table(comparison),
        "",
        f"Numerically highest CV macro-F1 model: `{numeric_best_model}`.",
        "",
        f"Numeric best validation macro-F1: `{numeric_metrics.get('macro_f1', 'not available')}`.",
        "",
        f"Final recommended submission model: `{final_recommended_model}`.",
        "",
        f"Final recommended validation macro-F1: `{final_metrics.get('macro_f1', 'not available')}`.",
        "",
        f"Robustness tie-breaker: {tie_breaker.get('reason', 'not applied')}",
        "",
        "The full-data training metrics are descriptive only, not validation metrics, because they are computed on the same 249 samples used for fitting the final model.",
        "",
        f"Descriptive full-data macro-F1: `{full_metrics.get('macro_f1', 'not available')}`.",
        "",
        "## 7. Clustering methods",
        "",
        "KMeans, AgglomerativeClustering, and DBSCAN are run on imputed and standardized handcrafted features. External metrics compare clusters to `bug type` labels only for interpretation, not for supervised model selection.",
        "",
        clustering_table,
        "",
        "KMeans forms a fixed number of compact clusters and obtains the strongest clustering scores among the tested methods, but the silhouette and external agreement scores remain moderate/low. AgglomerativeClustering gives a hierarchical grouping with similarly limited separation. DBSCAN marks all samples as noise under the selected settings, which indicates that the standardized feature space does not contain clean density-separated natural clusters. Overall, the clustering results suggest that the handcrafted features contain useful structure but do not perfectly separate bug types without supervision.",
        "",
        "## 8. Results",
        "",
        "The metric table remains unchanged: MLP has the numerically highest frequent-class CV macro-F1. Because SVM RBF is within 0.005 macro-F1 of MLP and is simpler/stabler on small imbalanced handcrafted-feature data, SVM RBF is the final recommended submission model. Confusion matrices and feature importance plots are generated under `reports/figures/`.",
        "",
        f"- Feature importance: `{_figure(config.FIGURES_DIR / 'feature_importance.png')}`",
        "",
        "## 9. Final prediction pipeline for 251-347",
        "",
        (
            "Official test data is available, so the final submission pipeline can be run."
            if test_available
            else PENDING_SUBMISSION_SENTENCE
        ),
        "",
        "When official test images and masks are available, `src.predict` will load `data/processed/test_features.csv`, apply the saved final model, and write `results/submission.csv` with exactly `ID,bug type` columns.",
        "",
        "No fake submission CSV was generated.",
        "",
        "## 10. Limitations",
        "",
        "The main limitations are the small dataset size, strong class imbalance, singleton classes, and the missing segmentation mask for ID 154. Some bug types are visually similar, so handcrafted color and shape descriptors can overlap. The final test labels are unknown, so the reported metrics are validation estimates on the labeled training data, not final leaderboard performance. The final recommended model remains feature-based and depends on segmentation mask quality.",
        "",
        "## 11. Reproducibility",
        "",
        "Run:",
        "",
        "```bash",
        "python3 -m src.data_loader --validate --split train",
        "python3 -m src.build_features --split train",
        "python3 -m src.train_models",
        "python3 -m src.clustering",
        "python3 -m src.visualize",
        "python3 -m src.make_report",
        "```",
        "",
        "Do not run the final submission pipeline until official test images and masks for IDs 251-347 are present.",
        "",
        "## Audit Notes",
        "",
        f"Data audit file: `results/data_audit.json`; latest audit keys: `{list(audit.keys())}`.",
        "",
    ]
    return "\n".join(lines)


def write_markdown_report() -> Path:
    ensure_output_dirs()
    path = config.REPORTS_DIR / "report.md"
    path.write_text(_markdown_report(), encoding="utf-8")
    return path


def _paragraph(text: str, style):
    return Paragraph(text.replace("&", "&amp;"), style)


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
    story.append(Paragraph(caption, styles["BodyText"]))
    story.append(Spacer(1, 0.08 * inch))
    image = RLImage(str(path))
    image._restrictSize(6.5 * inch, 4.4 * inch)
    story.append(image)
    story.append(Spacer(1, 0.16 * inch))


def write_pdf_report() -> Path:
    ensure_output_dirs()
    best_info = _load_json(config.BEST_MODEL_INFO_JSON)
    viz_audit = _load_json(config.VISUALIZATION_AUDIT_JSON)
    comparison = pd.read_csv(config.MODEL_COMPARISON_CSV) if config.MODEL_COMPARISON_CSV.exists() else pd.DataFrame()
    if not comparison.empty and "macro_f1" in comparison.columns:
        comparison = comparison.sort_values("macro_f1", ascending=False, na_position="last")
    clustering = pd.read_csv(config.CLUSTERING_METRICS_CSV) if config.CLUSTERING_METRICS_CSV.exists() else pd.DataFrame()
    train_features = _load_train_features()
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
    largest_bug_type = max(bug_counts, key=bug_counts.get) if bug_counts else "not available"
    largest_species = max(species_counts, key=species_counts.get) if species_counts else "not available"

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
    story.append(Paragraph("Team: Lianghong LI, Zekai YAN, Muzi LI", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Contributions", styles["Heading2"]))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Lianghong LI: project lead; implemented feature extraction, model training, validation, final prediction pipeline, and report integration.", styles["BodyText"])),
                ListItem(Paragraph("Zekai YAN: support contribution for data checking, visualization review, and report proofreading.", styles["BodyText"])),
                ListItem(Paragraph("Muzi LI: support contribution for reproducibility checking, documentation review, and result verification.", styles["BodyText"])),
            ],
            bulletType="bullet",
        )
    )

    story.append(Paragraph("Dataset and Data Issue", styles["Heading2"]))
    _add_paragraph(
        story,
        "Training labels are normalized to ID, bug type, and species. The supervised target is bug type; species is used only for visualization and descriptive comments. ID 154 was excluded because train/masks/binary_154.tif is unavailable. No artificial mask was created, estimated, manually drawn, or imputed. The effective training set is 249 samples.",
        styles["BodyText"],
    )
    _add_paragraph(story, f"Bug type counts after exclusion: {bug_counts}. Largest species group: {largest_species}.", styles["Small"])

    story.append(Paragraph("Feature Extraction", styles["Heading2"]))
    _add_paragraph(
        story,
        "All features are computed from the image and segmentation mask. Pixels outside the mask are excluded from color and texture statistics so that the background does not dominate the representation.",
        styles["BodyText"],
    )
    _add_bullets(
        story,
        [
            "Color symmetry compares the left-right RGB difference inside the bug bounding box where both original and flipped masks are foreground.",
            "Shape symmetry flips the binary mask crop left-right and measures normalized foreground overlap.",
            "Bug pixel ratio is bug_pixels / image_pixels.",
            "RGB features are min, max, mean, median, and standard deviation for R, G, and B inside the mask.",
            "Morphology features describe area, perimeter, circularity, aspect ratio, extent, solidity, orientation, eccentricity, and axis lengths.",
            "Hu moments provide global shape descriptors that are relatively stable under translation, scale, and rotation.",
            "HSV/LAB features complement RGB by separating perceptual color properties and lightness.",
            "Grayscale contrast and edge density capture texture and local structure that can help separate visually similar bug types.",
        ],
        styles["TightBody"],
    )

    story.append(Paragraph("Supervised Models", styles["Heading2"]))
    _add_paragraph(
        story,
        "The target is bug type, not species. Handcrafted features are used for the main task because the assignment requires a mask-based feature pipeline. Macro-F1 is the primary metric because class imbalance makes accuracy alone insufficient. Frequent-class stratified CV is used because singleton classes make full-label stratified CV invalid; full-data metrics are descriptive only.",
        styles["BodyText"],
    )
    story.append(_small_table_from_dataframe(comparison, ["model", "accuracy", "macro_f1", "weighted_f1", "status"]))
    story.append(Spacer(1, 0.1 * inch))
    _add_paragraph(
        story,
        f"Numerically highest CV macro-F1 model: {best_info.get('numeric_best_model', best_info.get('best_model', 'not selected'))} ({best_info.get('numeric_best_validation_metrics', best_info.get('best_validation_metrics', {})).get('macro_f1', 'not available')}). Final recommended submission model: {best_info.get('final_recommended_model', best_info.get('best_model', 'not selected'))} ({best_info.get('final_recommended_validation_metrics', best_info.get('best_validation_metrics', {})).get('macro_f1', 'not available')}). {best_info.get('tie_breaker', {}).get('reason', '')}",
        styles["BodyText"],
    )
    _add_bullets(
        story,
        [
            f"DummyClassifier is the imbalance baseline. Macro-F1: {_metric_for_model(comparison, 'dummy_most_frequent', 'macro_f1')}.",
            f"LogisticRegression is the linear non-ensemble baseline. Macro-F1: {_metric_for_model(comparison, 'logistic_regression', 'macro_f1')}.",
            f"SVM RBF models smooth nonlinear boundaries and is the final robust recommendation. Macro-F1: {_metric_for_model(comparison, 'svm_rbf', 'macro_f1')}.",
            f"KNN tests local neighborhood agreement. Macro-F1: {_metric_for_model(comparison, 'knn', 'macro_f1')}.",
            f"RandomForest, ExtraTrees, and GradientBoosting provide ensemble comparisons. Their macro-F1 values are {_metric_for_model(comparison, 'random_forest', 'macro_f1')}, {_metric_for_model(comparison, 'extra_trees', 'macro_f1')}, and {_metric_for_model(comparison, 'gradient_boosting', 'macro_f1')}.",
            f"MLP is a shallow neural network on extracted features only, not raw images. Macro-F1: {_metric_for_model(comparison, 'mlp', 'macro_f1')}.",
        ],
        styles["TightBody"],
    )

    story.append(Paragraph("Clustering", styles["Heading2"]))
    story.append(_small_table_from_dataframe(clustering, ["method", "cluster_count", "silhouette", "adjusted_rand_index", "normalized_mutual_information"]))
    _add_paragraph(
        story,
        "KMeans gives the strongest but still moderate/low clustering scores. AgglomerativeClustering finds related but imperfect hierarchical groups. DBSCAN marks all samples as noise under the selected settings. These results indicate useful feature structure but not perfectly separated natural clusters.",
        styles["BodyText"],
    )

    story.append(Paragraph("Final Prediction Pipeline", styles["Heading2"]))
    _add_paragraph(
        story,
        PENDING_SUBMISSION_SENTENCE
        if not test_data_available()
        else "Official test data is available, so final CSV generation can be run.",
        styles["BodyText"],
    )
    _add_paragraph(
        story,
        "No fake submission CSV is generated. When official test images and masks are available, the final pipeline writes exactly ID and bug type columns for IDs 251-347.",
        styles["BodyText"],
    )

    story.append(PageBreak())
    story.append(Paragraph("Generated Figures", styles["Heading2"]))
    _add_paragraph(
        story,
        f"The bug type distribution highlights imbalance, with {largest_bug_type} as the largest class. Species plots show finer-grained label imbalance but species is not the target. PCA, t-SNE, and {viz_audit.get('nonlinear_projection_method', 'isomap')} projections show partial grouping with overlap, supporting the need for supervised classification.",
        styles["BodyText"],
    )
    for figure_name, caption in [
        ("bug_type_distribution.png", "Bug type distribution"),
        ("species_distribution.png", "Species distribution"),
        ("species_by_bug_type.png", "Species grouped by bug type"),
        ("pca_2d.png", "PCA projection"),
        ("tsne_2d.png", "t-SNE projection"),
        (f"{viz_audit.get('nonlinear_projection_method', 'isomap')}_2d.png", "Nonlinear projection"),
        ("feature_importance.png", "Feature importance"),
    ]:
        _add_image_if_exists(story, config.FIGURES_DIR / figure_name, caption, styles)

    story.append(Paragraph("Reproducibility", styles["Heading2"]))
    _add_paragraph(
        story,
        "Run: python3 -m src.data_loader --validate --split train; python3 -m src.build_features --split train; python3 -m src.train_models; python3 -m src.clustering; python3 -m src.visualize; python3 -m src.make_report. Do not run final prediction until official test images and masks for IDs 251-347 are available.",
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
