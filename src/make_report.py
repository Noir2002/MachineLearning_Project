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
    "251–347 will be generated after the test images and masks are released."
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


def _markdown_report() -> str:
    audit = _load_json(config.DATA_AUDIT_JSON)
    best_info = _load_json(config.BEST_MODEL_INFO_JSON)
    viz_audit = _load_json(config.VISUALIZATION_AUDIT_JSON)

    comparison = pd.read_csv(config.MODEL_COMPARISON_CSV) if config.MODEL_COMPARISON_CSV.exists() else pd.DataFrame()
    clustering = pd.read_csv(config.CLUSTERING_METRICS_CSV) if config.CLUSTERING_METRICS_CSV.exists() else pd.DataFrame()
    train_rows = best_info.get("train_feature_rows", 249 if config.TRAIN_FEATURES_CSV.exists() else "not generated")
    numeric_best_model = best_info.get("numeric_best_model", best_info.get("best_model", "not selected"))
    final_recommended_model = best_info.get("final_recommended_model", best_info.get("best_model", "not selected"))
    numeric_metrics = best_info.get("numeric_best_validation_metrics", best_info.get("best_validation_metrics", {}))
    final_metrics = best_info.get("final_recommended_validation_metrics", best_info.get("best_validation_metrics", {}))
    tie_breaker = best_info.get("tie_breaker", {})
    full_metrics = best_info.get("full_data_descriptive_training_metrics", {})
    test_available = test_data_available()

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
        "The main task is supervised classification of insect `bug type` using handcrafted features extracted from each image and its segmentation mask. Raw-image deep learning is not used as the main pipeline.",
        "",
        "## 2. Dataset",
        "",
        "Training labels are read from `train/classif.xlsx` and normalized to `ID`, `bug type`, and `species`. The supervised target is `bug type`. The `species` column is retained only for distribution plots and descriptive analysis.",
        "",
        f"Effective feature-based training samples: `{train_rows}`.",
        "",
        "## 3. Data issue: missing mask for ID 154",
        "",
        "Image ID 154 was excluded because its segmentation mask `train/masks/binary_154.tif` is not available. No artificial mask was created, estimated, manually drawn, or imputed. Since the required feature extraction depends on the segmentation mask, ID 154 is not included in `data/processed/train_features.csv`.",
        "",
        "The exclusion is recorded in `results/data_audit.json`.",
        "",
        "## 4. Feature extraction",
        "",
        "The feature table includes the required color symmetry, shape symmetry, bug pixel ratio, and RGB min/max/mean/median/std features inside the foreground mask. Additional features include bounding box geometry, area, perimeter, circularity, extent, solidity, orientation, eccentricity, axis lengths, equivalent diameter, component count, Hu moments, HSV/LAB summaries, grayscale summaries, and edge density.",
        "",
        "Masks are converted to grayscale and binarized with `mask > 0`. Image-mask dimension mismatches fail clearly unless an explicit nearest-neighbor resize flag is added in code.",
        "",
        "## 5. Data visualization",
        "",
        "Generated figures:",
        "",
        f"- Bug type distribution: `{_figure(config.FIGURES_DIR / 'bug_type_distribution.png')}`",
        f"- Species distribution: `{_figure(config.FIGURES_DIR / 'species_distribution.png')}`",
        f"- Species by bug type: `{_figure(config.FIGURES_DIR / 'species_by_bug_type.png')}`",
        f"- PCA projection: `{_figure(config.FIGURES_DIR / 'pca_2d.png')}`",
        f"- t-SNE projection: `{_figure(config.FIGURES_DIR / 'tsne_2d.png')}`",
        f"- Nonlinear projection method used: `{viz_audit.get('nonlinear_projection_method', 'not generated')}`",
        "",
        "## 6. Supervised models",
        "",
        "The supervised target is `bug type`; `species` is not used as a target. Model comparison uses frequent-class stratified cross-validation because singleton classes make ordinary stratified 5-fold CV invalid for the full label set. The final recommended supervised model is refit on all 249 usable samples, including rare classes.",
        "",
        "Models evaluated: DummyClassifier, LogisticRegression, SVM RBF, KNeighborsClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, and MLPClassifier on extracted features.",
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
        "Full-data training metrics are descriptive only, not validation metrics.",
        "",
        f"Descriptive full-data macro-F1: `{full_metrics.get('macro_f1', 'not available')}`.",
        "",
        "## 7. Clustering methods",
        "",
        "KMeans, AgglomerativeClustering, and DBSCAN are run on imputed and standardized handcrafted features. External metrics compare clusters to `bug type` labels only for interpretation.",
        "",
        clustering_table,
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
        "## 10. Limitations",
        "",
        "The dataset is small and imbalanced, with very rare classes. Cross-validation excludes singleton classes for model comparison, so rare-class performance is documented mainly through the final full-data descriptive fit. The final recommended model remains feature-based and depends on segmentation mask quality.",
        "",
        "## 11. Reproducibility",
        "",
        "Run:",
        "",
        "```bash",
        "python -m compileall src",
        "pytest -q",
        "python -m src.data_loader --validate --split train",
        "python -m src.build_features --split train",
        "python -m src.train_models",
        "python -m src.clustering",
        "python -m src.visualize",
        "python -m src.make_report",
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
    story.append(
        Paragraph(
            "Training labels are normalized to ID, bug type, and species. The supervised target is bug type; species is used only for visualization. "
            "ID 154 was excluded because train/masks/binary_154.tif is unavailable. No artificial mask was created, estimated, manually drawn, or imputed. "
            "The effective training set is 249 samples.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Feature Extraction", styles["Heading2"]))
    story.append(
        Paragraph(
            "Features include color symmetry, shape symmetry, bug pixel ratio, RGB statistics inside the mask, morphology, Hu moments, HSV/LAB statistics, grayscale summaries, and edge density.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Supervised Models", styles["Heading2"]))
    story.append(
        Paragraph(
            "Model comparison uses frequent-class stratified cross-validation because singleton classes make full-label stratified CV invalid. "
            "The final recommended model is refit on all 249 usable samples.",
            styles["BodyText"],
        )
    )
    story.append(_small_table_from_dataframe(comparison, ["model", "accuracy", "macro_f1", "weighted_f1", "status"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            f"Numerically highest CV macro-F1 model: {best_info.get('numeric_best_model', best_info.get('best_model', 'not selected'))} "
            f"({best_info.get('numeric_best_validation_metrics', best_info.get('best_validation_metrics', {})).get('macro_f1', 'not available')}). "
            f"Final recommended submission model: {best_info.get('final_recommended_model', best_info.get('best_model', 'not selected'))} "
            f"({best_info.get('final_recommended_validation_metrics', best_info.get('best_validation_metrics', {})).get('macro_f1', 'not available')}). "
            f"{best_info.get('tie_breaker', {}).get('reason', '')}",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Clustering", styles["Heading2"]))
    story.append(_small_table_from_dataframe(clustering, ["method", "cluster_count", "silhouette", "adjusted_rand_index", "normalized_mutual_information"]))

    story.append(Paragraph("Final Prediction Pipeline", styles["Heading2"]))
    story.append(
        Paragraph(
            PENDING_SUBMISSION_SENTENCE
            if not test_data_available()
            else "Official test data is available, so final CSV generation can be run.",
            styles["BodyText"],
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("Generated Figures", styles["Heading2"]))
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
    story.append(
        Paragraph(
            "Run compileall, pytest, train validation, feature extraction, model training, clustering, visualization, and report generation from the project root. "
            "Do not run final prediction until official test images and masks for IDs 251-347 are available.",
            styles["BodyText"],
        )
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
