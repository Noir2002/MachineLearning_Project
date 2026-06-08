# Machine Learning Project: To bee or not to bee

Team:
- Lianghong LI
- Zekai YAN
- Muzi LI

This repository implements the required main pipeline using handcrafted features extracted from insect images and segmentation masks. The supervised target is `bug type`. The `species` column is used for visualization and reporting context only, not as the supervised learning target.

## Data Policy

The expected local training layout is:

```text
train/
  classif.xlsx
  1.JPG ... 250.JPG
  masks/
    binary_1.tif ... binary_250.tif
```

The segmentation mask for ID `154` is officially unavailable. The project does not fabricate, estimate, manually draw, or impute that mask. ID `154` is excluded from mask-based feature extraction, so the effective labeled training set contains `249` usable samples.

The official test data layout is:

```text
test/
  251.JPG ... 347.JPG
  masks/
    binary_251.tif ... binary_347.tif
```

The official test images and masks for IDs `251`-`347` have been processed. The final CSV is generated at `results/submission.csv` and copied to `deliverables/submission.csv` for submission.

## Reproducible Training Pipeline

From the project root:

```bash
python -m compileall src
pytest -q
python -m src.data_loader --validate --split train
python -m src.build_features --split train
python -m src.train_models
python -m src.clustering
python -m src.visualize
python -m src.make_report
```

If the shell does not provide a `python` executable, use `python3` for the same commands. The scripts in `scripts/` detect this automatically.

Generated local training features are written to `data/processed/train_features.csv`, but this file is derived from the course dataset and should not be committed.

## Final Submission Pipeline

Run this after the final training pipeline has produced the selected supervised model:

```bash
python -m src.data_loader --validate --split test
python -m src.build_features --split test
python -m src.predict
python -m src.validate_submission results/submission.csv
```

The final CSV must contain exactly two columns:

```csv
ID,bug type
```

The committed deliverables folder contains:
- `deliverables/report.pdf`
- `deliverables/submission.csv`

## Outputs

Important generated outputs:
- `results/data_audit.json`
- `results/model_comparison.csv`
- `results/classification_reports.json`
- `results/best_model_info.json`
- `results/clustering_metrics.csv`
- `results/visualization_audit.json`
- `reports/figures/*.png`
- `reports/report.md`
- `reports/report.pdf`

## Final Model Selection Policy

Model comparison keeps the raw validation metric table unchanged. If the numerically best model and a simpler/stabler model have frequent-class CV macro-F1 values within `0.005`, the final recommendation uses this robustness priority: SVM RBF, RandomForest, ExtraTrees, LogisticRegression, KNN, then MLP. This policy is recorded in `results/best_model_info.json`.

## Contributions

- Lianghong LI: project lead; implemented feature extraction, model training, validation, final prediction pipeline, and report integration.
- Zekai YAN: support contribution for data checking, visualization review, and report proofreading.
- Muzi LI: support contribution for reproducibility checking, documentation review, and result verification.
