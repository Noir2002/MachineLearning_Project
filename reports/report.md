# Machine Learning Project: To bee or not to bee

## Team

- Lianghong LI
- Zekai YAN
- Muzi LI

## Contributions

- Lianghong LI: project lead; implemented feature extraction, model training, validation, final prediction pipeline, and report integration.
- Zekai YAN: support contribution for data checking / visualization review / report proofreading.
- Muzi LI: support contribution for reproducibility checking / documentation review / result verification.

## 1. Introduction

The main task is supervised classification of insect `bug type` using handcrafted features extracted from each image and its segmentation mask. Raw-image deep learning is not used as the main pipeline.

## 2. Dataset

Training labels are read from `train/classif.xlsx` and normalized to `ID`, `bug type`, and `species`. The supervised target is `bug type`. The `species` column is retained only for distribution plots and descriptive analysis.

Effective feature-based training samples: `249`.

## 3. Data issue: missing mask for ID 154

Image ID 154 was excluded because its segmentation mask `train/masks/binary_154.tif` is not available. No artificial mask was created, estimated, manually drawn, or imputed. Since the required feature extraction depends on the segmentation mask, ID 154 is not included in `data/processed/train_features.csv`.

The exclusion is recorded in `results/data_audit.json`.

## 4. Feature extraction

The feature table includes the required color symmetry, shape symmetry, bug pixel ratio, and RGB min/max/mean/median/std features inside the foreground mask. Additional features include bounding box geometry, area, perimeter, circularity, extent, solidity, orientation, eccentricity, axis lengths, equivalent diameter, component count, Hu moments, HSV/LAB summaries, grayscale summaries, and edge density.

Masks are converted to grayscale and binarized with `mask > 0`. Image-mask dimension mismatches fail clearly unless an explicit nearest-neighbor resize flag is added in code.

## 5. Data visualization

Generated figures:

- Bug type distribution: `reports/figures/bug_type_distribution.png`
- Species distribution: `reports/figures/species_distribution.png`
- Species by bug type: `reports/figures/species_by_bug_type.png`
- PCA projection: `reports/figures/pca_2d.png`
- t-SNE projection: `reports/figures/tsne_2d.png`
- Nonlinear projection method used: `isomap`

## 6. Supervised models

The supervised target is `bug type`; `species` is not used as a target. Model comparison uses frequent-class stratified cross-validation because singleton classes make ordinary stratified 5-fold CV invalid for the full label set. The final selected supervised model is refit on all 249 usable samples, including rare classes.

Models evaluated: DummyClassifier, LogisticRegression, SVM RBF, KNeighborsClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, and MLPClassifier on extracted features.

Model comparison:

| model               |   accuracy |   macro_f1 |   weighted_f1 | status   |
|:--------------------|-----------:|-----------:|--------------:|:---------|
| mlp                 |   0.870968 |   0.755352 |      0.867893 | ok       |
| svm_rbf             |   0.862903 |   0.753539 |      0.858947 | ok       |
| logistic_regression |   0.850806 |   0.697796 |      0.856491 | ok       |
| extra_trees         |   0.846774 |   0.618401 |      0.826811 | ok       |
| gradient_boosting   |   0.798387 |   0.597056 |      0.789721 | ok       |
| knn                 |   0.78629  |   0.582899 |      0.766332 | ok       |
| random_forest       |   0.818548 |   0.563813 |      0.795637 | ok       |
| dummy_most_frequent |   0.46371  |   0.126722 |      0.293811 | ok       |

Selected model: `mlp`.

Selected validation macro-F1: `0.7553521749357212`.

Full-data training metrics are descriptive only, not validation metrics.

Descriptive full-data macro-F1: `1.0`.

## 7. Clustering methods

KMeans, AgglomerativeClustering, and DBSCAN are run on imputed and standardized handcrafted features. External metrics compare clusters to `bug type` labels only for interpretation.

| method        |   cluster_count |   noise_count |   silhouette |   adjusted_rand_index |   normalized_mutual_information |
|:--------------|----------------:|--------------:|-------------:|----------------------:|--------------------------------:|
| kmeans        |               6 |             0 |    0.112834  |             0.148929  |                        0.234865 |
| agglomerative |               6 |             0 |    0.0768386 |             0.0757599 |                        0.181108 |
| dbscan        |               0 |           249 |  nan         |             0         |                        0        |

## 8. Results

The selected model is chosen by frequent-class cross-validation macro-F1. Confusion matrices and feature importance plots are generated under `reports/figures/`.

- Feature importance: `reports/figures/feature_importance.png`

## 9. Final prediction pipeline for 251-347

The final submission generation pipeline has been implemented, but the official CSV for images 251–347 will be generated after the test images and masks are released.

When official test images and masks are available, `src.predict` will load `data/processed/test_features.csv`, apply the saved final model, and write `results/submission.csv` with exactly `ID,bug type` columns.

## 10. Limitations

The dataset is small and imbalanced, with very rare classes. Cross-validation excludes singleton classes for model comparison, so rare-class performance is documented mainly through the final full-data descriptive fit. The final model remains feature-based and depends on segmentation mask quality.

## 11. Reproducibility

Run:

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

Do not run the final submission pipeline until official test images and masks for IDs 251-347 are present.

## Audit Notes

Data audit file: `results/data_audit.json`; latest audit keys: `['train', 'latest_split', 'id_154_policy', 'excluded_train_ids']`.
