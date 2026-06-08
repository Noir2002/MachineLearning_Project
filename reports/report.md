# Machine Learning Project: To bee or not to bee

## Team

- Lianghong LI
- Zhekai YAN
- Muzi LI

## Contributions

- Lianghong LI: project lead; implemented feature extraction, model training, validation, final prediction pipeline, and report integration.
- Zhekai YAN: support contribution for data checking, visualization review, and report proofreading.
- Muzi LI: support contribution for reproducibility checking, documentation review, and result verification.

## Introduction

This project classifies insect images into the assignment target `bug type`. The main pipeline uses handcrafted features extracted from each image and its segmentation mask, rather than raw-image CNN classification. This design follows the assignment requirements and makes the solution depend on measurable foreground color, shape, and texture properties.

## Dataset and Preprocessing

The training labels contain `ID`, `bug type`, and `species`. The main supervised target is `bug type`; `species` is used only for visualization and descriptive analysis. The official test set contains 97 images with IDs 251-347 and corresponding segmentation masks.

ID 154 is excluded because `train/masks/binary_154.tif` is unavailable. No artificial, estimated, manually drawn, or imputed mask was created. The effective mask-based training set therefore contains 249 samples.

Effective training samples: `249`.

This exclusion is recorded in `results/data_audit.json`.

Training bug type distribution:

| bug type | count |
|---|---:|
| Bee | 115 |
| Bumblebee | 100 |
| Butterfly | 15 |
| Hover fly | 9 |
| Wasp | 9 |
| Dragonfly | 1 |

Largest species groups, shown for descriptive context only:

| species | count |
|---|---:|
| Bombus hortorum | 71 |
| Apis mellifera | 58 |
| Bombus pascuorum | 25 |
| Anthidium manicatum | 19 |
| Megachile centuncularis | 17 |
| Eristalis | 9 |
| Vespula germanica | 8 |
| Anthidium | 6 |
| Macroglossum stellatarum | 5 |
| Andrenidae | 4 |

## Feature Extraction

All mask-based statistics exclude pixels outside the segmentation mask, so background content does not dominate the representation.

- Color symmetry compares the insect bounding-box crop with its left-right flipped version. RGB differences are measured only where both original and flipped masks contain foreground pixels.
- Shape symmetry flips the binary mask crop left-right and measures normalized foreground overlap with the original mask.
- Bug pixel ratio is computed as `bug_pixels / image_pixels`.
- RGB features are the min, max, mean, median, and standard deviation for R, G, and B values inside the mask.
- Morphology features include area, perimeter, circularity, aspect ratio, extent, solidity, orientation, eccentricity, and major/minor axis lengths.
- Hu moments are global shape descriptors based on image moments and provide compact information about overall mask geometry.
- HSV and LAB features complement RGB by separating hue/saturation/value and perceptual lightness/color axes.
- Grayscale contrast and edge density describe texture and local structure inside the mask.

## Data Visualization

- Bug type distribution: `reports/figures/bug_type_distribution.png`. The labeled data is strongly imbalanced, with Bee and Bumblebee dominating.
- Species distribution: `reports/figures/species_distribution.png`. Species labels are finer-grained and imbalanced, but species is not used as the final target.
- Species grouped by bug type: `reports/figures/species_by_bug_type.png`. This shows how fine labels relate to the broader required bug type labels.
- PCA: `reports/figures/pca_2d.png`. The linear projection shows limited separation and substantial overlap.
- t-SNE: `reports/figures/tsne_2d.png`. Local grouping is visible, but classes still overlap.
- Isomap: `reports/figures/isomap_2d.png`. Nonlinear structure is present, but it does not fully separate all bug types.

## Supervised Learning Methods

The supervised target is `bug type`; `species` is never used as the target. Macro-F1 is the main validation metric because class imbalance makes accuracy alone insufficient. Frequent-class cross-validation is used because singleton classes make full stratified CV invalid. Full-data metrics are descriptive only and are not validation metrics.

- DummyClassifier is the majority-class baseline.
- LogisticRegression is a linear non-ensemble baseline.
- SVM RBF is a nonlinear and robust classifier for standardized tabular features.
- KNN is a local-neighborhood classifier.
- RandomForest, ExtraTrees, and GradientBoosting are supervised ensemble methods.
- MLP is an optional neural model trained on extracted features only, not raw images.

Model comparison:

| model | accuracy | macro-F1 | weighted-F1 |
|---|---:|---:|---:|
| dummy_most_frequent | 0.463710 | 0.126722 | 0.293811 |
| logistic_regression | 0.850806 | 0.697796 | 0.856491 |
| svm_rbf | 0.862903 | 0.753539 | 0.858947 |
| knn | 0.786290 | 0.582899 | 0.766332 |
| random_forest | 0.818548 | 0.563813 | 0.795637 |
| extra_trees | 0.846774 | 0.618401 | 0.826811 |
| gradient_boosting | 0.798387 | 0.597056 | 0.789721 |
| mlp | 0.870968 | 0.755352 | 0.867893 |

## Clustering Methods

KMeans, AgglomerativeClustering, and DBSCAN are evaluated on imputed and standardized handcrafted features. Their cluster assignments are compared with `bug type` only for interpretation.

| method        |   cluster_count |   noise_count |   silhouette |   adjusted_rand_index |   normalized_mutual_information |
|:--------------|----------------:|--------------:|-------------:|----------------------:|--------------------------------:|
| kmeans        |               6 |             0 |    0.112834  |             0.148929  |                        0.234865 |
| agglomerative |               6 |             0 |    0.0768386 |             0.0757599 |                        0.181108 |
| dbscan        |               0 |           249 |  nan         |             0         |                        0        |

KMeans has the strongest clustering metrics among the tested methods, but the scores remain low to moderate. AgglomerativeClustering finds imperfect hierarchical structure. DBSCAN marks all samples as noise under the selected settings. These results suggest that the features contain useful structure but not naturally perfect bug type clusters.

## Model Selection

The metric table is kept honest: MLP has the numerically highest CV macro-F1, while SVM RBF is selected as the final model by robustness tie-breaker.

- Numeric best model: `mlp` with CV macro-F1 `0.7553521749357212`.
- Final model: `svm_rbf` with CV macro-F1 `0.7535392750651922`.
- The difference is below 0.005.
- Tie-breaker reason: SVM RBF selected by robustness tie-breaker because its macro-F1 is within 0.005 of MLP and it is simpler/stabler for small imbalanced data.

SVM RBF is preferred for final prediction because it is simpler and more stable for a small imbalanced handcrafted-feature dataset while performing essentially the same as MLP in validation.

## Final Prediction on Test Images 251-347

The official test images and masks for IDs 251-347 were processed. The final SVM RBF model was applied to 97 test samples.

`results/submission.csv` and `deliverables/submission.csv` were generated with exactly two columns: `ID` and `bug type`. The CSV was validated for 97 rows, IDs 251-347, no duplicate IDs, no missing values, and valid training bug type labels.

Prediction distribution:

| predicted bug type | count |
|---|---:|
| Bee | 46 |
| Bumblebee | 38 |
| Butterfly | 6 |
| Wasp | 5 |
| Hover fly | 2 |

No test accuracy is reported because test labels are unavailable.

## Limitations

The dataset is small and strongly imbalanced. The Dragonfly class has only one usable sample. ID 154 is excluded because the official mask is missing. Bee and Bumblebee can visually overlap, making separation difficult for handcrafted features. Test labels are unavailable, so no test accuracy can be reported.

## Conclusion

The project satisfies the required feature extraction, visualization, supervised learning, clustering, and final prediction components. Handcrafted mask-based features are useful for the task, SVM RBF was selected for robust final prediction, and the final submission CSV was generated and validated.

## Reproducibility

```bash
python3 -m src.data_loader --validate --split train
python3 -m src.build_features --split train
python3 -m src.train_models
python3 -m src.clustering
python3 -m src.visualize
python3 -m src.make_report
python3 -m src.data_loader --validate --split test
python3 -m src.build_features --split test
python3 -m src.predict
python3 -m src.validate_submission results/submission.csv
```

Test data is not used for model selection or hyperparameter tuning.
