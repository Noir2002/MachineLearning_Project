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

The objective is to classify each insect image into the assignment target `bug type`. The main pipeline uses handcrafted features extracted from each image and its segmentation mask. This follows the assignment requirement that the main solution should not be a raw-image convolutional neural network. The mask is used to restrict measurements to the insect foreground rather than the full background-heavy photograph.

## 2. Dataset

Training labels are read from `train/classif.xlsx` and normalized to `ID`, `bug type`, and `species`. The supervised target is `bug type`, not `species`. The `species` column is retained only for distribution plots and descriptive analysis because species labels are more detailed than the required final task and could distract model selection from the requested output label.

Effective feature-based training samples: `249`.

Observed bug type counts after the ID 154 exclusion: `{'Bee': 115, 'Bumblebee': 100, 'Butterfly': 15, 'Hover fly': 9, 'Wasp': 9, 'Dragonfly': 1}`.

## 3. Data issue: missing mask for ID 154

Image ID 154 was excluded because its segmentation mask `train/masks/binary_154.tif` is not available. No artificial mask was created, estimated, manually drawn, or imputed. Since the required feature extraction depends on the segmentation mask, ID 154 is not included in `data/processed/train_features.csv`.

The exclusion is recorded in `results/data_audit.json`.

## 4. Feature extraction

All features are computed from the image and its segmentation mask. Pixels outside the mask are excluded from color and texture statistics so that the background does not dominate the representation.

- `color_symmetry`: the insect bounding-box crop is compared with its left-right flipped version. Only pixels that are foreground in both the original and flipped mask are compared. The mean absolute RGB difference is normalized by 255 and converted to a similarity score, so more symmetric left/right color patterns receive higher values.
- `shape_symmetry`: the binary mask crop is flipped left-right and overlapped with the original mask crop. The score is the normalized overlap between the original and flipped foreground regions, which measures whether the segmented insect shape is approximately bilaterally symmetric.
- `bug_pixel_ratio`: this is `bug_pixels / image_pixels`, where `bug_pixels` is the number of foreground mask pixels and `image_pixels` is image height times width. It captures the scale of the insect in the photograph.
- RGB summaries: for each R, G, and B channel, the pipeline computes min, max, mean, median, and standard deviation using foreground pixels only. These features summarize brightness and color variation inside the insect body.
- Morphology features: area, perimeter, circularity, bounding-box width and height, aspect ratio, extent, solidity, orientation, eccentricity, major/minor axis lengths, equivalent diameter, component count, and foreground bounding-box fill ratio describe size, elongation, compactness, and shape complexity.
- Hu moments: seven moment invariants provide compact shape descriptors that are relatively stable under translation, scale, and rotation. They complement simpler morphology features by encoding global mask geometry.
- HSV and LAB summaries: HSV separates hue, saturation, and value, while LAB separates lightness from opponent color axes. These color spaces complement RGB because visually similar RGB values can correspond to different perceptual color relationships.
- Grayscale and edge density: grayscale mean/std/contrast and edge density capture texture and local structural variation inside the mask. This can help separate bug types with similar colors but different body or wing texture.

Masks are converted to grayscale and binarized with `mask > 0`. Image-mask dimension mismatches fail clearly unless an explicit nearest-neighbor resize flag is added in code.

## 5. Data visualization

The visualization stage uses only training features and labels. Projection plots color samples by `bug type`; `species` is used only for descriptive distribution graphics.

- Bug type distribution (`reports/figures/bug_type_distribution.png`): the classes are strongly imbalanced. The largest class is `Bee`, and the smallest class count is `1`. This motivates macro-F1 rather than accuracy alone.
- Species distribution (`reports/figures/species_distribution.png`): species are also imbalanced; the largest species group is `Bombus hortorum`. Species is more granular than the required target, so it is visualized but not used as the supervised target.
- Species by bug type (`reports/figures/species_by_bug_type.png`): this plot shows how several species map into broader bug type categories. It helps explain why the final classifier predicts `bug type` rather than species.
- PCA 2D projection (`reports/figures/pca_2d.png`): PCA gives a linear summary of the feature space. The projection shows partial structure but also overlap, which is expected because visually similar insects share color and shape properties.
- t-SNE 2D projection (`reports/figures/tsne_2d.png`): t-SNE emphasizes local neighborhoods. It is useful for seeing local grouping, but distances between far-away groups should not be over-interpreted.
- Nonlinear projection (`isomap`): the generated figure is `reports/figures/isomap_2d.png`. This projection checks whether nonlinear geometry reveals class structure beyond PCA. The observed structure is informative but still overlapping, so supervised models are needed.

## 6. Supervised models

The supervised target is `bug type`; `species` is not used as a target. The main task therefore matches the required submission column exactly. Handcrafted features are used instead of a raw-image CNN because the assignment requests a feature-based pipeline using images and segmentation masks.

Macro-F1 is the primary model-selection metric because the label distribution is imbalanced. Accuracy can be dominated by Bee and Bumblebee, while macro-F1 gives each class equal weight. For validation, singleton classes make full stratified cross-validation invalid, so the model-comparison CV excludes only classes with too few samples for valid stratification. The selected final model is then refit on all 249 usable samples, including rare classes. Full-data training metrics are reported as descriptive checks only and are not validation metrics.

Models evaluated:

- DummyClassifier: a most-frequent baseline used to show the minimum expected performance from class imbalance alone. Macro-F1: `0.126722`.
- LogisticRegression: a linear probabilistic classifier with class balancing, useful as an interpretable non-ensemble baseline. Macro-F1: `0.697796`.
- SVM RBF: a nonlinear margin-based classifier that can model smooth boundaries in the standardized handcrafted-feature space. Macro-F1: `0.753539`.
- KNN: a distance-based non-parametric classifier, included to test whether local feature neighborhoods match labels. Macro-F1: `0.582899`.
- RandomForest: an ensemble of decision trees with class balancing, useful for nonlinear interactions and feature robustness. Macro-F1: `0.563813`.
- ExtraTrees: a more randomized tree ensemble that can reduce variance and provide a strong ensemble comparison. Macro-F1: `0.618401`.
- GradientBoosting: a sequential tree ensemble that builds additive corrections to previous trees. Macro-F1: `0.597056`.
- MLPClassifier: a shallow neural network trained only on extracted features, not raw images. It tests whether a nonlinear learned representation over handcrafted features improves performance. Macro-F1: `0.755352`.

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

Numerically highest CV macro-F1 model: `mlp`.

Numeric best validation macro-F1: `0.7553521749357212`.

Final recommended submission model: `svm_rbf`.

Final recommended validation macro-F1: `0.7535392750651922`.

Robustness tie-breaker: SVM RBF selected by robustness tie-breaker because its macro-F1 is within 0.005 of MLP and it is simpler/stabler for small imbalanced data.

The full-data training metrics are descriptive only, not validation metrics, because they are computed on the same 249 samples used for fitting the final model.

Descriptive full-data macro-F1: `0.9953290316974082`.

## 7. Clustering methods

KMeans, AgglomerativeClustering, and DBSCAN are run on imputed and standardized handcrafted features. External metrics compare clusters to `bug type` labels only for interpretation, not for supervised model selection.

| method        |   cluster_count |   noise_count |   silhouette |   adjusted_rand_index |   normalized_mutual_information |
|:--------------|----------------:|--------------:|-------------:|----------------------:|--------------------------------:|
| kmeans        |               6 |             0 |    0.112834  |             0.148929  |                        0.234865 |
| agglomerative |               6 |             0 |    0.0768386 |             0.0757599 |                        0.181108 |
| dbscan        |               0 |           249 |  nan         |             0         |                        0        |

KMeans forms a fixed number of compact clusters and obtains the strongest clustering scores among the tested methods, but the silhouette and external agreement scores remain moderate/low. AgglomerativeClustering gives a hierarchical grouping with similarly limited separation. DBSCAN marks all samples as noise under the selected settings, which indicates that the standardized feature space does not contain clean density-separated natural clusters. Overall, the clustering results suggest that the handcrafted features contain useful structure but do not perfectly separate bug types without supervision.

## 8. Results

The metric table remains unchanged: MLP has the numerically highest frequent-class CV macro-F1. Because SVM RBF is within 0.005 macro-F1 of MLP and is simpler/stabler on small imbalanced handcrafted-feature data, SVM RBF is the final recommended submission model. Confusion matrices and feature importance plots are generated under `reports/figures/`.

- Feature importance: `reports/figures/feature_importance.png`

## 9. Final prediction pipeline for 251-347

The final submission generation pipeline has been implemented, but the official CSV for images 251-347 will be generated after the test images and masks are released.

When official test images and masks are available, `src.predict` will load `data/processed/test_features.csv`, apply the saved final model, and write `results/submission.csv` with exactly `ID,bug type` columns.

No fake submission CSV was generated.

## 10. Limitations

The main limitations are the small dataset size, strong class imbalance, singleton classes, and the missing segmentation mask for ID 154. Some bug types are visually similar, so handcrafted color and shape descriptors can overlap. The final test labels are unknown, so the reported metrics are validation estimates on the labeled training data, not final leaderboard performance. The final recommended model remains feature-based and depends on segmentation mask quality.

## 11. Reproducibility

Run:

```bash
python3 -m src.data_loader --validate --split train
python3 -m src.build_features --split train
python3 -m src.train_models
python3 -m src.clustering
python3 -m src.visualize
python3 -m src.make_report
```

Do not run the final submission pipeline until official test images and masks for IDs 251-347 are present.

## Audit Notes

Data audit file: `results/data_audit.json`; latest audit keys: `['train', 'latest_split', 'id_154_policy', 'excluded_train_ids']`.
