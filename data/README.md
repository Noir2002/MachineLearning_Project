# Data Layout

Raw course data is expected locally but is not committed.

Training data:

```text
train/
  classif.xlsx
  1.JPG ... 250.JPG
  masks/
    binary_1.tif ... binary_250.tif
```

Official policy: `train/masks/binary_154.tif` is unavailable. ID `154` is excluded from the feature-based training set. No artificial mask is created.

Future test data:

```text
test/
  251.JPG ... 347.JPG
  masks/
    binary_251.tif ... binary_347.tif
```

Generated feature tables under `data/processed/` are local artifacts and should not be committed.

