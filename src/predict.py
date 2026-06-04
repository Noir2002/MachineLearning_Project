from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_loader import ensure_output_dirs


NON_FEATURE_COLUMNS = {config.ID_COLUMN, config.TARGET_COLUMN, config.SPECIES_COLUMN}


def generate_submission() -> pd.DataFrame:
    ensure_output_dirs()
    if not config.TEST_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Test features not found: {config.TEST_FEATURES_CSV}. "
            "Do not generate a submission until official test images and masks for IDs 251-347 exist."
        )
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {config.BEST_MODEL_PATH}. Run python -m src.train_models first."
        )

    test_features = pd.read_csv(config.TEST_FEATURES_CSV)
    ids = test_features[config.ID_COLUMN].astype(int).tolist()
    expected_ids = list(config.TEST_IDS)
    if ids != expected_ids:
        raise ValueError("Test feature IDs must be exactly 251 through 347 in order.")

    feature_columns = [col for col in test_features.columns if col not in NON_FEATURE_COLUMNS]
    X_test = test_features[feature_columns].replace([np.inf, -np.inf], np.nan)
    model = joblib.load(config.BEST_MODEL_PATH)
    predictions = model.predict(X_test)
    submission = pd.DataFrame({config.ID_COLUMN: ids, config.TARGET_COLUMN: predictions})
    submission.to_csv(config.SUBMISSION_CSV, index=False)
    print(f"Wrote submission to {config.SUBMISSION_CSV}")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final bug type submission.")
    parser.parse_args()
    generate_submission()


if __name__ == "__main__":
    main()

