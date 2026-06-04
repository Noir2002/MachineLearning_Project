from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .data_loader import load_labels


def valid_training_labels() -> set[str]:
    labels = load_labels()
    return set(labels[config.TARGET_COLUMN].dropna().astype(str))


def validate_submission_file(path: str | Path, *, check_known_labels: bool = True) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")

    df = pd.read_csv(path)
    expected_columns = [config.ID_COLUMN, config.TARGET_COLUMN]
    if list(df.columns) != expected_columns:
        raise ValueError(f"Submission columns must be exactly {expected_columns}, got {list(df.columns)}")
    if len(df) != 97:
        raise ValueError(f"Submission must contain exactly 97 rows, got {len(df)}")
    ids = df[config.ID_COLUMN].tolist()
    expected_ids = list(config.TEST_IDS)
    if ids != expected_ids:
        raise ValueError("Submission IDs must be exactly 251 through 347 in order.")
    if df[config.ID_COLUMN].duplicated().any():
        raise ValueError("Submission contains duplicate IDs.")
    if df.isna().any().any():
        raise ValueError("Submission contains missing values.")
    if check_known_labels:
        known = valid_training_labels()
        unknown = sorted(set(df[config.TARGET_COLUMN].astype(str)) - known)
        if unknown:
            raise ValueError(f"Submission contains labels not present in training data: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final submission CSV.")
    parser.add_argument("path", nargs="?", default=str(config.SUBMISSION_CSV))
    args = parser.parse_args()
    validate_submission_file(args.path)
    print(f"Submission is valid: {args.path}")


if __name__ == "__main__":
    main()

