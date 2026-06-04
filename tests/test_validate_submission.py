from pathlib import Path

import pandas as pd
import pytest

from src.validate_submission import validate_submission_file


def test_submission_validator_accepts_valid_file(tmp_path: Path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": list(range(251, 348)), "bug type": ["Bee"] * 97}).to_csv(path, index=False)
    validate_submission_file(path, check_known_labels=False)


def test_submission_validator_rejects_extra_column(tmp_path: Path):
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "ID": list(range(251, 348)),
            "bug type": ["Bee"] * 97,
            "species": ["Apis mellifera"] * 97,
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="columns"):
        validate_submission_file(path, check_known_labels=False)


def test_submission_validator_rejects_wrong_ids(tmp_path: Path):
    path = tmp_path / "submission.csv"
    pd.DataFrame({"ID": list(range(250, 347)), "bug type": ["Bee"] * 97}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="IDs"):
        validate_submission_file(path, check_known_labels=False)

