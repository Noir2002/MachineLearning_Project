from src.data_loader import expected_ids_for_split, extract_numeric_id, normalize_label_columns

import pandas as pd


def test_extract_numeric_id_uses_last_number():
    assert extract_numeric_id("image_001.jpg") == 1
    assert extract_numeric_id("mask_binary_251.tif") == 251


def test_id_154_excluded_from_expected_training_ids():
    ids = expected_ids_for_split("train")
    assert len(ids) == 249
    assert 154 not in ids


def test_normalize_label_columns():
    df = pd.DataFrame(
        {
            " image_id ": [1],
            "BugType": ["Bee"],
            "Species": ["Apis mellifera"],
        }
    )
    normalized = normalize_label_columns(df)
    assert list(normalized.columns) == ["ID", "bug type", "species"]
    assert normalized.loc[0, "bug type"] == "Bee"

