from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .data_loader import collect_split_records, ensure_output_dirs
from .features import extract_features


def build_features(split: str) -> pd.DataFrame:
    ensure_output_dirs()
    records = collect_split_records(split, validate=True)
    rows: list[dict] = []

    for index, record in enumerate(records, start=1):
        if index == 1 or index % 25 == 0 or index == len(records):
            print(f"[{split}] extracting features {index}/{len(records)} (ID {record.ID})")
        feature_values = extract_features(record.image_path, record.mask_path)
        row = {
            config.ID_COLUMN: record.ID,
        }
        if split == "train":
            row[config.TARGET_COLUMN] = record.bug_type
            row[config.SPECIES_COLUMN] = record.species
        row.update(feature_values)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(config.ID_COLUMN).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    if split == "train":
        output_path = config.TRAIN_FEATURES_CSV
        metadata_path = config.TRAIN_FEATURES_METADATA_JSON
        if len(df) != 249:
            raise ValueError(f"Expected 249 training feature rows, got {len(df)}")
        if 154 in set(df[config.ID_COLUMN].astype(int)):
            raise ValueError("ID 154 must not appear in training features.")
    elif split == "test":
        output_path = config.TEST_FEATURES_CSV
        metadata_path = config.TEST_FEATURES_METADATA_JSON
        if len(df) != 97:
            raise ValueError(f"Expected 97 test feature rows, got {len(df)}")
    else:
        raise ValueError(f"Unknown split: {split}")

    df.to_csv(output_path, index=False)
    metadata = {
        "split": split,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "feature_columns": [
            col
            for col in df.columns
            if col not in {config.ID_COLUMN, config.TARGET_COLUMN, config.SPECIES_COLUMN}
        ],
        "excluded_ids": sorted(config.EXCLUDED_TRAIN_IDS) if split == "train" else [],
        "id_154_policy": (
            "ID 154 is excluded because its segmentation mask is officially unavailable. "
            "No artificial mask was created, estimated, drawn, or imputed."
        )
        if split == "train"
        else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(df)} rows to {output_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build handcrafted feature tables.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()
    build_features(args.split)


if __name__ == "__main__":
    main()

