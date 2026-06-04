from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image

from . import config


@dataclass(frozen=True)
class DataRecord:
    ID: int
    image_path: Path
    mask_path: Path
    bug_type: str | None = None
    species: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["image_path"] = str(self.image_path)
        data["mask_path"] = str(self.mask_path)
        return data


def ensure_output_dirs() -> None:
    for path in [
        config.PROCESSED_DIR,
        config.RESULTS_DIR,
        config.REPORTS_DIR,
        config.FIGURES_DIR,
        config.MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def extract_numeric_id(path: str | Path) -> int | None:
    stem = Path(path).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    return int(numbers[-1])


def _image_files(directory: Path, recursive: bool = False) -> list[Path]:
    if not directory.exists():
        return []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        [
            p
            for p in iterator
            if p.is_file()
            and p.suffix.lower() in config.SUPPORTED_IMAGE_EXTENSIONS
            and not p.name.startswith(".")
        ]
    )


def discover_files(directory: Path, recursive: bool = False) -> tuple[dict[int, Path], dict[int, list[str]]]:
    by_id: dict[int, Path] = {}
    duplicates: dict[int, list[str]] = {}
    all_paths: dict[int, list[Path]] = {}

    for path in _image_files(directory, recursive=recursive):
        file_id = extract_numeric_id(path)
        if file_id is None:
            continue
        all_paths.setdefault(file_id, []).append(path)

    for file_id, paths in all_paths.items():
        sorted_paths = sorted(paths)
        by_id[file_id] = sorted_paths[0]
        if len(sorted_paths) > 1:
            duplicates[file_id] = [str(p) for p in sorted_paths]

    return by_id, duplicates


def expected_ids_for_split(split: str) -> list[int]:
    if split == "train":
        return [i for i in config.TRAIN_IDS if i not in config.EXCLUDED_TRAIN_IDS]
    if split == "test":
        return list(config.TEST_IDS)
    raise ValueError(f"Unknown split: {split}")


def normalize_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s_\-]+", " ", text)
    return text


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        config.ID_COLUMN: {"id", "image id", "image", "number", "no"},
        config.TARGET_COLUMN: {"bug type", "bugtype", "type", "label", "class", "classification"},
        config.SPECIES_COLUMN: {"species", "specie", "latin name"},
    }
    normalized_to_original = {normalize_column_name(col): col for col in df.columns}
    rename: dict[object, str] = {}

    for canonical, names in aliases.items():
        found = None
        for candidate in names:
            if candidate in normalized_to_original:
                found = normalized_to_original[candidate]
                break
        if found is None:
            raise ValueError(
                f"Could not find required label column {canonical!r}. "
                f"Available columns: {list(df.columns)}"
            )
        rename[found] = canonical

    result = df.rename(columns=rename)[[config.ID_COLUMN, config.TARGET_COLUMN, config.SPECIES_COLUMN]].copy()
    result[config.ID_COLUMN] = pd.to_numeric(result[config.ID_COLUMN], errors="raise").astype(int)
    result[config.TARGET_COLUMN] = result[config.TARGET_COLUMN].astype(str).str.strip()
    result[config.SPECIES_COLUMN] = result[config.SPECIES_COLUMN].astype(str).str.strip()
    return result


def load_labels() -> pd.DataFrame:
    if not config.LABEL_FILE.exists():
        raise FileNotFoundError(f"Training label file not found: {config.LABEL_FILE}")
    labels = pd.read_excel(config.LABEL_FILE)
    labels = normalize_label_columns(labels)
    duplicated = labels[labels[config.ID_COLUMN].duplicated()][config.ID_COLUMN].tolist()
    if duplicated:
        raise ValueError(f"Duplicate IDs in label file: {duplicated}")
    return labels


def _split_dirs(split: str) -> tuple[Path, Path]:
    if split == "train":
        return config.TRAIN_DIR, config.TRAIN_MASK_DIR
    if split == "test":
        return config.TEST_DIR, config.TEST_MASK_DIR
    raise ValueError(f"Unknown split: {split}")


def _load_mask_for_validation(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        gray = image.convert("L")
        arr = np.asarray(gray)
    return arr > 0


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def _audit_base(split: str) -> dict:
    return {
        "split": split,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(config.PROJECT_ROOT),
        "excluded_train_ids": sorted(config.EXCLUDED_TRAIN_IDS),
        "id_154_policy": (
            "ID 154 is excluded because train/masks/binary_154.tif is officially unavailable. "
            "No artificial mask was created, estimated, drawn, or imputed."
        ),
    }


def validate_split(split: str, write_audit: bool = True, fail_on_error: bool = True) -> dict:
    ensure_output_dirs()
    image_dir, mask_dir = _split_dirs(split)
    expected_ids = expected_ids_for_split(split)
    images_by_id, image_duplicates = discover_files(image_dir, recursive=False)
    masks_by_id, mask_duplicates = discover_files(mask_dir, recursive=False)

    errors: list[str] = []
    warnings: list[str] = []
    excluded = sorted(config.EXCLUDED_TRAIN_IDS) if split == "train" else []

    labels: pd.DataFrame | None = None
    label_ids: set[int] = set()
    if split == "train":
        try:
            labels = load_labels()
            label_ids = set(labels[config.ID_COLUMN].tolist())
        except Exception as exc:
            errors.append(str(exc))

    missing_images = [i for i in expected_ids if i not in images_by_id]
    missing_masks = [i for i in expected_ids if i not in masks_by_id]
    if missing_images:
        errors.append(f"Missing {split} images for IDs: {missing_images}")
    if missing_masks:
        errors.append(f"Missing {split} masks for IDs: {missing_masks}")

    if split == "train" and labels is not None:
        missing_labels = [i for i in expected_ids if i not in label_ids]
        if missing_labels:
            errors.append(f"Missing training labels for IDs: {missing_labels}")
        if 154 in label_ids:
            warnings.append("ID 154 has a label but is excluded because its segmentation mask is unavailable.")

    if image_duplicates:
        errors.append(f"Duplicate image IDs discovered: {image_duplicates}")
    if mask_duplicates:
        errors.append(f"Duplicate mask IDs discovered: {mask_duplicates}")

    checked_ids = [
        i for i in expected_ids if i in images_by_id and i in masks_by_id
    ]
    dimension_mismatches: list[dict] = []
    empty_masks: list[int] = []
    unreadable_files: list[str] = []

    for file_id in checked_ids:
        image_path = images_by_id[file_id]
        mask_path = masks_by_id[file_id]
        try:
            image_size = _image_size(image_path)
            mask_size = _image_size(mask_path)
            if image_size != mask_size:
                dimension_mismatches.append(
                    {
                        "ID": file_id,
                        "image_size": image_size,
                        "mask_size": mask_size,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                    }
                )
            mask = _load_mask_for_validation(mask_path)
            if not bool(mask.any()):
                empty_masks.append(file_id)
        except Exception as exc:
            unreadable_files.append(f"ID {file_id}: {exc}")

    if dimension_mismatches:
        errors.append(f"Image-mask dimension mismatches: {dimension_mismatches}")
    if empty_masks:
        errors.append(f"Empty masks for IDs: {empty_masks}")
    if unreadable_files:
        errors.append(f"Unreadable files: {unreadable_files}")

    audit = _audit_base(split)
    audit.update(
        {
            "status": "passed" if not errors else "failed",
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "expected_ids": expected_ids,
            "expected_count": len(expected_ids),
            "discovered_image_count": len(images_by_id),
            "discovered_mask_count": len(masks_by_id),
            "usable_record_count": len(checked_ids) if not errors else None,
            "excluded_ids": excluded,
            "missing_images": missing_images,
            "missing_masks": missing_masks,
            "duplicate_image_ids": image_duplicates,
            "duplicate_mask_ids": mask_duplicates,
            "dimension_mismatches": dimension_mismatches,
            "empty_masks": empty_masks,
            "warnings": warnings,
            "errors": errors,
        }
    )

    if split == "train" and labels is not None:
        labels_after_exclusion = labels[labels[config.ID_COLUMN].isin(expected_ids)]
        audit["label_file"] = str(config.LABEL_FILE)
        audit["label_count_raw"] = int(len(labels))
        audit["label_count_after_exclusion"] = int(len(labels_after_exclusion))
        audit["bug_type_counts_after_exclusion"] = (
            labels_after_exclusion[config.TARGET_COLUMN].value_counts().sort_index().to_dict()
        )
        audit["species_count_after_exclusion"] = int(labels_after_exclusion[config.SPECIES_COLUMN].nunique())

    if write_audit:
        existing: dict = {}
        if config.DATA_AUDIT_JSON.exists():
            try:
                existing = json.loads(config.DATA_AUDIT_JSON.read_text())
            except json.JSONDecodeError:
                existing = {}
        existing[split] = audit
        existing["latest_split"] = split
        existing["id_154_policy"] = audit["id_154_policy"]
        existing["excluded_train_ids"] = sorted(config.EXCLUDED_TRAIN_IDS)
        config.DATA_AUDIT_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    if errors and fail_on_error:
        raise ValueError(f"{split} validation failed: " + "; ".join(errors))

    return audit


def collect_split_records(split: str, validate: bool = True) -> list[DataRecord]:
    if validate:
        validate_split(split, write_audit=True, fail_on_error=True)

    image_dir, mask_dir = _split_dirs(split)
    images_by_id, _ = discover_files(image_dir, recursive=False)
    masks_by_id, _ = discover_files(mask_dir, recursive=False)
    expected_ids = expected_ids_for_split(split)

    label_lookup: dict[int, tuple[str, str]] = {}
    if split == "train":
        labels = load_labels()
        label_lookup = {
            int(row[config.ID_COLUMN]): (str(row[config.TARGET_COLUMN]), str(row[config.SPECIES_COLUMN]))
            for _, row in labels.iterrows()
        }

    records: list[DataRecord] = []
    for file_id in expected_ids:
        if file_id not in images_by_id or file_id not in masks_by_id:
            raise FileNotFoundError(f"Missing {split} image or mask for ID {file_id}")
        bug_type, species = label_lookup.get(file_id, (None, None))
        records.append(
            DataRecord(
                ID=file_id,
                image_path=images_by_id[file_id],
                mask_path=masks_by_id[file_id],
                bug_type=bug_type,
                species=species,
            )
        )
    return records


def test_data_available() -> bool:
    try:
        validate_split("test", write_audit=False, fail_on_error=True)
    except Exception:
        return False
    return True


def records_to_frame(records: Iterable[DataRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate project data layout.")
    parser.add_argument("--validate", action="store_true", help="Run validation.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    args = parser.parse_args()

    if args.validate:
        audit = validate_split(args.split, write_audit=True, fail_on_error=True)
        print(json.dumps({"split": args.split, "status": audit["status"]}, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

