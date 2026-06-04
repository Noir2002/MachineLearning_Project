from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, filters, measure, morphology, transform

from . import config


def mask_to_bool(mask: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D after conversion, got shape {arr.shape}")
    return arr > 0


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_bool_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return mask_to_bool(image)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute bounding box for an empty mask.")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return int(y_min), int(x_min), int(y_max), int(x_max)


def _downscale_pair_for_shape(rgb_crop: np.ndarray, mask_crop: np.ndarray, max_side: int = 768) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask_crop.shape
    longest = max(height, width)
    if longest <= max_side:
        return rgb_crop, mask_crop
    scale = max_side / float(longest)
    new_shape = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    small_rgb = transform.resize(
        rgb_crop,
        (*new_shape, 3),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)
    small_mask = transform.resize(
        mask_crop.astype(float),
        new_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ) > 0.5
    return small_rgb, small_mask


def shape_symmetry(mask: np.ndarray) -> float:
    mask = mask_to_bool(mask)
    if not mask.any():
        return np.nan
    y_min, x_min, y_max, x_max = _bbox(mask)
    crop = mask[y_min : y_max + 1, x_min : x_max + 1]
    flipped = np.fliplr(crop)
    denom = crop.sum() + flipped.sum()
    if denom == 0:
        return np.nan
    return float(2.0 * np.logical_and(crop, flipped).sum() / denom)


def color_symmetry(rgb: np.ndarray, mask: np.ndarray) -> float:
    mask = mask_to_bool(mask)
    if not mask.any():
        return np.nan
    y_min, x_min, y_max, x_max = _bbox(mask)
    rgb_crop = np.asarray(rgb)[y_min : y_max + 1, x_min : x_max + 1, :3]
    mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1]
    rgb_crop, mask_crop = _downscale_pair_for_shape(rgb_crop, mask_crop)
    flipped_rgb = np.fliplr(rgb_crop)
    flipped_mask = np.fliplr(mask_crop)
    valid = mask_crop & flipped_mask
    if not valid.any():
        return np.nan
    diffs = np.abs(rgb_crop.astype(float) - flipped_rgb.astype(float))[valid]
    return float(max(0.0, 1.0 - (diffs.mean() / 255.0)))


def _channel_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _hu_moments(mask: np.ndarray) -> dict[str, float]:
    try:
        _, small_mask = _downscale_pair_for_shape(
            np.repeat(mask[..., None].astype(np.uint8) * 255, 3, axis=2),
            mask,
            max_side=512,
        )
        image = small_mask.astype(float)
        central = measure.moments_central(image)
        normalized = measure.moments_normalized(central)
        hu = measure.moments_hu(normalized)
    except Exception:
        hu = np.full(7, np.nan)
    return {f"hu_moment_{idx + 1}": float(value) for idx, value in enumerate(hu)}


def _largest_region(mask: np.ndarray):
    labels = measure.label(mask)
    regions = measure.regionprops(labels)
    if not regions:
        return None, labels
    return max(regions, key=lambda region: region.area), labels


def _color_space_stats(rgb_pixels: np.ndarray, rgb_crop: np.ndarray, mask_crop: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}

    if len(rgb_pixels) > 500_000:
        indices = np.linspace(0, len(rgb_pixels) - 1, 500_000).astype(int)
        color_pixels = rgb_pixels[indices]
    else:
        color_pixels = rgb_pixels

    rgb_pixels_float = color_pixels.astype(np.float32) / 255.0
    hsv_pixels = color.rgb2hsv(rgb_pixels_float.reshape(-1, 1, 3)).reshape(-1, 3)
    for idx, name in enumerate(["h", "s", "v"]):
        channel = hsv_pixels[:, idx]
        features[f"hsv_{name}_mean"] = float(np.mean(channel))
        features[f"hsv_{name}_median"] = float(np.median(channel))
        features[f"hsv_{name}_std"] = float(np.std(channel, ddof=0))

    lab_pixels = color.rgb2lab(rgb_pixels_float.reshape(-1, 1, 3)).reshape(-1, 3)
    for idx, name in enumerate(["l", "a", "b"]):
        channel = lab_pixels[:, idx]
        features[f"lab_{name}_mean"] = float(np.mean(channel))
        features[f"lab_{name}_std"] = float(np.std(channel, ddof=0))

    gray_pixels = color.rgb2gray(rgb_pixels_float.reshape(-1, 1, 3)).reshape(-1)
    features["gray_mean"] = float(np.mean(gray_pixels))
    features["gray_std"] = float(np.std(gray_pixels, ddof=0))
    features["gray_contrast"] = float(np.percentile(gray_pixels, 95) - np.percentile(gray_pixels, 5))

    small_rgb, small_mask = _downscale_pair_for_shape(rgb_crop, mask_crop, max_side=768)
    small_gray = color.rgb2gray(small_rgb.astype(np.float32) / 255.0)
    sobel = filters.sobel(small_gray)
    inside = sobel[small_mask]
    if inside.size == 0:
        features["edge_density"] = np.nan
    else:
        threshold = float(np.mean(inside) + np.std(inside))
        features["edge_density"] = float(np.mean(inside > threshold))

    return features


def extract_features_from_arrays(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    allow_mask_resize: bool = False,
) -> dict[str, float]:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB image must have shape (height, width, 3), got {rgb.shape}")
    rgb = rgb[..., :3].astype(np.uint8, copy=False)
    bool_mask = mask_to_bool(mask)

    if rgb.shape[:2] != bool_mask.shape:
        if allow_mask_resize:
            bool_mask = _resize_mask(bool_mask, rgb.shape[:2])
        else:
            raise ValueError(f"Image and mask dimensions differ: image={rgb.shape[:2]}, mask={bool_mask.shape}")

    if not bool_mask.any():
        raise ValueError("Mask contains no foreground pixels.")

    height, width = bool_mask.shape
    area = int(bool_mask.sum())
    y_min, x_min, y_max, x_max = _bbox(bool_mask)
    bbox_height = y_max - y_min + 1
    bbox_width = x_max - x_min + 1
    bbox_area = bbox_height * bbox_width
    rgb_pixels = rgb[bool_mask]
    rgb_crop = rgb[y_min : y_max + 1, x_min : x_max + 1]
    mask_crop = bool_mask[y_min : y_max + 1, x_min : x_max + 1]

    largest_region, labels = _largest_region(bool_mask)
    component_count = int(labels.max())
    perimeter = float(measure.perimeter(bool_mask, neighborhood=8))
    circularity = _safe_ratio(4.0 * math.pi * area, perimeter * perimeter) if perimeter > 0 else np.nan

    features: dict[str, float] = {
        "color_symmetry": color_symmetry(rgb, bool_mask),
        "shape_symmetry": shape_symmetry(bool_mask),
        "bug_pixel_ratio": _safe_ratio(area, height * width),
        "bbox_width": float(bbox_width),
        "bbox_height": float(bbox_height),
        "aspect_ratio": _safe_ratio(bbox_width, bbox_height),
        "area": float(area),
        "perimeter": perimeter,
        "circularity": circularity,
        "extent": _safe_ratio(area, bbox_area),
        "solidity": float(largest_region.solidity) if largest_region is not None else np.nan,
        "orientation": float(largest_region.orientation) if largest_region is not None else np.nan,
        "eccentricity": float(largest_region.eccentricity) if largest_region is not None else np.nan,
        "major_axis_length": float(largest_region.major_axis_length) if largest_region is not None else np.nan,
        "minor_axis_length": float(largest_region.minor_axis_length) if largest_region is not None else np.nan,
        "equivalent_diameter": float(largest_region.equivalent_diameter) if largest_region is not None else np.nan,
        "component_count": float(component_count),
        "foreground_bbox_fill_ratio": _safe_ratio(area, bbox_area),
        "convex_area": float(largest_region.convex_area) if largest_region is not None else np.nan,
        "bbox_area": float(bbox_area),
    }

    for idx, prefix in enumerate(["r", "g", "b"]):
        features.update(_channel_stats(prefix, rgb_pixels[:, idx]))

    features.update(_hu_moments(mask_crop))
    features.update(_color_space_stats(rgb_pixels, rgb_crop, mask_crop))

    for key, value in list(features.items()):
        if isinstance(value, (float, int)) and not np.isfinite(value):
            features[key] = np.nan

    return features


def extract_features(image_path: Path, mask_path: Path, *, allow_mask_resize: bool = False) -> dict[str, float]:
    rgb = load_rgb_image(image_path)
    mask = load_bool_mask(mask_path)
    return extract_features_from_arrays(rgb, mask, allow_mask_resize=allow_mask_resize)


def mandatory_columns_present(columns: list[str] | set[str]) -> bool:
    column_set = set(columns)
    return all(column in column_set for column in config.MANDATORY_FEATURE_COLUMNS)
