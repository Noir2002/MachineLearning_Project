import numpy as np

from src import config
from src.features import extract_features_from_arrays, mandatory_columns_present, mask_to_bool, shape_symmetry


def test_mask_binarization_rgb():
    mask = np.zeros((3, 3, 3), dtype=np.uint8)
    mask[1, 1] = [0, 10, 0]
    result = mask_to_bool(mask)
    assert result.dtype == bool
    assert result.sum() == 1


def test_rgb_statistics_and_bug_pixel_ratio():
    rgb = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    mask = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    features = extract_features_from_arrays(rgb, mask)
    assert features["bug_pixel_ratio"] == 0.5
    assert features["r_min"] == 10
    assert features["r_max"] == 70
    assert features["r_mean"] == 40
    assert features["g_median"] == 50
    assert features["b_mean"] == 60


def test_shape_symmetry_on_symmetric_mask():
    mask = np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert shape_symmetry(mask) > 0.99


def test_mandatory_feature_columns_present():
    rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.uint8)
    features = extract_features_from_arrays(rgb, mask)
    assert mandatory_columns_present(set(features))
    for column in config.MANDATORY_FEATURE_COLUMNS:
        assert column in features

