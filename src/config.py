from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = PROJECT_ROOT / "train"
TRAIN_MASK_DIR = TRAIN_DIR / "masks"
TEST_DIR = PROJECT_ROOT / "test"
TEST_MASK_DIR = TEST_DIR / "masks"
LABEL_FILE = TRAIN_DIR / "classif.xlsx"

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TRAIN_FEATURES_CSV = PROCESSED_DIR / "train_features.csv"
TEST_FEATURES_CSV = PROCESSED_DIR / "test_features.csv"
TRAIN_FEATURES_METADATA_JSON = PROCESSED_DIR / "train_features_metadata.json"
TEST_FEATURES_METADATA_JSON = PROCESSED_DIR / "test_features_metadata.json"

DATA_AUDIT_JSON = RESULTS_DIR / "data_audit.json"
MODEL_COMPARISON_CSV = RESULTS_DIR / "model_comparison.csv"
CLASSIFICATION_REPORTS_JSON = RESULTS_DIR / "classification_reports.json"
BEST_MODEL_INFO_JSON = RESULTS_DIR / "best_model_info.json"
CLUSTERING_METRICS_CSV = RESULTS_DIR / "clustering_metrics.csv"
VISUALIZATION_AUDIT_JSON = RESULTS_DIR / "visualization_audit.json"
SUBMISSION_CSV = RESULTS_DIR / "submission.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"

TRAIN_IDS = tuple(range(1, 251))
TEST_IDS = tuple(range(251, 348))
EXCLUDED_TRAIN_IDS = {154}
RANDOM_STATE = 42

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
TARGET_COLUMN = "bug type"
SPECIES_COLUMN = "species"
ID_COLUMN = "ID"

MANDATORY_FEATURE_COLUMNS = [
    "color_symmetry",
    "shape_symmetry",
    "bug_pixel_ratio",
    "r_min",
    "r_max",
    "r_mean",
    "r_median",
    "r_std",
    "g_min",
    "g_max",
    "g_mean",
    "g_median",
    "g_std",
    "b_min",
    "b_max",
    "b_mean",
    "b_median",
    "b_std",
]

