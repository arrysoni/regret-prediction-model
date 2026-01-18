from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

TRANSACTIONS_CSV = DATA_RAW_DIR / "transactions.csv"
LABELS_CSV = DATA_RAW_DIR / "labels.csv"

FEATURE_DATASET_PATH = DATA_PROCESSED_DIR / "feature_dataset.parquet"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
TRAIN_METADATA_PATH = MODELS_DIR / "train_metadata.joblib"

RANDOM_SEED = 42
EPS = 1e-9
