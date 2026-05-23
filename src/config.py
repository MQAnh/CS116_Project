from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

TRANSACTIONS_PATH = RAW_DIR / "transaction_full_2025_final.parquet"
ITEMS_PATH = RAW_DIR / "items.parquet"

TRAIN_CANDIDATES_PATH = PROCESSED_DIR / "train_candidates.parquet"
TRAIN_DATASET_LABELS_PATH = PROCESSED_DIR / "train_dataset_labels.parquet"
TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
TRAIN_MODEL_READY_PATH = PROCESSED_DIR / "train_model_ready.parquet"

VALID_CANDIDATES_PATH = PROCESSED_DIR / "valid_candidates.parquet"
VALID_GT_PATH = PROCESSED_DIR / "valid_gt.parquet"
VALID_DATASET_LABELS_PATH = PROCESSED_DIR / "valid_dataset_labels.parquet"
VALID_FEATURES_PATH = PROCESSED_DIR / "valid_features.parquet"
VALID_MODEL_READY_PATH = PROCESSED_DIR / "valid_model_ready.parquet"
FINAL_CANDIDATES_PATH = PROCESSED_DIR / "final_candidates.parquet"
FINAL_FEATURES_PATH = PROCESSED_DIR / "final_features.parquet"
FINAL_MODEL_READY_PATH = PROCESSED_DIR / "final_model_ready.parquet"

MODEL_PATH = MODEL_DIR / "lgbm_baseline.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.pkl"

TRAIN_HISTORY_MONTHS = (1, 10)
TRAIN_LABEL_MONTH = 11
VALID_HISTORY_MONTHS = (1, 11)
VALID_LABEL_MONTH = 12  # giữ giống notebook gốc
FINAL_HISTORY_MONTHS = (1, 12)

MIN_BILLS_ACTIVE_USER = 2
RECENT_TOP_K_TRAIN = 20
FREQUENT_TOP_K_TRAIN = 20
RECENT_TOP_K_VALID = 30
FREQUENT_TOP_K_VALID = 20
POPULAR_TOP_K = 50
CATEGORY_CANDIDATE_COL = "category_l2"
USER_TOP_CATEGORIES = 3
CATEGORY_ITEMS_PER_CATEGORY = 20
COOCCURRENCE_ANCHOR_TOP_K = 20
COOCCURRENCE_TOP_K = 10
COOCCURRENCE_MAX_BILL_ITEMS = 30
PREDICT_BATCH_SIZE = 500_000

DROP_COLS = [
    "customer_id",
    "item_id",
    "target",
    "ui_last_date",
    "ui_first_date",
    "description",
]

CAT_COLS = [
    "category_l1",
    "category_l2",
    "category_l3",
    "category",
    "brand",
    "manufacturer",
    "size",
]
