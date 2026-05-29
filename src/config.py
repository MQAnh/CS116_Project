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
TRAIN_FEATURES_CHUNKS_DIR = PROCESSED_DIR / "train_features_chunks"
TRAIN_FEATURE_SOURCES_DIR = PROCESSED_DIR / "train_feature_sources"
TRAIN_MODEL_READY_PATH = PROCESSED_DIR / "train_model_ready.parquet"
TRAIN_MODEL_READY_CHUNKS_DIR = PROCESSED_DIR / "train_model_ready_chunks"

VALID_CANDIDATES_PATH = PROCESSED_DIR / "valid_candidates.parquet"
VALID_GT_PATH = PROCESSED_DIR / "valid_gt.parquet"
VALID_DATASET_LABELS_PATH = PROCESSED_DIR / "valid_dataset_labels.parquet"
VALID_FEATURES_PATH = PROCESSED_DIR / "valid_features.parquet"
VALID_FEATURES_CHUNKS_DIR = PROCESSED_DIR / "valid_features_chunks"
VALID_FEATURE_SOURCES_DIR = PROCESSED_DIR / "valid_feature_sources"
VALID_MODEL_READY_PATH = PROCESSED_DIR / "valid_model_ready.parquet"
VALID_MODEL_READY_CHUNKS_DIR = PROCESSED_DIR / "valid_model_ready_chunks"
FINAL_CANDIDATES_PATH = PROCESSED_DIR / "final_candidates.parquet"
FINAL_FEATURES_PATH = PROCESSED_DIR / "final_features.parquet"
FINAL_FEATURES_CHUNKS_DIR = PROCESSED_DIR / "final_features_chunks"
FINAL_FEATURE_SOURCES_DIR = PROCESSED_DIR / "final_feature_sources"
FINAL_MODEL_READY_PATH = PROCESSED_DIR / "final_model_ready.parquet"
FINAL_MODEL_READY_CHUNKS_DIR = PROCESSED_DIR / "final_model_ready_chunks"

MODEL_PATH = MODEL_DIR / "lgbm_baseline.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
PREPROCESS_METADATA_PATH = MODEL_DIR / "preprocess_metadata.pkl"
IMPORTANCE_PATH = OUTPUT_DIR / "feature_importance.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission.pkl"

TRAIN_HISTORY_MONTHS = (1, 10)
TRAIN_LABEL_MONTH = 11
VALID_HISTORY_MONTHS = (1, 11)
VALID_LABEL_MONTH = 12  # giữ giống notebook gốc
FINAL_HISTORY_MONTHS = (1, 12)

MIN_BILLS_ACTIVE_USER = 2
RECENT_TOP_K_TRAIN = 20
FREQUENT_TOP_K_TRAIN = 10
RECENT_TOP_K_VALID = 10
FREQUENT_TOP_K_VALID = 10
POPULAR_TOP_K = 0
POPULAR_CANDIDATE_TOP_K = 0
FALLBACK_ITEM_TOP_K = 0
FALLBACK_ITEM_SKIP_TOP_K = 0
ROTATE_FALLBACK_ITEMS = False
CATEGORY_CANDIDATE_COL = "category_l2"
USER_TOP_CATEGORIES = 0
CATEGORY_ITEMS_PER_CATEGORY = 0
USER_TOP_LOCATIONS = 1
LOCATION_ITEMS_PER_LOCATION = 3
COOCCURRENCE_ANCHOR_TOP_K = 10
COOCCURRENCE_TOP_K = 5
COOCCURRENCE_MAX_BILL_ITEMS = 15
FINAL_COOCCURRENCE_ENABLED = True
FINAL_COOCCURRENCE_HISTORY_MONTHS = None
FINAL_COOCCURRENCE_ANCHOR_TOP_K = COOCCURRENCE_ANCHOR_TOP_K
FINAL_COOCCURRENCE_TOP_K = COOCCURRENCE_TOP_K
FINAL_COOCCURRENCE_MAX_BILL_ITEMS = COOCCURRENCE_MAX_BILL_ITEMS
PREDICT_BATCH_SIZE = 5_000_000
FEATURE_BUILD_CHUNKS = 16
TRAIN_READ_BATCH_SIZE = 500_000
TRAIN_NEGATIVE_RATIO = 3.0
TRAIN_POSITIVE_FRACTION = 1.0
TRAIN_MAX_ROWS = 3_000_000
POPULAR_NEGATIVE_WEIGHT_COLUMN = None
POPULAR_NEGATIVE_WEIGHT_ALPHA = 0.0
RECENT_FALLBACK_FOR_ALL_USERS = False
CLEAN_INTERMEDIATE_AFTER_TRAIN = True
RERANK_REPEAT_BOOST = 0.02
RERANK_AFFINITY_BOOST = 0.015
RERANK_POPULARITY_PENALTY = 0.0
FEATURE_SELECTION_ENABLED = True
LIGHTGBM_CATEGORICAL_FEATURES_ENABLED = False
DROP_POPULAR_SIGNAL_FEATURES = True

# Selected from feature_importance_7.7_points.csv with importance >= 150,
# plus stable high-importance features from feature_importance_7.98_points.csv.
SELECTED_FEATURES = [
    "item_avg_bill_items",
    "ui_recency_days",
    "category_l1",
    "item_avg_bill_value",
    "ui_repeat_due_ratio",
    "item_transaction_share_30d",
    "item_discount_positive_rate",
    "category_l2",
    "item_avg_price",
    "item_avg_discount",
    "item_n_customers",
    "item_avg_discount_rate",
    "item_n_transactions_30d",
    "ub_recency_days",
    "brand",
    "category_l3",
    "ul3_recency_days",
    "ul1_recency_days",
    "category",
    "item_total_quantity",
    "item_catalog_price",
    "ui_avg_repeat_interval_days",
    "item_n_customers_30d",
    "user_days_since_last_bill",
    "item_transaction_share_60d",
    "ul2_recency_days",
    "item_n_transactions",
    "manufacturer",
    "user_n_active_days",
    "ui_total_quantity",
    "item_n_customers_60d",
    "ui_purchase_span_days",
    "user_avg_discount",
    "uman_transaction_share",
    "uman_recency_days",
    "user_price_p25",
    "is_cooccurrence_candidate",
    "item_days_since_first_seen",
    "user_price_p75",
    "ui_n_transactions",
    "user_median_price",
    "item_n_transactions_60d",
    "user_n_unique_items",
    "item_user_discount_rate_diff",
    "is_location_candidate",
    "location_item_rank",
    "user_location_rank",
    "item_location_transactions",
    "item_location_customers",
    "ui_n_transactions_60d",
    "ul1_n_transactions_30d",
    "user_discount_positive_rate",
    "ul3_n_bills",
    "user_main_location",
    "ul1_n_bills",
    "item_price_to_user_avg",
    "ui_n_bills",
    "ul1_transaction_share",
    "ub_transaction_share",
    "user_avg_bill_value",
    "item_main_location_transaction_share",
    "ub_n_transactions",
    "ul3_transaction_share",
    "user_n_transactions_60d",
    "user_avg_discount_rate",
    "user_total_spend",
    "user_avg_quantity_per_bill",
    "user_total_quantity",
    "ul3_n_transactions_30d",
    "ul2_total_quantity",
    "user_n_bills_30d",
    "ul3_n_transactions",
    "user_avg_price",
    "user_avg_items_per_bill",
    "uc_transaction_share",
    "sale_status",
    "uc_n_bills",
    "uc_total_quantity",
    "item_n_bills",
    "ub_n_bills",
    "uc_n_transactions",
    "ub_total_quantity",
    "user_n_transactions",
]

POPULAR_SIGNAL_FEATURES = {
    "item_n_transactions",
    "item_n_customers",
    "item_n_bills",
    "item_total_quantity",
    "item_n_customers_30d",
    "item_n_customers_60d",
    "item_n_transactions_30d",
    "item_n_transactions_60d",
    "item_transaction_share_30d",
    "item_transaction_share_60d",
    "item_days_since_first_seen",
    "item_main_location_transaction_share",
}

if DROP_POPULAR_SIGNAL_FEATURES:
    SELECTED_FEATURES = [
        feature
        for feature in SELECTED_FEATURES
        if feature not in POPULAR_SIGNAL_FEATURES
    ]

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
