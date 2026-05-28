import polars as pl

from src import config as cfg
from src.cleanup import cleanup_paths
from src.data_loader import load_data
from src.splits import make_time_splits
from src.candidates import build_train_candidates
from src.labels import make_ground_truth, make_labeled_dataset
from src.features import build_features_chunked
from src.logging_utils import log_step, log_time
from src.preprocess import prepare_train_matrix_chunked
from src.train import train_lgbm


def main():
    log_step("train pipeline started")
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with log_time("clean stale train intermediates"):
        cleanup_paths([
            cfg.TRAIN_FEATURES_CHUNKS_DIR,
            cfg.TRAIN_MODEL_READY_CHUNKS_DIR,
            cfg.TRAIN_FEATURES_PATH,
            cfg.TRAIN_MODEL_READY_PATH,
        ])

    with log_time("load data and make time splits"):
        transactions_lf, items_lf = load_data(cfg.TRANSACTIONS_PATH, cfg.ITEMS_PATH)
        splits = make_time_splits(
            transactions_lf,
            train_history_months=cfg.TRAIN_HISTORY_MONTHS,
            train_label_month=cfg.TRAIN_LABEL_MONTH,
            valid_history_months=cfg.VALID_HISTORY_MONTHS,
            valid_label_month=cfg.VALID_LABEL_MONTH,
            final_history_months=cfg.FINAL_HISTORY_MONTHS,
        )

    with log_time("build train candidates"):
        train_candidates_lf = build_train_candidates(
            splits["train_hist_lf"],
            items_lf=items_lf,
            min_bills=cfg.MIN_BILLS_ACTIVE_USER,
            recent_top_k=cfg.RECENT_TOP_K_TRAIN,
            frequent_top_k=cfg.FREQUENT_TOP_K_TRAIN,
            popular_top_k=cfg.POPULAR_CANDIDATE_TOP_K,
            category_col=cfg.CATEGORY_CANDIDATE_COL,
            user_top_categories=cfg.USER_TOP_CATEGORIES,
            category_items_per_category=cfg.CATEGORY_ITEMS_PER_CATEGORY,
            co_anchor_top_k=cfg.COOCCURRENCE_ANCHOR_TOP_K,
            co_top_k=cfg.COOCCURRENCE_TOP_K,
            co_max_bill_items=cfg.COOCCURRENCE_MAX_BILL_ITEMS,
        )
        train_candidates_lf.sink_parquet(cfg.TRAIN_CANDIDATES_PATH)
        train_candidates_lf = pl.scan_parquet(cfg.TRAIN_CANDIDATES_PATH)

    with log_time("create train labels"):
        train_gt_lf = make_ground_truth(splits["train_label_lf"])
        train_dataset_lf = make_labeled_dataset(train_candidates_lf, train_gt_lf)
        train_dataset_lf.sink_parquet(cfg.TRAIN_DATASET_LABELS_PATH)
        train_dataset_lf = pl.scan_parquet(cfg.TRAIN_DATASET_LABELS_PATH)

    with log_time("build train features"):
        train_features_lf = build_features_chunked(
            splits["train_hist_lf"],
            train_dataset_lf,
            items_lf,
            cfg.TRAIN_FEATURES_CHUNKS_DIR,
            n_chunks=cfg.FEATURE_BUILD_CHUNKS,
        )

    with log_time("prepare train matrix"):
        train_model_lf, feature_cols = prepare_train_matrix_chunked(
            cfg.TRAIN_FEATURES_CHUNKS_DIR,
            cfg.TRAIN_MODEL_READY_CHUNKS_DIR,
            cfg.DROP_COLS,
            cfg.CAT_COLS,
            metadata_path=cfg.PREPROCESS_METADATA_PATH,
            selected_features=cfg.SELECTED_FEATURES if cfg.FEATURE_SELECTION_ENABLED else None,
        )

    with log_time("train LightGBM"):
        train_lgbm(
            cfg.TRAIN_MODEL_READY_CHUNKS_DIR,
            cfg.MODEL_PATH,
            feature_columns_path=cfg.FEATURE_COLUMNS_PATH,
            importance_path=cfg.IMPORTANCE_PATH,
            train_batch_size=cfg.TRAIN_READ_BATCH_SIZE,
            negative_ratio=cfg.TRAIN_NEGATIVE_RATIO,
            positive_fraction=cfg.TRAIN_POSITIVE_FRACTION,
            max_train_rows=cfg.TRAIN_MAX_ROWS,
            categorical_features=cfg.CAT_COLS if cfg.LIGHTGBM_CATEGORICAL_FEATURES_ENABLED else None,
        )
    if cfg.CLEAN_INTERMEDIATE_AFTER_TRAIN:
        with log_time("clean train intermediates"):
            cleanup_paths([
                cfg.TRAIN_FEATURES_CHUNKS_DIR,
                cfg.TRAIN_MODEL_READY_CHUNKS_DIR,
                cfg.TRAIN_CANDIDATES_PATH,
                cfg.TRAIN_DATASET_LABELS_PATH,
                cfg.TRAIN_FEATURES_PATH,
                cfg.TRAIN_MODEL_READY_PATH,
            ])
    log_step("train pipeline finished")


if __name__ == "__main__":
    main()
