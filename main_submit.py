import polars as pl

from src import config as cfg
from src.candidates import build_valid_candidates
from src.data_loader import load_data
from src.evaluate import (
    predict_topk_from_parquet,
    save_submission_pickle,
    topk_to_submission_dict,
)
from src.features import build_features_chunked
from src.logging_utils import log_step, log_time
from src.preprocess import prepare_inference_matrix
from src.splits import make_time_splits


def main():
    log_step("submission pipeline started")
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    with log_time("build final candidates"):
        final_candidates_lf = build_valid_candidates(
            splits["final_hist_lf"],
            items_lf=items_lf,
            min_bills=cfg.MIN_BILLS_ACTIVE_USER,
            recent_top_k=cfg.RECENT_TOP_K_VALID,
            frequent_top_k=cfg.FREQUENT_TOP_K_VALID,
            category_col=cfg.CATEGORY_CANDIDATE_COL,
            user_top_categories=cfg.USER_TOP_CATEGORIES,
            category_items_per_category=cfg.CATEGORY_ITEMS_PER_CATEGORY,
            co_anchor_top_k=cfg.COOCCURRENCE_ANCHOR_TOP_K,
            co_top_k=cfg.COOCCURRENCE_TOP_K,
            co_max_bill_items=cfg.COOCCURRENCE_MAX_BILL_ITEMS,
        )
        final_candidates_lf.sink_parquet(cfg.FINAL_CANDIDATES_PATH)
        final_candidates_lf = pl.scan_parquet(cfg.FINAL_CANDIDATES_PATH)

    with log_time("build final features"):
        final_features_lf = build_features_chunked(
            splits["final_hist_lf"],
            final_candidates_lf,
            items_lf,
            cfg.FINAL_FEATURES_CHUNKS_DIR,
            n_chunks=cfg.FEATURE_BUILD_CHUNKS,
        )

    with log_time("prepare final matrix"):
        train_features_lf = pl.scan_parquet(str(cfg.TRAIN_FEATURES_CHUNKS_DIR / "*.parquet"))
        final_model_lf, _ = prepare_inference_matrix(
            final_features_lf,
            train_features_lf,
            cfg.DROP_COLS,
            cfg.CAT_COLS,
        )
        final_model_lf.sink_parquet(cfg.FINAL_MODEL_READY_PATH)

    with log_time("predict final top-k"):
        top10 = predict_topk_from_parquet(
            cfg.FINAL_MODEL_READY_PATH,
            cfg.MODEL_PATH,
            feature_columns_path=cfg.FEATURE_COLUMNS_PATH,
            id_cols=["customer_id", "item_id"],
            k=10,
            batch_size=cfg.PREDICT_BATCH_SIZE,
        )

    with log_time("prepare fallback items"):
        user_ids = (
            final_candidates_lf
            .select("customer_id")
            .unique()
            .collect()
            .get_column("customer_id")
            .to_list()
        )
        fallback_items = (
            splits["final_hist_lf"]
            .group_by("item_id")
            .agg([
                pl.col("customer_id").n_unique().alias("n_customers"),
                pl.len().alias("n_transactions"),
            ])
            .sort(["n_customers", "n_transactions"], descending=[True, True])
            .head(50)
            .select("item_id")
            .collect()
            .get_column("item_id")
            .to_list()
        )

    with log_time("save final submission pickle"):
        submission_dict = topk_to_submission_dict(
            top10,
            k=10,
            user_ids=user_ids,
            fallback_items=fallback_items,
        )
        save_submission_pickle(submission_dict, cfg.SUBMISSION_PATH)
    log_step("submission pipeline finished")


if __name__ == "__main__":
    main()
