import polars as pl

from src import config as cfg
from src.candidates import build_valid_candidates
from src.data_loader import load_data
from src.evaluate import (
    get_topk,
    predict_matrix,
    save_submission_pickle,
    topk_to_submission_dict,
)
from src.features import build_features
from src.preprocess import prepare_inference_matrix
from src.splits import make_time_splits


def main():
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transactions_lf, items_lf = load_data(cfg.TRANSACTIONS_PATH, cfg.ITEMS_PATH)
    splits = make_time_splits(
        transactions_lf,
        train_history_months=cfg.TRAIN_HISTORY_MONTHS,
        train_label_month=cfg.TRAIN_LABEL_MONTH,
        valid_history_months=cfg.VALID_HISTORY_MONTHS,
        valid_label_month=cfg.VALID_LABEL_MONTH,
        final_history_months=cfg.FINAL_HISTORY_MONTHS,
    )

    final_candidates_lf = build_valid_candidates(
        splits["final_hist_lf"],
        min_bills=cfg.MIN_BILLS_ACTIVE_USER,
        recent_top_k=cfg.RECENT_TOP_K_VALID,
        frequent_top_k=cfg.FREQUENT_TOP_K_VALID,
    )
    final_candidates_lf.sink_parquet(cfg.FINAL_CANDIDATES_PATH)
    final_candidates_lf = pl.scan_parquet(cfg.FINAL_CANDIDATES_PATH)

    final_features_lf = build_features(splits["final_hist_lf"], final_candidates_lf, items_lf)
    final_features_lf.sink_parquet(cfg.FINAL_FEATURES_PATH)
    final_features_lf = pl.scan_parquet(cfg.FINAL_FEATURES_PATH)

    train_features_lf = pl.scan_parquet(cfg.TRAIN_FEATURES_PATH)
    final_model_lf, _ = prepare_inference_matrix(
        final_features_lf,
        train_features_lf,
        cfg.DROP_COLS,
        cfg.CAT_COLS,
    )
    final_model_lf.sink_parquet(cfg.FINAL_MODEL_READY_PATH)

    final_df = predict_matrix(
        cfg.FINAL_MODEL_READY_PATH,
        cfg.MODEL_PATH,
        feature_columns_path=cfg.FEATURE_COLUMNS_PATH,
        id_cols=["customer_id", "item_id"],
    )
    top10 = get_topk(final_df, k=10)

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

    submission_dict = topk_to_submission_dict(
        top10,
        k=10,
        user_ids=user_ids,
        fallback_items=fallback_items,
    )
    save_submission_pickle(submission_dict, cfg.SUBMISSION_PATH)


if __name__ == "__main__":
    main()
