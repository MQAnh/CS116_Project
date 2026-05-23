import polars as pl
from src import config as cfg
from src.data_loader import load_data
from src.splits import make_time_splits
from src.candidates import build_valid_candidates
from src.labels import make_ground_truth, make_labeled_dataset
from src.features import build_features
from src.preprocess import prepare_valid_matrix
from src.evaluate import (
    predict_valid,
    get_topk,
    ground_truth_to_dict,
    server_precision_at_k,
    topk_to_submission_dict,
    save_submission_pickle,
)


def main():
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    transactions_lf, items_lf = load_data(cfg.TRANSACTIONS_PATH, cfg.ITEMS_PATH)
    splits = make_time_splits(
        transactions_lf,
        train_history_months=cfg.TRAIN_HISTORY_MONTHS,
        train_label_month=cfg.TRAIN_LABEL_MONTH,
        valid_history_months=cfg.VALID_HISTORY_MONTHS,
        valid_label_month=cfg.VALID_LABEL_MONTH,
        final_history_months=cfg.FINAL_HISTORY_MONTHS,
    )

    valid_candidates_lf = build_valid_candidates(
        splits["valid_hist_lf"],
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
    valid_candidates_lf.sink_parquet(cfg.VALID_CANDIDATES_PATH)
    valid_candidates_lf = pl.scan_parquet(cfg.VALID_CANDIDATES_PATH)

    valid_gt_lf = make_ground_truth(splits["valid_label_lf"])
    valid_gt_lf.sink_parquet(cfg.VALID_GT_PATH)

    valid_dataset_lf = make_labeled_dataset(valid_candidates_lf, valid_gt_lf)
    valid_dataset_lf.sink_parquet(cfg.VALID_DATASET_LABELS_PATH)
    valid_dataset_lf = pl.scan_parquet(cfg.VALID_DATASET_LABELS_PATH)

    valid_features_lf = build_features(splits["valid_hist_lf"], valid_dataset_lf, items_lf)
    valid_features_lf.sink_parquet(cfg.VALID_FEATURES_PATH)
    valid_features_lf = pl.scan_parquet(cfg.VALID_FEATURES_PATH)

    train_features_lf = pl.scan_parquet(cfg.TRAIN_FEATURES_PATH)
    valid_model_lf, feature_cols = prepare_valid_matrix(
        valid_features_lf,
        train_features_lf,
        cfg.DROP_COLS,
        cfg.CAT_COLS,
    )
    valid_model_lf.sink_parquet(cfg.VALID_MODEL_READY_PATH)

    valid_df = predict_valid(
        cfg.VALID_MODEL_READY_PATH,
        cfg.MODEL_PATH,
        feature_columns_path=cfg.FEATURE_COLUMNS_PATH,
    )
    top10 = get_topk(valid_df, k=10)

    submission_dict = topk_to_submission_dict(top10)
    answer_dict = ground_truth_to_dict(splits["valid_label_lf"])
    print("Server Precision@10:", server_precision_at_k(submission_dict, answer_dict, k=10))

    save_submission_pickle(
        submission_dict,
        "submission_valid.pkl"
    )


if __name__ == "__main__":
    main()
