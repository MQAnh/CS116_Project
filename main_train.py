from src import config as cfg
from src.data_loader import load_data
from src.splits import make_time_splits
from src.candidates import build_train_candidates
from src.labels import make_ground_truth, make_labeled_dataset
from src.features import build_features
from src.preprocess import prepare_train_matrix
from src.train import train_lgbm


def main():
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
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

    train_candidates_lf = build_train_candidates(
        splits["train_hist_lf"],
        min_bills=cfg.MIN_BILLS_ACTIVE_USER,
        recent_top_k=cfg.RECENT_TOP_K_TRAIN,
        frequent_top_k=cfg.FREQUENT_TOP_K_TRAIN,
    )
    train_candidates_lf.sink_parquet(cfg.TRAIN_CANDIDATES_PATH)
    train_candidates_lf = __import__("polars").scan_parquet(cfg.TRAIN_CANDIDATES_PATH)

    train_gt_lf = make_ground_truth(splits["train_label_lf"])
    train_dataset_lf = make_labeled_dataset(train_candidates_lf, train_gt_lf)
    train_dataset_lf.sink_parquet(cfg.TRAIN_DATASET_LABELS_PATH)
    train_dataset_lf = __import__("polars").scan_parquet(cfg.TRAIN_DATASET_LABELS_PATH)

    train_features_lf = build_features(splits["train_hist_lf"], train_dataset_lf, items_lf)
    train_features_lf.sink_parquet(cfg.TRAIN_FEATURES_PATH)
    train_features_lf = __import__("polars").scan_parquet(cfg.TRAIN_FEATURES_PATH)

    train_model_lf, feature_cols = prepare_train_matrix(train_features_lf, cfg.DROP_COLS, cfg.CAT_COLS)
    train_model_lf.sink_parquet(cfg.TRAIN_MODEL_READY_PATH)

    train_lgbm(
        cfg.TRAIN_MODEL_READY_PATH,
        cfg.MODEL_PATH,
        feature_columns_path=cfg.FEATURE_COLUMNS_PATH,
        importance_path=cfg.IMPORTANCE_PATH,
    )


if __name__ == "__main__":
    main()
