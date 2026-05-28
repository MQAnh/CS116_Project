import polars as pl
from src import config as cfg
from src.cleanup import cleanup_paths
from src.data_loader import load_data
from src.splits import make_time_splits
from src.candidates import build_valid_candidates
from src.labels import make_ground_truth, make_labeled_dataset
from src.features import build_feature_chunk, make_feature_sources
from src.logging_utils import log_step, log_time
from src.preprocess import (
    get_preprocess_spec,
    inference_selected_features,
    prepare_matrix_lf,
    reset_output_dir,
)
from src.evaluate import (
    ground_truth_to_dict,
    print_candidate_recall_report,
    predict_topk_from_parquet,
    server_precision_at_k,
    topk_to_submission_dict,
    save_submission_pickle,
)
from src.fallbacks import (
    collect_user_ids,
    popular_items,
    recent_history_fallback_by_user,
)


def main():
    log_step("validation pipeline started")
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with log_time("clean stale validation intermediates"):
        cleanup_paths([
            cfg.VALID_FEATURES_CHUNKS_DIR,
            cfg.VALID_MODEL_READY_CHUNKS_DIR,
            cfg.VALID_FEATURES_PATH,
            cfg.VALID_MODEL_READY_PATH,
        ])
        if cfg.PREPROCESS_METADATA_PATH.exists():
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

    with log_time("build validation candidates"):
        valid_candidates_lf = build_valid_candidates(
            splits["valid_hist_lf"],
            items_lf=items_lf,
            min_bills=cfg.MIN_BILLS_ACTIVE_USER,
            recent_top_k=cfg.RECENT_TOP_K_VALID,
            frequent_top_k=cfg.FREQUENT_TOP_K_VALID,
            popular_top_k=cfg.POPULAR_CANDIDATE_TOP_K,
            category_col=cfg.CATEGORY_CANDIDATE_COL,
            user_top_categories=cfg.USER_TOP_CATEGORIES,
            category_items_per_category=cfg.CATEGORY_ITEMS_PER_CATEGORY,
            co_anchor_top_k=cfg.COOCCURRENCE_ANCHOR_TOP_K,
            co_top_k=cfg.COOCCURRENCE_TOP_K,
            co_max_bill_items=cfg.COOCCURRENCE_MAX_BILL_ITEMS,
        )
        valid_candidates_lf.sink_parquet(cfg.VALID_CANDIDATES_PATH)
        valid_candidates_lf = pl.scan_parquet(cfg.VALID_CANDIDATES_PATH)

    with log_time("create validation labels"):
        valid_gt_lf = make_ground_truth(splits["valid_label_lf"])
        valid_gt_lf.sink_parquet(cfg.VALID_GT_PATH)

    with log_time("label validation candidates"):
        valid_dataset_lf = make_labeled_dataset(valid_candidates_lf, valid_gt_lf)
        valid_dataset_lf.sink_parquet(cfg.VALID_DATASET_LABELS_PATH)
        valid_dataset_lf = pl.scan_parquet(cfg.VALID_DATASET_LABELS_PATH)

    with log_time("estimate validation candidate recall"):
        print_candidate_recall_report(valid_dataset_lf, valid_gt_lf, k=10)
        cleanup_paths([
            cfg.VALID_CANDIDATES_PATH,
            cfg.VALID_GT_PATH,
        ])

    with log_time("prepare validation preprocess spec"):
        selected_features = inference_selected_features(
            cfg.FEATURE_COLUMNS_PATH,
            cfg.SELECTED_FEATURES if cfg.FEATURE_SELECTION_ENABLED else None,
        )
        numeric_cols, cat_cols, category_mappings = get_preprocess_spec(
            cfg.TRAIN_FEATURES_CHUNKS_DIR,
            cfg.DROP_COLS,
            cfg.CAT_COLS,
            metadata_path=cfg.PREPROCESS_METADATA_PATH,
            selected_features=selected_features,
        )
        feature_cols = numeric_cols + cat_cols

    with log_time("build validation features and prepare matrix"):
        feature_sources = make_feature_sources(splits["valid_hist_lf"], items_lf)
        output_dir = reset_output_dir(cfg.VALID_MODEL_READY_CHUNKS_DIR)
        for chunk_idx in range(cfg.FEATURE_BUILD_CHUNKS):
            chunk_name = f"part_{chunk_idx:03d}.parquet"
            log_step(
                "build+prepare validation chunk "
                f"{chunk_idx + 1}/{cfg.FEATURE_BUILD_CHUNKS}: {chunk_name}"
            )
            chunk_features_lf = build_feature_chunk(
                valid_dataset_lf,
                feature_sources,
                chunk_idx,
                cfg.FEATURE_BUILD_CHUNKS,
            )
            chunk_model_lf = prepare_matrix_lf(
                chunk_features_lf,
                numeric_cols,
                cat_cols,
                category_mappings,
                ["customer_id", "item_id", "target"],
            )
            chunk_model_lf.sink_parquet(output_dir / chunk_name)
        cleanup_paths([cfg.VALID_DATASET_LABELS_PATH])

    with log_time("predict validation top-k"):
        top10 = predict_topk_from_parquet(
            cfg.VALID_MODEL_READY_CHUNKS_DIR,
            cfg.MODEL_PATH,
            feature_columns_path=cfg.FEATURE_COLUMNS_PATH,
            id_cols=["customer_id", "item_id", "target"],
            k=10,
            batch_size=cfg.PREDICT_BATCH_SIZE,
            repeat_boost=cfg.RERANK_REPEAT_BOOST,
            affinity_boost=cfg.RERANK_AFFINITY_BOOST,
            popularity_penalty=cfg.RERANK_POPULARITY_PENALTY,
        )

    with log_time("prepare validation fallback items"):
        history_user_ids_lf = splits["valid_hist_lf"].select("customer_id").unique()
        predicted_user_ids_lf = pl.LazyFrame({
            "customer_id": top10["customer_id"].astype("int32").unique()
        })
        missing_prediction_user_ids_lf = history_user_ids_lf.join(
            predicted_user_ids_lf,
            on="customer_id",
            how="anti",
        )
        user_ids = collect_user_ids(history_user_ids_lf)
        fallback_by_user = recent_history_fallback_by_user(
            splits["valid_hist_lf"],
            user_ids_lf=missing_prediction_user_ids_lf,
            k=10,
        )
        fallback_items = popular_items(splits["valid_hist_lf"], top_k=cfg.POPULAR_TOP_K)

    with log_time("evaluate validation predictions"):
        raw_submission_dict = topk_to_submission_dict(top10, k=10)
        answer_dict = ground_truth_to_dict(splits["valid_label_lf"])
        print(
            "Server Precision@10 (model only):",
            server_precision_at_k(raw_submission_dict, answer_dict, k=10),
        )
        submission_dict = topk_to_submission_dict(
            top10,
            k=10,
            user_ids=user_ids,
            fallback_items=fallback_items,
            fallback_by_user=fallback_by_user,
        )
        print(
            "Server Precision@10 (with fallback):",
            server_precision_at_k(submission_dict, answer_dict, k=10),
        )

    with log_time("save validation submission pickle"):
        save_submission_pickle(
            submission_dict,
            "submission_valid.pkl"
        )
    log_step("validation pipeline finished")


if __name__ == "__main__":
    main()
