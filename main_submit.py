import polars as pl

from src import config as cfg
from src.cleanup import cleanup_paths
from src.candidates import build_valid_candidates
from src.data_loader import load_data
from src.evaluate import (
    print_submission_item_concentration,
    predict_topk_from_parquet,
    save_submission_pickle,
    topk_to_submission_dict,
)
from src.fallbacks import (
    collect_user_ids,
    popular_items,
    recent_history_fallback_by_user,
)
from src.features import build_feature_chunk, make_cached_feature_sources
from src.logging_utils import log_step, log_time
from src.preprocess import (
    get_preprocess_spec,
    inference_selected_features,
    prepare_matrix_lf,
    reset_output_dir,
)
from src.splits import make_time_splits


def main():
    log_step("submission pipeline started")
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with log_time("clean stale submission intermediates"):
        cleanup_paths([
            cfg.FINAL_FEATURES_CHUNKS_DIR,
            cfg.FINAL_FEATURE_SOURCES_DIR,
            cfg.FINAL_MODEL_READY_CHUNKS_DIR,
            cfg.FINAL_FEATURES_PATH,
            cfg.FINAL_MODEL_READY_PATH,
        ])
        if cfg.PREPROCESS_METADATA_PATH.exists():
            cleanup_paths([
                cfg.TRAIN_FEATURES_CHUNKS_DIR,
                cfg.TRAIN_FEATURE_SOURCES_DIR,
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

    with log_time("build final candidates"):
        final_co_hist_lf = splits["final_hist_lf"]
        if cfg.FINAL_COOCCURRENCE_HISTORY_MONTHS is not None:
            final_co_hist_lf = final_co_hist_lf.filter(
                pl.col("month").is_between(*cfg.FINAL_COOCCURRENCE_HISTORY_MONTHS)
            )
        final_candidates_lf = build_valid_candidates(
            splits["final_hist_lf"],
            items_lf=items_lf,
            min_bills=cfg.MIN_BILLS_ACTIVE_USER,
            recent_top_k=cfg.RECENT_TOP_K_VALID,
            frequent_top_k=cfg.FREQUENT_TOP_K_VALID,
            popular_top_k=cfg.POPULAR_CANDIDATE_TOP_K,
            category_col=cfg.CATEGORY_CANDIDATE_COL,
            user_top_categories=cfg.USER_TOP_CATEGORIES,
            category_items_per_category=cfg.CATEGORY_ITEMS_PER_CATEGORY,
            co_anchor_top_k=cfg.FINAL_COOCCURRENCE_ANCHOR_TOP_K,
            co_top_k=cfg.FINAL_COOCCURRENCE_TOP_K,
            co_max_bill_items=cfg.FINAL_COOCCURRENCE_MAX_BILL_ITEMS,
            include_cooccurrence=cfg.FINAL_COOCCURRENCE_ENABLED,
            co_hist_lf=final_co_hist_lf,
        )
        final_candidates_lf.sink_parquet(cfg.FINAL_CANDIDATES_PATH)
        final_candidates_lf = pl.scan_parquet(cfg.FINAL_CANDIDATES_PATH)

    with log_time("prepare final preprocess spec"):
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

    with log_time("build final features and prepare matrix"):
        feature_sources = make_cached_feature_sources(
            splits["final_hist_lf"],
            items_lf,
            cfg.FINAL_FEATURE_SOURCES_DIR,
        )
        output_dir = reset_output_dir(cfg.FINAL_MODEL_READY_CHUNKS_DIR)
        for chunk_idx in range(cfg.FEATURE_BUILD_CHUNKS):
            chunk_name = f"part_{chunk_idx:03d}.parquet"
            log_step(
                "build+prepare final chunk "
                f"{chunk_idx + 1}/{cfg.FEATURE_BUILD_CHUNKS}: {chunk_name}"
            )
            chunk_features_lf = build_feature_chunk(
                final_candidates_lf,
                feature_sources,
                chunk_idx,
                cfg.FEATURE_BUILD_CHUNKS,
            )
            chunk_model_lf = prepare_matrix_lf(
                chunk_features_lf,
                numeric_cols,
                cat_cols,
                category_mappings,
                ["customer_id", "item_id"],
            )
            chunk_model_lf.sink_parquet(output_dir / chunk_name)
        cleanup_paths([cfg.FINAL_FEATURE_SOURCES_DIR])

    with log_time("predict final top-k"):
        top10 = predict_topk_from_parquet(
            cfg.FINAL_MODEL_READY_CHUNKS_DIR,
            cfg.MODEL_PATH,
            feature_columns_path=cfg.FEATURE_COLUMNS_PATH,
            id_cols=["customer_id", "item_id"],
            k=10,
            batch_size=cfg.PREDICT_BATCH_SIZE,
            repeat_boost=cfg.RERANK_REPEAT_BOOST,
            affinity_boost=cfg.RERANK_AFFINITY_BOOST,
            popularity_penalty=cfg.RERANK_POPULARITY_PENALTY,
        )

    with log_time("prepare fallback items"):
        history_user_ids_lf = splits["final_hist_lf"].select("customer_id").unique()
        candidate_user_ids_lf = final_candidates_lf.select("customer_id").unique()
        missing_candidate_user_ids_lf = history_user_ids_lf.join(
            candidate_user_ids_lf,
            on="customer_id",
            how="anti",
        )
        fallback_user_ids_lf = (
            history_user_ids_lf
            if cfg.RECENT_FALLBACK_FOR_ALL_USERS
            else missing_candidate_user_ids_lf
        )
        user_ids = collect_user_ids(history_user_ids_lf)
        fallback_by_user = recent_history_fallback_by_user(
            splits["final_hist_lf"],
            user_ids_lf=fallback_user_ids_lf,
            k=10,
        )
        fallback_items = popular_items(
            splits["final_hist_lf"],
            top_k=cfg.FALLBACK_ITEM_TOP_K,
            skip_top_k=cfg.FALLBACK_ITEM_SKIP_TOP_K,
        )

    with log_time("save final submission pickle"):
        submission_dict = topk_to_submission_dict(
            top10,
            k=10,
            user_ids=user_ids,
            fallback_items=fallback_items,
            fallback_by_user=fallback_by_user,
            rotate_fallback_items=cfg.ROTATE_FALLBACK_ITEMS,
        )
        print_submission_item_concentration(
            submission_dict,
            k=10,
            label="Final submission",
        )
        save_submission_pickle(submission_dict, cfg.SUBMISSION_PATH)
    log_step("submission pipeline finished")


if __name__ == "__main__":
    main()
