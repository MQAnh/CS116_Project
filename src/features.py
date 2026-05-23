import shutil

import polars as pl

from src.logging_utils import log_step


def make_item_meta(items_lf):
    return items_lf.select([
        "item_id",
        "category_l1",
        "category_l2",
        "category_l3",
        "category",
        "brand",
        "manufacturer",
        "sale_status",
        "size",
    ])


def make_user_features(hist_lf):
    return (
        hist_lf
        .group_by("customer_id")
        .agg([
            pl.len().alias("user_n_transactions"),
            pl.col("bill_id").n_unique().alias("user_n_bills"),
            pl.col("item_id").n_unique().alias("user_n_unique_items"),
            pl.col("quantity").sum().alias("user_total_quantity"),
            pl.col("date").n_unique().alias("user_n_active_days"),
            pl.col("price").mean().alias("user_avg_price"),
            pl.col("discount").mean().alias("user_avg_discount"),
        ])
        .with_columns([
            (pl.col("user_n_transactions") / pl.col("user_n_bills")).alias("user_avg_items_per_bill"),
            (pl.col("user_total_quantity") / pl.col("user_n_bills")).alias("user_avg_quantity_per_bill"),
        ])
    )


def make_item_features(hist_lf, item_meta_lf):
    return (
        hist_lf
        .group_by("item_id")
        .agg([
            pl.len().alias("item_n_transactions"),
            pl.col("customer_id").n_unique().alias("item_n_customers"),
            pl.col("bill_id").n_unique().alias("item_n_bills"),
            pl.col("quantity").sum().alias("item_total_quantity"),
            pl.col("price").mean().alias("item_avg_price"),
            pl.col("discount").mean().alias("item_avg_discount"),
        ])
        .join(item_meta_lf, on="item_id", how="left")
    )


def make_user_item_features(hist_lf):
    max_date = hist_lf.select(pl.col("date").max()).collect()[0, 0]
    return (
        hist_lf
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("ui_n_transactions"),
            pl.col("bill_id").n_unique().alias("ui_n_bills"),
            pl.col("quantity").sum().alias("ui_total_quantity"),
            pl.col("date").max().alias("ui_last_date"),
            pl.col("date").min().alias("ui_first_date"),
        ])
        .with_columns([
            (pl.lit(max_date) - pl.col("ui_last_date")).dt.total_days().alias("ui_recency_days")
        ])
    )


def join_features(dataset_lf, user_features_lf, item_features_lf, user_item_features_lf):
    return (
        dataset_lf
        .join(user_features_lf, on="customer_id", how="left")
        .join(item_features_lf, on="item_id", how="left")
        .join(user_item_features_lf, on=["customer_id", "item_id"], how="left")
        .with_columns([
            pl.col("ui_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("ui_n_bills").fill_null(0).cast(pl.Int32),
            pl.col("ui_total_quantity").fill_null(0).cast(pl.Int32),
            pl.col("ui_recency_days").fill_null(9999).cast(pl.Int32),
        ])
    )


def build_features(hist_lf, dataset_lf, items_lf):
    item_meta_lf = make_item_meta(items_lf)
    user_features_lf = make_user_features(hist_lf)
    item_features_lf = make_item_features(hist_lf, item_meta_lf)
    user_item_features_lf = make_user_item_features(hist_lf)
    return join_features(dataset_lf, user_features_lf, item_features_lf, user_item_features_lf)


def build_features_chunked(hist_lf, dataset_lf, items_lf, output_dir, n_chunks=16):
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_meta_lf = make_item_meta(items_lf)
    user_features_lf = make_user_features(hist_lf)
    item_features_lf = make_item_features(hist_lf, item_meta_lf)
    user_item_features_lf = make_user_item_features(hist_lf)

    for chunk_idx in range(n_chunks):
        chunk_path = output_dir / f"part_{chunk_idx:03d}.parquet"
        log_step(f"build features chunk {chunk_idx + 1}/{n_chunks}: {chunk_path.name}")
        chunk_dataset_lf = dataset_lf.filter(
            (pl.col("customer_id") % n_chunks) == chunk_idx
        )
        chunk_features_lf = join_features(
            chunk_dataset_lf,
            user_features_lf,
            item_features_lf,
            user_item_features_lf,
        )
        chunk_features_lf.sink_parquet(chunk_path)

    return pl.scan_parquet(str(output_dir / "*.parquet"))
