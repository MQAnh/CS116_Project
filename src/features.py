import shutil
from datetime import timedelta

import polars as pl

from src.logging_utils import log_step


def make_item_meta(items_lf):
    return items_lf.select([
        "item_id",
        pl.col("price").cast(pl.Float32).alias("item_catalog_price"),
        "category_l1",
        "category_l2",
        "category_l3",
        "category",
        "brand",
        "manufacturer",
        "sale_status",
        "size",
    ])


def recent_expr(max_date, days):
    return pl.col("date") >= pl.lit(max_date - timedelta(days=days))


def discount_rate_expr():
    return (
        pl.when(pl.col("price") > 0)
        .then(pl.col("discount") / pl.col("price"))
        .otherwise(0)
    )


def make_user_features(hist_lf, max_date):
    enriched_lf = hist_lf.with_columns([
        (pl.col("price") * pl.col("quantity")).alias("line_value"),
        discount_rate_expr().alias("discount_rate"),
    ])
    return (
        enriched_lf
        .group_by("customer_id")
        .agg([
            pl.len().alias("user_n_transactions"),
            pl.col("bill_id").n_unique().alias("user_n_bills"),
            pl.col("item_id").n_unique().alias("user_n_unique_items"),
            pl.col("quantity").sum().alias("user_total_quantity"),
            pl.col("date").n_unique().alias("user_n_active_days"),
            (pl.lit(max_date) - pl.col("date").max()).dt.total_days().alias("user_days_since_last_bill"),
            pl.col("location").n_unique().alias("user_n_locations"),
            pl.col("price").mean().alias("user_avg_price"),
            pl.col("price").median().alias("user_median_price"),
            pl.col("price").quantile(0.25).alias("user_price_p25"),
            pl.col("price").quantile(0.75).alias("user_price_p75"),
            pl.col("discount").mean().alias("user_avg_discount"),
            pl.col("discount_rate").mean().alias("user_avg_discount_rate"),
            (pl.col("discount") > 0).mean().alias("user_discount_positive_rate"),
            pl.col("line_value").sum().alias("user_total_spend"),
            pl.col("line_value").mean().alias("user_avg_line_value"),
            pl.col("item_id").filter(recent_expr(max_date, 30)).count().alias("user_n_transactions_30d"),
            pl.col("item_id").filter(recent_expr(max_date, 60)).count().alias("user_n_transactions_60d"),
            pl.col("bill_id").filter(recent_expr(max_date, 30)).n_unique().alias("user_n_bills_30d"),
            pl.col("bill_id").filter(recent_expr(max_date, 60)).n_unique().alias("user_n_bills_60d"),
            pl.col("item_id").filter(recent_expr(max_date, 30)).n_unique().alias("user_n_unique_items_30d"),
            pl.col("item_id").filter(recent_expr(max_date, 60)).n_unique().alias("user_n_unique_items_60d"),
        ])
        .with_columns([
            (pl.col("user_n_transactions") / pl.col("user_n_bills")).alias("user_avg_items_per_bill"),
            (pl.col("user_total_quantity") / pl.col("user_n_bills")).alias("user_avg_quantity_per_bill"),
            (pl.col("user_total_spend") / pl.col("user_n_bills")).alias("user_avg_bill_value"),
        ])
    )


def make_item_features(hist_lf, item_meta_lf, max_date):
    enriched_lf = hist_lf.with_columns([
        discount_rate_expr().alias("discount_rate"),
    ])
    return (
        enriched_lf
        .group_by("item_id")
        .agg([
            pl.len().alias("item_n_transactions"),
            pl.col("customer_id").n_unique().alias("item_n_customers"),
            pl.col("bill_id").n_unique().alias("item_n_bills"),
            pl.col("quantity").sum().alias("item_total_quantity"),
            pl.col("price").mean().alias("item_avg_price"),
            pl.col("discount").mean().alias("item_avg_discount"),
            pl.col("discount_rate").mean().alias("item_avg_discount_rate"),
            (pl.col("discount") > 0).mean().alias("item_discount_positive_rate"),
            (pl.lit(max_date) - pl.col("date").max()).dt.total_days().alias("item_days_since_last_seen"),
            (pl.lit(max_date) - pl.col("date").min()).dt.total_days().alias("item_days_since_first_seen"),
            pl.col("customer_id").filter(recent_expr(max_date, 30)).n_unique().alias("item_n_customers_30d"),
            pl.col("customer_id").filter(recent_expr(max_date, 60)).n_unique().alias("item_n_customers_60d"),
            pl.col("item_id").filter(recent_expr(max_date, 30)).count().alias("item_n_transactions_30d"),
            pl.col("item_id").filter(recent_expr(max_date, 60)).count().alias("item_n_transactions_60d"),
        ])
        .join(item_meta_lf, on="item_id", how="left")
        .with_columns([
            (pl.col("item_n_transactions_30d") / pl.col("item_n_transactions")).fill_null(0).alias("item_transaction_share_30d"),
            (pl.col("item_n_transactions_60d") / pl.col("item_n_transactions")).fill_null(0).alias("item_transaction_share_60d"),
        ])
    )


def make_user_taxonomy_features(hist_lf, item_meta_lf, column, prefix, max_date):
    return (
        hist_lf
        .join(item_meta_lf.select(["item_id", column]), on="item_id", how="left")
        .filter(pl.col(column).is_not_null())
        .group_by(["customer_id", column])
        .agg([
            pl.len().alias(f"{prefix}_n_transactions"),
            pl.col("bill_id").n_unique().alias(f"{prefix}_n_bills"),
            pl.col("quantity").sum().alias(f"{prefix}_total_quantity"),
            pl.col("item_id").filter(recent_expr(max_date, 30)).count().alias(f"{prefix}_n_transactions_30d"),
            (pl.lit(max_date) - pl.col("date").max()).dt.total_days().alias(f"{prefix}_recency_days"),
        ])
    )


def make_user_item_features(hist_lf, max_date):
    return (
        hist_lf
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("ui_n_transactions"),
            pl.col("bill_id").n_unique().alias("ui_n_bills"),
            pl.col("quantity").sum().alias("ui_total_quantity"),
            pl.col("date").max().alias("ui_last_date"),
            pl.col("date").min().alias("ui_first_date"),
            pl.col("item_id").filter(recent_expr(max_date, 30)).count().alias("ui_n_transactions_30d"),
            pl.col("item_id").filter(recent_expr(max_date, 60)).count().alias("ui_n_transactions_60d"),
        ])
        .with_columns([
            (pl.lit(max_date) - pl.col("ui_last_date")).dt.total_days().alias("ui_recency_days"),
            (pl.col("ui_last_date") - pl.col("ui_first_date")).dt.total_days().alias("ui_purchase_span_days"),
        ])
        .with_columns([
            (
                pl.col("ui_purchase_span_days")
                / pl.when(pl.col("ui_n_bills") > 1).then(pl.col("ui_n_bills") - 1).otherwise(None)
            ).fill_null(9999).alias("ui_avg_repeat_interval_days"),
        ])
        .with_columns([
            (
                pl.col("ui_recency_days")
                / pl.when(pl.col("ui_avg_repeat_interval_days") > 0)
                .then(pl.col("ui_avg_repeat_interval_days"))
                .otherwise(None)
            ).fill_null(0).alias("ui_repeat_due_ratio")
        ])
    )


def make_user_basket_features(hist_lf):
    bill_lf = (
        hist_lf
        .with_columns((pl.col("price") * pl.col("quantity")).alias("line_value"))
        .group_by(["customer_id", "bill_id"])
        .agg([
            pl.len().alias("bill_n_lines"),
            pl.col("item_id").n_unique().alias("bill_n_items"),
            pl.col("quantity").sum().alias("bill_total_quantity"),
            pl.col("line_value").sum().alias("bill_value"),
        ])
    )
    return (
        bill_lf
        .group_by("customer_id")
        .agg([
            pl.col("bill_n_lines").mean().alias("user_avg_bill_lines"),
            pl.col("bill_n_items").mean().alias("user_avg_bill_items"),
            pl.col("bill_total_quantity").mean().alias("user_avg_bill_quantity"),
            pl.col("bill_value").mean().alias("user_avg_bill_value_from_bills"),
        ])
    )


def make_item_basket_features(hist_lf):
    bill_lf = (
        hist_lf
        .with_columns((pl.col("price") * pl.col("quantity")).alias("line_value"))
        .group_by("bill_id")
        .agg([
            pl.len().alias("bill_n_lines"),
            pl.col("item_id").n_unique().alias("bill_n_items"),
            pl.col("line_value").sum().alias("bill_value"),
        ])
    )
    return (
        hist_lf
        .select(["bill_id", "item_id"])
        .join(bill_lf, on="bill_id", how="left")
        .group_by("item_id")
        .agg([
            pl.col("bill_n_items").mean().alias("item_avg_bill_items"),
            pl.col("bill_value").mean().alias("item_avg_bill_value"),
        ])
    )


def make_user_main_location(hist_lf):
    return (
        hist_lf
        .group_by(["customer_id", "location"])
        .agg(pl.len().alias("user_location_transactions"))
        .sort(["customer_id", "user_location_transactions"], descending=[False, True])
        .group_by("customer_id")
        .agg([
            pl.col("location").first().alias("user_main_location"),
            pl.col("user_location_transactions").first().alias("user_main_location_transactions"),
        ])
    )


def make_item_location_features(hist_lf):
    return (
        hist_lf
        .group_by(["item_id", "location"])
        .agg([
            pl.len().alias("item_location_transactions"),
            pl.col("customer_id").n_unique().alias("item_location_customers"),
        ])
        .rename({"location": "user_main_location"})
    )


def join_features(
    dataset_lf,
    user_features_lf,
    item_features_lf,
    user_item_features_lf,
    user_l1_features_lf,
    user_l2_features_lf,
    user_l3_features_lf,
    user_category_features_lf,
    user_brand_features_lf,
    user_manufacturer_features_lf,
    user_basket_features_lf,
    item_basket_features_lf,
    user_main_location_lf,
    item_location_features_lf,
):
    return (
        dataset_lf
        .join(user_features_lf, on="customer_id", how="left")
        .join(user_basket_features_lf, on="customer_id", how="left")
        .join(user_main_location_lf, on="customer_id", how="left")
        .join(item_features_lf, on="item_id", how="left")
        .join(item_basket_features_lf, on="item_id", how="left")
        .join(item_location_features_lf, on=["item_id", "user_main_location"], how="left")
        .join(user_l1_features_lf, on=["customer_id", "category_l1"], how="left")
        .join(user_l2_features_lf, on=["customer_id", "category_l2"], how="left")
        .join(user_l3_features_lf, on=["customer_id", "category_l3"], how="left")
        .join(user_category_features_lf, on=["customer_id", "category"], how="left")
        .join(user_brand_features_lf, on=["customer_id", "brand"], how="left")
        .join(user_manufacturer_features_lf, on=["customer_id", "manufacturer"], how="left")
        .join(user_item_features_lf, on=["customer_id", "item_id"], how="left")
        .with_columns([
            pl.col("ui_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("ui_n_bills").fill_null(0).cast(pl.Int32),
            pl.col("ui_total_quantity").fill_null(0).cast(pl.Int32),
            pl.col("ui_recency_days").fill_null(9999).cast(pl.Int32),
            pl.col("uc_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("uc_n_bills").fill_null(0).cast(pl.Int32),
            pl.col("uc_total_quantity").fill_null(0).cast(pl.Int32),
            pl.col("ub_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("ub_n_bills").fill_null(0).cast(pl.Int32),
            pl.col("ub_total_quantity").fill_null(0).cast(pl.Int32),
            pl.col("ul1_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("ul2_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("ul3_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("uman_n_transactions").fill_null(0).cast(pl.Int32),
            pl.col("ui_n_transactions_30d").fill_null(0).cast(pl.Int32),
            pl.col("ui_n_transactions_60d").fill_null(0).cast(pl.Int32),
            pl.col("item_location_transactions").fill_null(0).cast(pl.Int32),
            pl.col("item_location_customers").fill_null(0).cast(pl.Int32),
        ])
        .with_columns([
            (pl.col("uc_n_transactions") / pl.col("user_n_transactions")).fill_null(0).alias("uc_transaction_share"),
            (pl.col("ub_n_transactions") / pl.col("user_n_transactions")).fill_null(0).alias("ub_transaction_share"),
            (pl.col("ul1_n_transactions") / pl.col("user_n_transactions")).fill_null(0).alias("ul1_transaction_share"),
            (pl.col("ul2_n_transactions") / pl.col("user_n_transactions")).fill_null(0).alias("ul2_transaction_share"),
            (pl.col("ul3_n_transactions") / pl.col("user_n_transactions")).fill_null(0).alias("ul3_transaction_share"),
            (pl.col("uman_n_transactions") / pl.col("user_n_transactions")).fill_null(0).alias("uman_transaction_share"),
            (pl.col("item_avg_price") / pl.col("user_avg_price")).fill_null(0).alias("item_price_to_user_avg"),
            (pl.col("item_avg_price") - pl.col("user_avg_price")).abs().fill_null(0).alias("item_price_user_avg_abs_diff"),
            (
                (pl.col("item_avg_price") >= pl.col("user_price_p25"))
                & (pl.col("item_avg_price") <= pl.col("user_price_p75"))
            ).fill_null(False).cast(pl.Int8).alias("item_price_in_user_iqr"),
            (pl.col("item_avg_discount_rate") - pl.col("user_avg_discount_rate")).fill_null(0).alias("item_user_discount_rate_diff"),
            (pl.col("item_location_transactions") / pl.col("item_n_transactions")).fill_null(0).alias("item_main_location_transaction_share"),
        ])
    )


def build_features(hist_lf, dataset_lf, items_lf):
    max_date = hist_lf.select(pl.col("date").max()).collect()[0, 0]
    feature_sources = make_feature_sources(hist_lf, items_lf, max_date=max_date)
    return join_features(dataset_lf, **feature_sources)


def make_feature_sources(hist_lf, items_lf, max_date=None):
    if max_date is None:
        max_date = hist_lf.select(pl.col("date").max()).collect()[0, 0]
    item_meta_lf = make_item_meta(items_lf)
    return {
        "user_features_lf": make_user_features(hist_lf, max_date),
        "item_features_lf": make_item_features(hist_lf, item_meta_lf, max_date),
        "user_item_features_lf": make_user_item_features(hist_lf, max_date),
        "user_l1_features_lf": make_user_taxonomy_features(hist_lf, item_meta_lf, "category_l1", "ul1", max_date),
        "user_l2_features_lf": make_user_taxonomy_features(hist_lf, item_meta_lf, "category_l2", "ul2", max_date),
        "user_l3_features_lf": make_user_taxonomy_features(hist_lf, item_meta_lf, "category_l3", "ul3", max_date),
        "user_category_features_lf": make_user_taxonomy_features(hist_lf, item_meta_lf, "category", "uc", max_date),
        "user_brand_features_lf": make_user_taxonomy_features(hist_lf, item_meta_lf, "brand", "ub", max_date),
        "user_manufacturer_features_lf": make_user_taxonomy_features(hist_lf, item_meta_lf, "manufacturer", "uman", max_date),
        "user_basket_features_lf": make_user_basket_features(hist_lf),
        "item_basket_features_lf": make_item_basket_features(hist_lf),
        "user_main_location_lf": make_user_main_location(hist_lf),
        "item_location_features_lf": make_item_location_features(hist_lf),
    }


def materialize_feature_sources(feature_sources, output_dir):
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cached_sources = {}
    for name, source_lf in feature_sources.items():
        source_path = output_dir / f"{name}.parquet"
        log_step(f"cache feature source: {source_path.name}")
        source_lf.sink_parquet(source_path)
        cached_sources[name] = pl.scan_parquet(source_path)
    return cached_sources


def make_cached_feature_sources(hist_lf, items_lf, output_dir, max_date=None):
    return materialize_feature_sources(
        make_feature_sources(hist_lf, items_lf, max_date=max_date),
        output_dir,
    )


def build_feature_chunk(dataset_lf, feature_sources, chunk_idx, n_chunks):
    chunk_dataset_lf = dataset_lf.filter(
        (pl.col("customer_id") % n_chunks) == chunk_idx
    )
    return join_features(chunk_dataset_lf, **feature_sources)


def build_features_chunked(
    hist_lf,
    dataset_lf,
    items_lf,
    output_dir,
    n_chunks=16,
    feature_sources_dir=None,
):
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_sources = make_feature_sources(hist_lf, items_lf)
    if feature_sources_dir is not None:
        feature_sources = materialize_feature_sources(
            feature_sources,
            feature_sources_dir,
        )

    for chunk_idx in range(n_chunks):
        chunk_path = output_dir / f"part_{chunk_idx:03d}.parquet"
        log_step(f"build features chunk {chunk_idx + 1}/{n_chunks}: {chunk_path.name}")
        chunk_features_lf = build_feature_chunk(
            dataset_lf,
            feature_sources,
            chunk_idx,
            n_chunks,
        )
        chunk_features_lf.sink_parquet(chunk_path)

    return pl.scan_parquet(str(output_dir / "*.parquet"))
