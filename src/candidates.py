import polars as pl


def get_active_users(hist_lf, min_bills=2):
    return (
        hist_lf
        .group_by("customer_id")
        .agg(pl.col("bill_id").n_unique().alias("n_bills"))
        .filter(pl.col("n_bills") >= min_bills)
        .select("customer_id")
    )


def recent_candidates(hist_lf, active_users_lf, top_k=20, use_unique=True, source_name="recent"):
    item_expr = pl.col("item_id").unique().head(top_k) if use_unique else pl.col("item_id").head(top_k)
    lf = (
        hist_lf
        .join(active_users_lf, on="customer_id", how="inner")
        .sort(["customer_id", "updated_date"], descending=[False, True])
        .group_by("customer_id")
        .agg(item_expr.alias("candidate_items"))
        .explode("candidate_items")
        .rename({"candidate_items": "item_id"})
    )
    if source_name is not None:
        lf = lf.with_columns(pl.lit(source_name).alias("candidate_source"))
    return lf


def frequent_candidates(hist_lf, active_users_lf, top_k=20, source_name="frequent"):
    lf = (
        hist_lf
        .join(active_users_lf, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("n_transactions"),
            pl.col("quantity").sum().alias("total_quantity"),
        ])
        .sort(["customer_id", "n_transactions", "total_quantity"], descending=[False, True, True])
        .group_by("customer_id")
        .agg(pl.col("item_id").head(top_k).alias("candidate_items"))
        .explode("candidate_items")
        .rename({"candidate_items": "item_id"})
    )
    if source_name is not None:
        lf = lf.with_columns(pl.lit(source_name).alias("candidate_source"))
    return lf


def popular_candidates(hist_lf, active_users_lf, top_k=50, source_name="popular"):
    top_popular_items_lf = (
        hist_lf
        .group_by("item_id")
        .agg([
            pl.len().alias("n_transactions"),
            pl.col("customer_id").n_unique().alias("n_customers"),
            pl.col("quantity").sum().alias("total_quantity"),
        ])
        .sort(["n_customers", "n_transactions"], descending=[True, True])
        .head(top_k)
        .select("item_id")
    )
    return active_users_lf.join(top_popular_items_lf, how="cross").with_columns(
        pl.lit(source_name).alias("candidate_source")
    )


def merge_candidates(candidate_lfs):
    cols = ["customer_id", "item_id"]
    return pl.concat([lf.select(cols) for lf in candidate_lfs]).unique(cols)


def build_train_candidates(train_hist_lf, min_bills=2, recent_top_k=20, frequent_top_k=20):
    active_users_lf = get_active_users(train_hist_lf, min_bills=min_bills)
    recent_lf = recent_candidates(train_hist_lf, active_users_lf, top_k=recent_top_k, use_unique=True)
    freq_lf = frequent_candidates(train_hist_lf, active_users_lf, top_k=frequent_top_k)
    return merge_candidates([recent_lf, freq_lf])


def build_valid_candidates(valid_hist_lf, min_bills=2, recent_top_k=30, frequent_top_k=20):
    active_users_lf = get_active_users(valid_hist_lf, min_bills=min_bills)
    recent_lf = recent_candidates(valid_hist_lf, active_users_lf, top_k=recent_top_k, use_unique=False, source_name=None).unique(["customer_id", "item_id"])
    freq_lf = frequent_candidates(valid_hist_lf, active_users_lf, top_k=frequent_top_k, source_name=None).unique(["customer_id", "item_id"])
    return merge_candidates([recent_lf, freq_lf])
