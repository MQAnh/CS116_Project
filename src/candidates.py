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


def category_popular_candidates(
    hist_lf,
    items_lf,
    active_users_lf,
    category_col="category_l2",
    user_top_categories=3,
    items_per_category=20,
    source_name="category_popular",
):
    item_categories_lf = items_lf.select(["item_id", category_col])
    user_categories_lf = (
        hist_lf
        .join(active_users_lf, on="customer_id", how="inner")
        .join(item_categories_lf, on="item_id", how="left")
        .filter(pl.col(category_col).is_not_null())
        .group_by(["customer_id", category_col])
        .agg(pl.len().alias("n_transactions"))
        .sort(["customer_id", "n_transactions"], descending=[False, True])
        .group_by("customer_id")
        .agg(pl.col(category_col).head(user_top_categories).alias(category_col))
        .explode(category_col)
    )
    category_items_lf = (
        hist_lf
        .join(item_categories_lf, on="item_id", how="left")
        .filter(pl.col(category_col).is_not_null())
        .group_by([category_col, "item_id"])
        .agg([
            pl.col("customer_id").n_unique().alias("n_customers"),
            pl.len().alias("n_transactions"),
            pl.col("quantity").sum().alias("total_quantity"),
        ])
        .sort(
            [category_col, "n_customers", "n_transactions", "total_quantity"],
            descending=[False, True, True, True],
        )
        .group_by(category_col)
        .agg(pl.col("item_id").head(items_per_category).alias("item_id"))
        .explode("item_id")
    )
    lf = user_categories_lf.join(category_items_lf, on=category_col, how="inner").select([
        "customer_id",
        "item_id",
    ])
    if source_name is not None:
        lf = lf.with_columns(pl.lit(source_name).alias("candidate_source"))
    return lf


def cooccurrence_candidates(
    hist_lf,
    active_users_lf,
    anchor_top_k=20,
    co_top_k=10,
    max_bill_items=30,
    source_name="cooccurrence",
):
    user_anchor_items_lf = (
        hist_lf
        .join(active_users_lf, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.len().alias("n_transactions"),
            pl.col("quantity").sum().alias("total_quantity"),
        ])
        .sort(["customer_id", "n_transactions", "total_quantity"], descending=[False, True, True])
        .group_by("customer_id")
        .agg(pl.col("item_id").head(anchor_top_k).alias("anchor_item_id"))
        .explode("anchor_item_id")
    )
    bill_items_lf = (
        hist_lf
        .select(["bill_id", "item_id"])
        .unique()
        .join(
            hist_lf
            .group_by("bill_id")
            .agg(pl.col("item_id").n_unique().alias("n_bill_items"))
            .filter(pl.col("n_bill_items") <= max_bill_items)
            .select("bill_id"),
            on="bill_id",
            how="inner",
        )
    )
    item_pairs_lf = (
        bill_items_lf
        .join(bill_items_lf, on="bill_id", how="inner", suffix="_co")
        .filter(pl.col("item_id") != pl.col("item_id_co"))
        .group_by(["item_id", "item_id_co"])
        .agg(pl.len().alias("n_co_bills"))
        .sort(["item_id", "n_co_bills"], descending=[False, True])
        .group_by("item_id")
        .agg(pl.col("item_id_co").head(co_top_k).alias("candidate_item_id"))
        .explode("candidate_item_id")
        .rename({"item_id": "anchor_item_id"})
    )
    lf = (
        user_anchor_items_lf
        .join(item_pairs_lf, on="anchor_item_id", how="inner")
        .select([
            "customer_id",
            pl.col("candidate_item_id").alias("item_id"),
        ])
    )
    if source_name is not None:
        lf = lf.with_columns(pl.lit(source_name).alias("candidate_source"))
    return lf


def merge_candidates(candidate_lfs):
    source_lfs = []
    for lf in candidate_lfs:
        if "candidate_source" not in lf.collect_schema():
            lf = lf.with_columns(pl.lit("unknown").alias("candidate_source"))
        source_lfs.append(lf.select(["customer_id", "item_id", "candidate_source"]))

    return (
        pl.concat(source_lfs)
        .unique(["customer_id", "item_id", "candidate_source"])
        .with_columns([
            (pl.col("candidate_source") == "recent").cast(pl.Int8).alias("is_recent_candidate"),
            (pl.col("candidate_source") == "frequent").cast(pl.Int8).alias("is_frequent_candidate"),
            (pl.col("candidate_source") == "category_popular").cast(pl.Int8).alias("is_category_candidate"),
            (pl.col("candidate_source") == "cooccurrence").cast(pl.Int8).alias("is_cooccurrence_candidate"),
        ])
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.col("is_recent_candidate").max(),
            pl.col("is_frequent_candidate").max(),
            pl.col("is_category_candidate").max(),
            pl.col("is_cooccurrence_candidate").max(),
            pl.col("candidate_source").n_unique().alias("n_candidate_sources"),
        ])
    )


def build_train_candidates(
    train_hist_lf,
    items_lf=None,
    min_bills=2,
    recent_top_k=20,
    frequent_top_k=20,
    category_col="category_l2",
    user_top_categories=3,
    category_items_per_category=20,
    co_anchor_top_k=20,
    co_top_k=10,
    co_max_bill_items=30,
):
    active_users_lf = get_active_users(train_hist_lf, min_bills=min_bills)
    recent_lf = recent_candidates(train_hist_lf, active_users_lf, top_k=recent_top_k, use_unique=True)
    freq_lf = frequent_candidates(train_hist_lf, active_users_lf, top_k=frequent_top_k)
    candidate_lfs = [recent_lf, freq_lf]
    if items_lf is not None:
        candidate_lfs.append(category_popular_candidates(
            train_hist_lf,
            items_lf,
            active_users_lf,
            category_col=category_col,
            user_top_categories=user_top_categories,
            items_per_category=category_items_per_category,
        ))
    candidate_lfs.append(cooccurrence_candidates(
        train_hist_lf,
        active_users_lf,
        anchor_top_k=co_anchor_top_k,
        co_top_k=co_top_k,
        max_bill_items=co_max_bill_items,
    ))
    return merge_candidates(candidate_lfs)


def build_valid_candidates(
    valid_hist_lf,
    items_lf=None,
    min_bills=2,
    recent_top_k=30,
    frequent_top_k=20,
    category_col="category_l2",
    user_top_categories=3,
    category_items_per_category=20,
    co_anchor_top_k=20,
    co_top_k=10,
    co_max_bill_items=30,
):
    active_users_lf = get_active_users(valid_hist_lf, min_bills=min_bills)
    recent_lf = recent_candidates(valid_hist_lf, active_users_lf, top_k=recent_top_k, use_unique=False).unique(["customer_id", "item_id"])
    freq_lf = frequent_candidates(valid_hist_lf, active_users_lf, top_k=frequent_top_k).unique(["customer_id", "item_id"])
    candidate_lfs = [recent_lf, freq_lf]
    if items_lf is not None:
        candidate_lfs.append(category_popular_candidates(
            valid_hist_lf,
            items_lf,
            active_users_lf,
            category_col=category_col,
            user_top_categories=user_top_categories,
            items_per_category=category_items_per_category,
        ))
    candidate_lfs.append(cooccurrence_candidates(
        valid_hist_lf,
        active_users_lf,
        anchor_top_k=co_anchor_top_k,
        co_top_k=co_top_k,
        max_bill_items=co_max_bill_items,
    ))
    return merge_candidates(candidate_lfs)
