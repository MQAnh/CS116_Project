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
    lf = (
        hist_lf
        .join(active_users_lf, on="customer_id", how="inner")
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.col("updated_date").max().alias("last_seen_at"),
            pl.len().alias("recent_source_transactions"),
        ])
        .sort(
            ["customer_id", "last_seen_at", "recent_source_transactions"],
            descending=[False, True, True],
        )
        .with_columns(
            pl.col("last_seen_at").rank("ordinal", descending=True).over("customer_id").alias("recent_rank")
        )
        .filter(pl.col("recent_rank") <= top_k)
        .select(["customer_id", "item_id", "recent_rank"])
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
        .with_columns(
            pl.col("n_transactions").rank("ordinal", descending=True).over("customer_id").alias("frequent_rank")
        )
        .filter(pl.col("frequent_rank") <= top_k)
        .select(["customer_id", "item_id", "frequent_rank"])
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
        .with_columns(pl.col("n_customers").rank("ordinal", descending=True).alias("popular_rank"))
        .head(top_k)
        .select(["item_id", "popular_rank"])
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
        .with_columns(
            pl.col(category_col).cum_count().over("customer_id").alias("user_category_rank")
        )
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
        .with_columns(
            pl.col("n_customers").rank("ordinal", descending=True).over(category_col).alias("category_item_rank")
        )
        .filter(pl.col("category_item_rank") <= items_per_category)
        .select([category_col, "item_id", "category_item_rank"])
    )
    lf = user_categories_lf.join(category_items_lf, on=category_col, how="inner").select([
        "customer_id",
        "item_id",
        "user_category_rank",
        "category_item_rank",
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
    anchor_items_lf = (
        user_anchor_items_lf
        .select(pl.col("anchor_item_id").alias("item_id"))
        .unique()
    )
    anchor_bill_items_lf = (
        bill_items_lf
        .join(anchor_items_lf, on="item_id", how="inner")
        .rename({"item_id": "anchor_item_id"})
    )
    item_pairs_lf = (
        anchor_bill_items_lf
        .join(bill_items_lf, on="bill_id", how="inner")
        .filter(pl.col("anchor_item_id") != pl.col("item_id"))
        .group_by(["anchor_item_id", "item_id"])
        .agg(pl.len().alias("n_co_bills"))
        .sort(["anchor_item_id", "n_co_bills"], descending=[False, True])
        .with_columns(
            pl.col("n_co_bills").rank("ordinal", descending=True).over("anchor_item_id").alias("cooccurrence_rank")
        )
        .filter(pl.col("cooccurrence_rank") <= co_top_k)
        .select([
            "anchor_item_id",
            pl.col("item_id").alias("candidate_item_id"),
            "cooccurrence_rank",
        ])
    )
    lf = (
        user_anchor_items_lf
        .join(item_pairs_lf, on="anchor_item_id", how="inner")
        .select([
            "customer_id",
            pl.col("candidate_item_id").alias("item_id"),
            "cooccurrence_rank",
        ])
    )
    if source_name is not None:
        lf = lf.with_columns(pl.lit(source_name).alias("candidate_source"))
    return lf


def merge_candidates(candidate_lfs):
    rank_cols = [
        "recent_rank",
        "frequent_rank",
        "popular_rank",
        "user_category_rank",
        "category_item_rank",
        "cooccurrence_rank",
    ]
    source_lfs = []
    for lf in candidate_lfs:
        if "candidate_source" not in lf.collect_schema():
            lf = lf.with_columns(pl.lit("unknown").alias("candidate_source"))
        schema = lf.collect_schema()
        lf = lf.with_columns([
            pl.lit(None, dtype=pl.UInt32).alias(c)
            for c in rank_cols
            if c not in schema
        ])
        source_lfs.append(lf.select(["customer_id", "item_id", "candidate_source"] + rank_cols))

    return (
        pl.concat(source_lfs)
        .unique(["customer_id", "item_id", "candidate_source"])
        .with_columns([
            (pl.col("candidate_source") == "recent").cast(pl.Int8).alias("is_recent_candidate"),
            (pl.col("candidate_source") == "frequent").cast(pl.Int8).alias("is_frequent_candidate"),
            (pl.col("candidate_source") == "popular").cast(pl.Int8).alias("is_popular_candidate"),
            (pl.col("candidate_source") == "category_popular").cast(pl.Int8).alias("is_category_candidate"),
            (pl.col("candidate_source") == "cooccurrence").cast(pl.Int8).alias("is_cooccurrence_candidate"),
        ])
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.col("is_recent_candidate").max(),
            pl.col("is_frequent_candidate").max(),
            pl.col("is_popular_candidate").max(),
            pl.col("is_category_candidate").max(),
            pl.col("is_cooccurrence_candidate").max(),
            pl.col("candidate_source").n_unique().alias("n_candidate_sources"),
            *[pl.col(c).min().fill_null(9999).cast(pl.Int32).alias(c) for c in rank_cols],
        ])
    )


def build_train_candidates(
    train_hist_lf,
    items_lf=None,
    min_bills=2,
    recent_top_k=20,
    frequent_top_k=20,
    popular_top_k=0,
    category_col="category_l2",
    user_top_categories=3,
    category_items_per_category=20,
    co_anchor_top_k=20,
    co_top_k=10,
    co_max_bill_items=30,
    include_cooccurrence=True,
):
    active_users_lf = get_active_users(train_hist_lf, min_bills=min_bills)
    recent_lf = recent_candidates(train_hist_lf, active_users_lf, top_k=recent_top_k, use_unique=True)
    freq_lf = frequent_candidates(train_hist_lf, active_users_lf, top_k=frequent_top_k)
    candidate_lfs = [recent_lf, freq_lf]
    if popular_top_k > 0:
        candidate_lfs.append(popular_candidates(
            train_hist_lf,
            active_users_lf,
            top_k=popular_top_k,
        ))
    if items_lf is not None:
        candidate_lfs.append(category_popular_candidates(
            train_hist_lf,
            items_lf,
            active_users_lf,
            category_col=category_col,
            user_top_categories=user_top_categories,
            items_per_category=category_items_per_category,
        ))
    if include_cooccurrence:
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
    popular_top_k=0,
    category_col="category_l2",
    user_top_categories=3,
    category_items_per_category=20,
    co_anchor_top_k=20,
    co_top_k=10,
    co_max_bill_items=30,
    include_cooccurrence=True,
    co_hist_lf=None,
):
    active_users_lf = get_active_users(valid_hist_lf, min_bills=min_bills)
    recent_lf = recent_candidates(valid_hist_lf, active_users_lf, top_k=recent_top_k, use_unique=False).unique(["customer_id", "item_id"])
    freq_lf = frequent_candidates(valid_hist_lf, active_users_lf, top_k=frequent_top_k).unique(["customer_id", "item_id"])
    candidate_lfs = [recent_lf, freq_lf]
    if popular_top_k > 0:
        candidate_lfs.append(popular_candidates(
            valid_hist_lf,
            active_users_lf,
            top_k=popular_top_k,
        ))
    if items_lf is not None:
        candidate_lfs.append(category_popular_candidates(
            valid_hist_lf,
            items_lf,
            active_users_lf,
            category_col=category_col,
            user_top_categories=user_top_categories,
            items_per_category=category_items_per_category,
        ))
    if include_cooccurrence:
        co_source_lf = valid_hist_lf if co_hist_lf is None else co_hist_lf
        candidate_lfs.append(cooccurrence_candidates(
            co_source_lf,
            active_users_lf,
            anchor_top_k=co_anchor_top_k,
            co_top_k=co_top_k,
            max_bill_items=co_max_bill_items,
        ))
    return merge_candidates(candidate_lfs)
