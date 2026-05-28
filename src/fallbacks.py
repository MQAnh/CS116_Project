import polars as pl


def collect_user_ids(user_ids_lf):
    return (
        user_ids_lf
        .select(pl.col("customer_id").cast(pl.Int32))
        .unique()
        .collect()
        .get_column("customer_id")
        .to_list()
    )


def popular_items(hist_lf, top_k=50, skip_top_k=0):
    if top_k <= 0:
        return []

    return (
        hist_lf
        .group_by("item_id")
        .agg([
            pl.col("customer_id").n_unique().alias("n_customers"),
            pl.len().alias("n_transactions"),
        ])
        .sort(["n_customers", "n_transactions"], descending=[True, True])
        .slice(skip_top_k, top_k)
        .select("item_id")
        .collect()
        .get_column("item_id")
        .to_list()
    )


def recent_history_fallback_by_user(hist_lf, user_ids_lf=None, k=10):
    source_lf = hist_lf
    if user_ids_lf is not None:
        source_lf = source_lf.join(
            user_ids_lf.select(pl.col("customer_id").cast(pl.Int32)).unique(),
            on="customer_id",
            how="inner",
        )

    fallback_df = (
        source_lf
        .group_by(["customer_id", "item_id"])
        .agg([
            pl.col("updated_date").max().alias("last_seen_at"),
            pl.len().alias("n_transactions"),
        ])
        .sort(
            ["customer_id", "last_seen_at", "n_transactions"],
            descending=[False, True, True],
        )
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("item_id").head(k).alias("items"))
        .collect()
    )

    return {
        int(customer_id): items
        for customer_id, items in fallback_df.iter_rows()
    }
