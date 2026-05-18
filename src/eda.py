import polars as pl


def basic_stats(transactions_lf):
    return transactions_lf.select([
        pl.len().alias("n_rows"),
        pl.col("customer_id").n_unique().alias("n_customers"),
        pl.col("item_id").n_unique().alias("n_items"),
        pl.col("bill_id").n_unique().alias("n_bills"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ]).collect()


def month_stats(transactions_lf):
    return transactions_lf.group_by("month").agg([
        pl.len().alias("n_rows"),
        pl.col("customer_id").n_unique().alias("n_customers"),
        pl.col("item_id").n_unique().alias("n_items"),
        pl.col("bill_id").n_unique().alias("n_bills"),
        pl.col("date").n_unique().alias("n_days"),
    ]).sort("month").collect()


def basket_stats(train_hist_lf):
    basket_lf = train_hist_lf.group_by("bill_id").agg([
        pl.col("item_id").n_unique().alias("basket_size"),
        pl.col("quantity").sum().alias("total_quantity"),
    ])
    return basket_lf.select([
        pl.col("basket_size").mean().alias("avg_basket_size"),
        pl.col("basket_size").median().alias("median_basket_size"),
        pl.col("basket_size").quantile(0.95).alias("p95_basket_size"),
        pl.col("basket_size").max().alias("max_basket_size"),
    ]).collect()


def customer_stats(train_hist_lf):
    customer_lf = train_hist_lf.group_by("customer_id").agg([
        pl.len().alias("n_transactions"),
        pl.col("bill_id").n_unique().alias("n_bills"),
        pl.col("item_id").n_unique().alias("n_unique_items"),
        pl.col("quantity").sum().alias("total_quantity"),
    ])
    return customer_lf.select([
        pl.col("n_bills").mean().alias("avg_bills"),
        pl.col("n_bills").median().alias("median_bills"),
        pl.col("n_bills").quantile(0.95).alias("p95_bills"),
    ]).collect()


def top_item_popularity(train_hist_lf, n=20):
    return train_hist_lf.group_by("item_id").agg([
        pl.len().alias("n_transactions"),
        pl.col("customer_id").n_unique().alias("n_customers"),
        pl.col("quantity").sum().alias("total_quantity"),
    ]).sort("n_transactions", descending=True).head(n).collect()


def repeat_stats(train_hist_lf, n=20):
    user_item_repeat = train_hist_lf.group_by(["customer_id", "item_id"]).agg([
        pl.col("bill_id").n_unique().alias("n_bills")
    ])
    return user_item_repeat.group_by("item_id").agg([
        (pl.col("n_bills") > 1).mean().alias("repeat_rate")
    ]).sort("repeat_rate", descending=True).head(n).collect()
