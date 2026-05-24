import polars as pl


def make_ground_truth(label_lf):
    return (
        label_lf
        .select(["customer_id", "item_id"])
        .unique()
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("target"))
    )


def make_labeled_dataset(candidates_lf, gt_lf):
    candidate_cols = candidates_lf.collect_schema().names()
    passthrough_cols = [
        c for c in candidate_cols
        if c not in ["customer_id", "item_id", "target"]
    ]
    return (
        candidates_lf
        .select([
            pl.col("customer_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.String),
            *[pl.col(c) for c in passthrough_cols],
        ])
        .join(
            gt_lf.select([
                pl.col("customer_id").cast(pl.Int32),
                pl.col("item_id").cast(pl.String),
                pl.col("target").cast(pl.Int8),
            ]),
            on=["customer_id", "item_id"],
            how="left",
        )
        .with_columns(pl.col("target").fill_null(pl.lit(0, dtype=pl.Int8)).cast(pl.Int8))
    )
