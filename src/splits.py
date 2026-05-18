import polars as pl


def make_time_splits(
    transactions_lf,
    train_history_months=(1, 9),
    train_label_month=10,
    valid_history_months=(1, 10),
    valid_label_month=12,
    final_history_months=(1, 11),
):
    """Step 2 từ notebook: chia train/valid/final theo tháng."""
    train_hist_lf = transactions_lf.filter(pl.col("month").is_between(*train_history_months))
    train_label_lf = transactions_lf.filter(pl.col("month") == train_label_month)

    valid_hist_lf = transactions_lf.filter(pl.col("month").is_between(*valid_history_months))
    valid_label_lf = transactions_lf.filter(pl.col("month") == valid_label_month)

    final_hist_lf = transactions_lf.filter(pl.col("month").is_between(*final_history_months))
    test_target_lf = transactions_lf.filter(pl.col("month") == valid_label_month)

    return {
        "train_hist_lf": train_hist_lf,
        "train_label_lf": train_label_lf,
        "valid_hist_lf": valid_hist_lf,
        "valid_label_lf": valid_label_lf,
        "final_hist_lf": final_hist_lf,
        "test_target_lf": test_target_lf,
    }
