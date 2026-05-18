import polars as pl


def load_data(transactions_path, items_path):
    """Step 0-1 từ notebook: đọc parquet và chuẩn hóa kiểu dữ liệu."""
    transactions_lf = pl.scan_parquet(transactions_path)
    items_lf = pl.scan_parquet(items_path)

    transactions_lf = transactions_lf.with_columns([
        pl.col("updated_date").dt.date().alias("date"),
        pl.col("updated_date").dt.month().alias("month"),
        pl.col("updated_date").dt.year().alias("year"),
        pl.col("price").cast(pl.Float32),
        pl.col("discount").cast(pl.Float32),
    ])

    items_lf = items_lf.with_columns([
        pl.col("price").cast(pl.Float32),
    ])

    return transactions_lf, items_lf


def print_schema(transactions_lf, items_lf):
    print("Transactions schema:")
    print(transactions_lf.collect_schema())
    print("Items schema:")
    print(items_lf.collect_schema())
