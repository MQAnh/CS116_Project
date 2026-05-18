import polars as pl


def infer_numeric_cols(features_lf, drop_cols, cat_cols):
    return [
        c for c, dtype in features_lf.collect_schema().items()
        if c not in drop_cols + cat_cols
    ]


def prepare_train_matrix(train_features_lf, drop_cols, cat_cols):
    numeric_cols = infer_numeric_cols(train_features_lf, drop_cols, cat_cols)
    train_model_lf = (
        train_features_lf
        .with_columns([
            pl.col(c).fill_null("unknown").cast(pl.Categorical).to_physical().alias(c)
            for c in cat_cols
        ])
        .with_columns([
            pl.col(c).fill_null(0)
            for c in numeric_cols
        ])
        .select(["target"] + numeric_cols + cat_cols)
    )
    feature_cols = numeric_cols + cat_cols
    return train_model_lf, feature_cols


def prepare_valid_matrix(valid_features_lf, train_features_lf, drop_cols, cat_cols):
    numeric_cols = infer_numeric_cols(train_features_lf, drop_cols, cat_cols)
    valid_model_lf = (
        valid_features_lf
        .with_columns([
            pl.col(c).fill_null("unknown").cast(pl.Categorical).to_physical().alias(c)
            for c in cat_cols
        ])
        .with_columns([
            pl.col(c).fill_null(0)
            for c in numeric_cols
        ])
        .select(["customer_id", "item_id", "target"] + numeric_cols + cat_cols)
    )
    feature_cols = numeric_cols + cat_cols
    return valid_model_lf, feature_cols
