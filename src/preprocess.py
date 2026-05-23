import polars as pl


def infer_numeric_cols(features_lf, drop_cols, cat_cols):
    return [
        c for c, dtype in features_lf.collect_schema().items()
        if c not in drop_cols + cat_cols
    ]


def build_category_mappings(train_features_lf, cat_cols):
    mappings = {}
    for c in cat_cols:
        values = (
            train_features_lf
            .select(pl.col(c).cast(pl.Utf8).fill_null("unknown").unique().sort())
            .collect()
            .get_column(c)
            .to_list()
        )
        mappings[c] = {value: idx + 1 for idx, value in enumerate(values)}
        mappings[c]["unknown"] = 0
    return mappings


def encode_categorical_exprs(cat_cols, mappings):
    return [
        (
            pl.col(c)
            .cast(pl.Utf8)
            .fill_null("unknown")
            .replace_strict(mappings[c], default=0)
            .cast(pl.Int32)
            .alias(c)
        )
        for c in cat_cols
    ]


def fill_numeric_exprs(numeric_cols):
    return [pl.col(c).fill_null(0) for c in numeric_cols]


def prepare_train_matrix(train_features_lf, drop_cols, cat_cols):
    numeric_cols = infer_numeric_cols(train_features_lf, drop_cols, cat_cols)
    category_mappings = build_category_mappings(train_features_lf, cat_cols)
    train_model_lf = (
        train_features_lf
        .with_columns(encode_categorical_exprs(cat_cols, category_mappings))
        .with_columns(fill_numeric_exprs(numeric_cols))
        .select(["target"] + numeric_cols + cat_cols)
    )
    feature_cols = numeric_cols + cat_cols
    return train_model_lf, feature_cols


def prepare_valid_matrix(valid_features_lf, train_features_lf, drop_cols, cat_cols):
    numeric_cols = infer_numeric_cols(train_features_lf, drop_cols, cat_cols)
    category_mappings = build_category_mappings(train_features_lf, cat_cols)
    valid_model_lf = (
        valid_features_lf
        .with_columns(encode_categorical_exprs(cat_cols, category_mappings))
        .with_columns(fill_numeric_exprs(numeric_cols))
        .select(["customer_id", "item_id", "target"] + numeric_cols + cat_cols)
    )
    feature_cols = numeric_cols + cat_cols
    return valid_model_lf, feature_cols


def prepare_inference_matrix(features_lf, train_features_lf, drop_cols, cat_cols):
    numeric_cols = infer_numeric_cols(train_features_lf, drop_cols, cat_cols)
    category_mappings = build_category_mappings(train_features_lf, cat_cols)
    model_lf = (
        features_lf
        .with_columns(encode_categorical_exprs(cat_cols, category_mappings))
        .with_columns(fill_numeric_exprs(numeric_cols))
        .select(["customer_id", "item_id"] + numeric_cols + cat_cols)
    )
    feature_cols = numeric_cols + cat_cols
    return model_lf, feature_cols
