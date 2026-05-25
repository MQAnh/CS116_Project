import shutil

import joblib
import polars as pl

from src.logging_utils import log_step


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


def scan_parquet_dir(path):
    return pl.scan_parquet(str(path / "*.parquet"))


def reset_output_dir(output_dir):
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_matrix_lf(features_lf, numeric_cols, cat_cols, category_mappings, id_cols):
    return (
        features_lf
        .with_columns(encode_categorical_exprs(cat_cols, category_mappings))
        .with_columns(fill_numeric_exprs(numeric_cols))
        .select(id_cols + numeric_cols + cat_cols)
    )


def save_preprocess_metadata(metadata_path, numeric_cols, cat_cols, category_mappings):
    if metadata_path is None:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "category_mappings": category_mappings,
    }, metadata_path)


def load_preprocess_metadata(metadata_path):
    return joblib.load(metadata_path)


def get_preprocess_spec(train_features_dir, drop_cols, cat_cols, metadata_path=None):
    if metadata_path is not None and metadata_path.exists():
        metadata = load_preprocess_metadata(metadata_path)
        return (
            metadata["numeric_cols"],
            metadata["cat_cols"],
            metadata["category_mappings"],
        )

    train_features_lf = scan_parquet_dir(train_features_dir)
    numeric_cols = infer_numeric_cols(train_features_lf, drop_cols, cat_cols)
    category_mappings = build_category_mappings(train_features_lf, cat_cols)
    return numeric_cols, cat_cols, category_mappings


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


def prepare_train_matrix_chunked(train_features_dir, output_dir, drop_cols, cat_cols, metadata_path=None):
    all_train_features_lf = scan_parquet_dir(train_features_dir)
    numeric_cols = infer_numeric_cols(all_train_features_lf, drop_cols, cat_cols)
    category_mappings = build_category_mappings(all_train_features_lf, cat_cols)
    save_preprocess_metadata(metadata_path, numeric_cols, cat_cols, category_mappings)
    output_dir = reset_output_dir(output_dir)

    for idx, part_path in enumerate(sorted(train_features_dir.glob("*.parquet")), start=1):
        log_step(f"prepare train matrix chunk {idx}: {part_path.name}")
        features_lf = pl.scan_parquet(part_path)
        model_lf = prepare_matrix_lf(
            features_lf,
            numeric_cols,
            cat_cols,
            category_mappings,
            ["target"],
        )
        model_lf.sink_parquet(output_dir / part_path.name)

    feature_cols = numeric_cols + cat_cols
    return scan_parquet_dir(output_dir), feature_cols


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


def prepare_valid_matrix_chunked(
    valid_features_dir,
    train_features_dir,
    output_dir,
    drop_cols,
    cat_cols,
    metadata_path=None,
):
    numeric_cols, cat_cols, category_mappings = get_preprocess_spec(
        train_features_dir,
        drop_cols,
        cat_cols,
        metadata_path=metadata_path,
    )
    output_dir = reset_output_dir(output_dir)

    for idx, part_path in enumerate(sorted(valid_features_dir.glob("*.parquet")), start=1):
        log_step(f"prepare validation matrix chunk {idx}: {part_path.name}")
        features_lf = pl.scan_parquet(part_path)
        model_lf = prepare_matrix_lf(
            features_lf,
            numeric_cols,
            cat_cols,
            category_mappings,
            ["customer_id", "item_id", "target"],
        )
        model_lf.sink_parquet(output_dir / part_path.name)

    feature_cols = numeric_cols + cat_cols
    return scan_parquet_dir(output_dir), feature_cols


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


def prepare_inference_matrix_chunked(
    features_dir,
    train_features_dir,
    output_dir,
    drop_cols,
    cat_cols,
    metadata_path=None,
):
    numeric_cols, cat_cols, category_mappings = get_preprocess_spec(
        train_features_dir,
        drop_cols,
        cat_cols,
        metadata_path=metadata_path,
    )
    output_dir = reset_output_dir(output_dir)

    for idx, part_path in enumerate(sorted(features_dir.glob("*.parquet")), start=1):
        log_step(f"prepare inference matrix chunk {idx}: {part_path.name}")
        features_lf = pl.scan_parquet(part_path)
        model_lf = prepare_matrix_lf(
            features_lf,
            numeric_cols,
            cat_cols,
            category_mappings,
            ["customer_id", "item_id"],
        )
        model_lf.sink_parquet(output_dir / part_path.name)

    feature_cols = numeric_cols + cat_cols
    return scan_parquet_dir(output_dir), feature_cols
