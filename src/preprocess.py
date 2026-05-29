import shutil
from pathlib import Path

import joblib
import polars as pl

from src.logging_utils import log_step


def infer_numeric_cols(features_lf, drop_cols, cat_cols):
    numeric_cols, _ = infer_feature_columns(features_lf, drop_cols, cat_cols)
    return numeric_cols


def infer_feature_columns(features_lf, drop_cols, cat_cols, selected_features=None):
    schema = features_lf.collect_schema()
    selected = set(selected_features) if selected_features else None
    cat_col_set = set(cat_cols)

    active_cat_cols = [
        c for c in cat_cols
        if c in schema and (selected is None or c in selected)
    ]
    numeric_cols = [
        c for c, dtype in schema.items()
        if c not in drop_cols
        and c not in cat_col_set
        and (selected is None or c in selected)
    ]
    return numeric_cols, active_cat_cols


def selected_features_for_metadata(selected_features):
    if selected_features is None:
        return None
    return [
        str(feature)
        for feature in selected_features
    ]


def inference_selected_features(feature_columns_path, fallback_selected_features=None):
    feature_columns_path = Path(feature_columns_path)
    if feature_columns_path.exists():
        return joblib.load(feature_columns_path)
    return fallback_selected_features


def assert_no_blocked_features(feature_cols, blocked_features, label="feature columns"):
    blocked = sorted(set(feature_cols or []) & set(blocked_features or []))
    if blocked:
        blocked_preview = ", ".join(blocked[:20])
        if len(blocked) > 20:
            blocked_preview += f", ... (+{len(blocked) - 20} more)"
        raise ValueError(
            f"{label} still contains disabled popular features: {blocked_preview}. "
            "Retrain the model after clearing old feature metadata."
        )


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
    path = Path(path)
    if path.is_file():
        return pl.scan_parquet(path)

    files = sorted(path.glob("*.parquet")) if path.exists() else []
    if files:
        return pl.scan_parquet([str(file_path) for file_path in files])

    if path.name.endswith("_chunks"):
        fallback_path = path.with_name(f"{path.name.removesuffix('_chunks')}.parquet")
        if fallback_path.exists():
            return pl.scan_parquet(fallback_path)

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


def save_preprocess_metadata(
    metadata_path,
    numeric_cols,
    cat_cols,
    category_mappings,
    selected_features=None,
):
    if metadata_path is None:
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "category_mappings": category_mappings,
        "selected_features": selected_features_for_metadata(selected_features),
    }, metadata_path)


def load_preprocess_metadata(metadata_path):
    return joblib.load(metadata_path)


def get_preprocess_spec(
    train_features_dir,
    drop_cols,
    cat_cols,
    metadata_path=None,
    selected_features=None,
):
    if metadata_path is not None and metadata_path.exists():
        metadata = load_preprocess_metadata(metadata_path)
        expected_selected_features = selected_features_for_metadata(selected_features)
        metadata_selected_features = metadata.get("selected_features")
        metadata_matches_selection = (
            expected_selected_features is None
            or (
                metadata_selected_features is not None
                and set(metadata_selected_features) == set(expected_selected_features)
            )
        )
        if metadata_matches_selection:
            return (
                metadata["numeric_cols"],
                metadata["cat_cols"],
                metadata["category_mappings"],
            )

    train_features_lf = scan_parquet_dir(train_features_dir)
    numeric_cols, cat_cols = infer_feature_columns(
        train_features_lf,
        drop_cols,
        cat_cols,
        selected_features=selected_features,
    )
    category_mappings = build_category_mappings(train_features_lf, cat_cols)
    return numeric_cols, cat_cols, category_mappings


def prepare_train_matrix(train_features_lf, drop_cols, cat_cols, selected_features=None):
    numeric_cols, cat_cols = infer_feature_columns(
        train_features_lf,
        drop_cols,
        cat_cols,
        selected_features=selected_features,
    )
    category_mappings = build_category_mappings(train_features_lf, cat_cols)
    train_model_lf = (
        train_features_lf
        .with_columns(encode_categorical_exprs(cat_cols, category_mappings))
        .with_columns(fill_numeric_exprs(numeric_cols))
        .select(["target"] + numeric_cols + cat_cols)
    )
    feature_cols = numeric_cols + cat_cols
    return train_model_lf, feature_cols


def prepare_train_matrix_chunked(
    train_features_dir,
    output_dir,
    drop_cols,
    cat_cols,
    metadata_path=None,
    selected_features=None,
):
    all_train_features_lf = scan_parquet_dir(train_features_dir)
    numeric_cols, cat_cols = infer_feature_columns(
        all_train_features_lf,
        drop_cols,
        cat_cols,
        selected_features=selected_features,
    )
    category_mappings = build_category_mappings(all_train_features_lf, cat_cols)
    save_preprocess_metadata(
        metadata_path,
        numeric_cols,
        cat_cols,
        category_mappings,
        selected_features=selected_features,
    )
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


def prepare_valid_matrix(
    valid_features_lf,
    train_features_lf,
    drop_cols,
    cat_cols,
    selected_features=None,
):
    numeric_cols, cat_cols = infer_feature_columns(
        train_features_lf,
        drop_cols,
        cat_cols,
        selected_features=selected_features,
    )
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
    selected_features=None,
):
    numeric_cols, cat_cols, category_mappings = get_preprocess_spec(
        train_features_dir,
        drop_cols,
        cat_cols,
        metadata_path=metadata_path,
        selected_features=selected_features,
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


def prepare_inference_matrix(
    features_lf,
    train_features_lf,
    drop_cols,
    cat_cols,
    selected_features=None,
):
    numeric_cols, cat_cols = infer_feature_columns(
        train_features_lf,
        drop_cols,
        cat_cols,
        selected_features=selected_features,
    )
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
    selected_features=None,
):
    numeric_cols, cat_cols, category_mappings = get_preprocess_spec(
        train_features_dir,
        drop_cols,
        cat_cols,
        metadata_path=metadata_path,
        selected_features=selected_features,
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
