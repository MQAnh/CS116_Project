import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.logging_utils import log_step


def parquet_files(path):
    path = Path(path)
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    return [path]


def parquet_num_rows(files):
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def load_training_sample(
    train_model_path,
    batch_size=500_000,
    negative_ratio=3.0,
    positive_fraction=1.0,
    max_rows=3_000_000,
    random_state=42,
):
    files = parquet_files(train_model_path)
    total_rows = parquet_num_rows(files)
    sampled_batches = []
    sampled_rows = 0
    read_rows = 0
    batch_idx = 0

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_idx += 1
            batch_df = batch.to_pandas()
            read_rows += len(batch_df)
            pos_df = batch_df[batch_df["target"] == 1]
            neg_df = batch_df[batch_df["target"] == 0]

            if positive_fraction < 1.0 and len(pos_df) > 0:
                pos_df = pos_df.sample(frac=positive_fraction, random_state=random_state + batch_idx)

            n_neg = min(len(neg_df), int(np.ceil(len(pos_df) * negative_ratio)))
            if n_neg > 0:
                neg_df = neg_df.sample(n=n_neg, random_state=random_state + batch_idx)
            else:
                neg_df = neg_df.iloc[0:0]

            sample_df = pd.concat([pos_df, neg_df], ignore_index=True)
            if len(sample_df) == 0:
                continue

            remaining_rows = max_rows - sampled_rows
            if len(sample_df) > remaining_rows:
                sample_df = sample_df.sample(n=remaining_rows, random_state=random_state + batch_idx)

            sampled_batches.append(sample_df)
            sampled_rows += len(sample_df)
            log_step(
                "train sample batch "
                f"{batch_idx}: read {read_rows:,}/{total_rows:,} rows, "
                f"kept {sampled_rows:,}/{max_rows:,} sampled rows"
            )

            if sampled_rows >= max_rows:
                break
        if sampled_rows >= max_rows:
            break

    if not sampled_batches:
        raise ValueError("No training rows were sampled.")

    train_df = pd.concat(sampled_batches, ignore_index=True)
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    log_step(
        "training sample ready: "
        f"rows={len(train_df):,}, positives={int(train_df['target'].sum()):,}, "
        f"negatives={int((train_df['target'] == 0).sum()):,}"
    )
    return train_df


def train_lgbm(
    train_model_path,
    model_path,
    feature_columns_path=None,
    importance_path=None,
    train_batch_size=500_000,
    negative_ratio=3.0,
    positive_fraction=1.0,
    max_train_rows=3_000_000,
    categorical_features=None,
    popular_negative_weight_column=None,
    popular_negative_weight_alpha=0.0,
):
    """Step 8 từ notebook: train LightGBM binary baseline."""
    train_df = load_training_sample(
        train_model_path,
        batch_size=train_batch_size,
        negative_ratio=negative_ratio,
        positive_fraction=positive_fraction,
        max_rows=max_train_rows,
    )
    y = train_df["target"]
    X = train_df.drop(columns=["target"])
    feature_cols = list(X.columns)
    categorical_features = [] if categorical_features is None else [
        c for c in categorical_features
        if c in feature_cols
    ]
    sample_weight = build_popular_negative_weights(
        X,
        y,
        column=popular_negative_weight_column,
        alpha=popular_negative_weight_alpha,
    )

    if sample_weight is not None:
        X_train, X_holdout, y_train, y_holdout, w_train, _ = train_test_split(
            X,
            y,
            sample_weight,
            test_size=0.1,
            random_state=42,
            stratify=y,
        )
    else:
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X,
            y,
            test_size=0.1,
            random_state=42,
            stratify=y,
        )
        w_train = None

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        scale_pos_weight=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_holdout, y_holdout)],
        "eval_metric": "auc",
        "callbacks": [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50),
        ],
    }
    if w_train is not None:
        fit_kwargs["sample_weight"] = w_train
    if categorical_features:
        fit_kwargs["categorical_feature"] = categorical_features

    model.fit(**fit_kwargs)

    pred_holdout = model.predict_proba(X_holdout)[:, 1]
    auc = roc_auc_score(y_holdout, pred_holdout)
    print("Holdout AUC:", auc)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    if feature_columns_path is not None:
        joblib.dump(feature_cols, feature_columns_path)

    if importance_path is not None:
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(importance_path, index=False)

    return model, feature_cols, auc


def build_popular_negative_weights(X, y, column=None, alpha=0.0):
    if not column or alpha <= 0 or column not in X.columns:
        return None

    popularity = pd.to_numeric(X[column], errors="coerce").fillna(0).clip(lower=0)
    logged = np.log1p(popularity)
    max_value = logged.max()
    if max_value <= 0:
        return None

    normalized = logged / max_value
    weights = np.ones(len(X), dtype=np.float32)
    negative_mask = (y.to_numpy() == 0)
    weights[negative_mask] += alpha * normalized.to_numpy(dtype=np.float32)[negative_mask]
    log_step(
        "popular negative weights enabled: "
        f"column={column}, alpha={alpha}, "
        f"min={weights.min():.3f}, mean={weights.mean():.3f}, max={weights.max():.3f}"
    )
    return weights
