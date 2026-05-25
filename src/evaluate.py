import pickle
import heapq

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

from src.logging_utils import log_step


def parquet_files(path):
    path = Path(path)
    if path.is_dir():
        return sorted(path.glob("*.parquet"))
    return [path]


def parquet_num_rows(files):
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def predict_matrix(model_ready_path, model_path, feature_columns_path=None, id_cols=None):
    df = pd.read_parquet(model_ready_path)
    id_cols = ["customer_id", "item_id", "target"] if id_cols is None else id_cols
    drop_cols = [c for c in id_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    if feature_columns_path is not None:
        feature_cols = joblib.load(feature_columns_path)
        X = X[feature_cols]

    model = joblib.load(model_path)
    df["score"] = model.predict_proba(X)[:, 1]
    return df


def predict_valid(valid_model_path, model_path, feature_columns_path=None):
    return predict_matrix(
        valid_model_path,
        model_path,
        feature_columns_path=feature_columns_path,
        id_cols=["customer_id", "item_id", "target"],
    )


def get_topk(df, k=10):
    return (
        df
        .sort_values(["customer_id", "score"], ascending=[True, False])
        .groupby("customer_id")
        .head(k)
    )


def predict_topk_from_parquet(
    model_ready_path,
    model_path,
    feature_columns_path=None,
    id_cols=None,
    k=10,
    batch_size=500_000,
    log_every=1,
    repeat_boost=0.04,
    affinity_boost=0.03,
    popularity_penalty=0.03,
):
    model = joblib.load(model_path)
    feature_cols = joblib.load(feature_columns_path) if feature_columns_path is not None else None
    id_cols = ["customer_id", "item_id", "target"] if id_cols is None else id_cols

    files = parquet_files(model_ready_path)
    total_rows = parquet_num_rows(files)
    schema_cols = set(pq.ParquetFile(files[0]).schema_arrow.names)
    read_cols = None
    if feature_cols is not None:
        read_cols = [c for c in id_cols if c in schema_cols] + feature_cols
    topk_by_user = {}
    processed_rows = 0
    kept_rows = 0
    sequence = 0
    batch_idx = 0

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_cols):
            batch_idx += 1
            batch_df = batch.to_pandas()
            processed_rows += len(batch_df)
            drop_cols = [c for c in id_cols if c in batch_df.columns]
            X = batch_df.drop(columns=drop_cols)

            if feature_cols is not None:
                X = X[feature_cols]

            scores = model.predict_proba(X)[:, 1]
            adjusted_scores = adjust_prediction_scores(
                scores,
                batch_df,
                repeat_boost=repeat_boost,
                affinity_boost=affinity_boost,
                popularity_penalty=popularity_penalty,
            )
            customer_ids = batch_df["customer_id"].to_numpy()
            item_ids = batch_df["item_id"].to_numpy()
            targets = batch_df["target"].to_numpy() if "target" in batch_df.columns else None

            for row_idx, score in enumerate(adjusted_scores):
                customer_id = int(customer_ids[row_idx])
                item_id = item_ids[row_idx]
                target = int(targets[row_idx]) if targets is not None else None
                heap = topk_by_user.setdefault(customer_id, [])
                entry = (float(score), sequence, item_id, target)
                sequence += 1

                if len(heap) < k:
                    heapq.heappush(heap, entry)
                    kept_rows += 1
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, entry)

            if log_every and batch_idx % log_every == 0:
                log_step(
                    "predict batch "
                    f"{batch_idx}: {processed_rows:,}/{total_rows:,} rows, "
                    f"kept {kept_rows:,} top-k rows for {len(topk_by_user):,} users"
                )

    if not topk_by_user:
        return pd.DataFrame(columns=[c for c in id_cols if c in ["customer_id", "item_id", "target"]] + ["score"])

    records = []
    has_target = "target" in id_cols
    for customer_id, heap in topk_by_user.items():
        for score, _, item_id, target in sorted(heap, key=lambda row: row[0], reverse=True):
            record = {
                "customer_id": customer_id,
                "item_id": item_id,
                "score": score,
            }
            if has_target:
                record["target"] = target
            records.append(record)

    column_order = [c for c in id_cols if c in ["customer_id", "item_id", "target"]] + ["score"]
    return pd.DataFrame.from_records(records, columns=column_order)


def normalized_log_feature(df, column):
    if column not in df.columns:
        return 0
    values = pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)
    logged = pd.Series(np.log1p(values), index=df.index)
    max_value = logged.max()
    if max_value <= 0:
        return 0
    return logged / max_value


def adjust_prediction_scores(
    scores,
    batch_df,
    repeat_boost=0.04,
    affinity_boost=0.03,
    popularity_penalty=0.03,
):
    adjusted = pd.Series(scores, index=batch_df.index, dtype="float64")

    if "ui_n_transactions" in batch_df.columns:
        adjusted += repeat_boost * (batch_df["ui_n_transactions"].fillna(0) > 0).astype(float)

    if "uc_transaction_share" in batch_df.columns:
        adjusted += affinity_boost * batch_df["uc_transaction_share"].fillna(0).clip(lower=0, upper=1)

    if "ub_transaction_share" in batch_df.columns:
        adjusted += affinity_boost * batch_df["ub_transaction_share"].fillna(0).clip(lower=0, upper=1)

    pop_norm = normalized_log_feature(batch_df, "item_n_customers")
    if not isinstance(pop_norm, int):
        adjusted -= popularity_penalty * pop_norm

    return adjusted.to_numpy()


def precision_at_k(topk_df, k=10):
    return topk_df.groupby("customer_id")["target"].sum().mean() / k


def precision_at_k_buyers_only(topk_df, valid_label_lf, k=10):
    valid_buyers = set(
        valid_label_lf
        .select("customer_id")
        .unique()
        .collect()["customer_id"]
    )
    topk_eval = topk_df[topk_df["customer_id"].isin(valid_buyers)]
    return precision_at_k(topk_eval, k=k)


def ground_truth_to_dict(label_lf):
    gt_df = (
        label_lf
        .select(["customer_id", "item_id"])
        .unique()
        .group_by("customer_id")
        .agg("item_id")
        .collect()
    )
    return {
        row["customer_id"]: row["item_id"]
        for row in gt_df.iter_rows(named=True)
    }


def server_precision_at_k(submission, answer, k=10, scale=100):
    precisions = []

    for customer_id, true_items in answer.items():
        pred_items = submission.get(customer_id, [])
        pred_topk = pred_items[:k]
        true_set = set(true_items)
        hits = sum(1 for item in pred_topk if item in true_set)
        precisions.append(hits / float(k))

    if not precisions:
        return 0.0

    return (sum(precisions) / len(precisions)) * scale

def topk_to_submission_dict(topk_df, k=10, user_ids=None, fallback_items=None):
    """
    Convert dataframe top-k prediction thành:

    {
        customer_id: [item1, item2, ...]
    }
    """

    submission = {}
    rows = (
        topk_df.iter_rows(named=True)
        if hasattr(topk_df, "iter_rows")
        else (
            {"customer_id": row.customer_id, "item_id": row.item_id}
            for row in topk_df[["customer_id", "item_id"]].itertuples(index=False)
        )
    )

    for row in rows:
        customer_id = row["customer_id"]
        item_id = row["item_id"]

        if customer_id not in submission:
            submission[customer_id] = []

        if item_id not in submission[customer_id] and len(submission[customer_id]) < k:
            submission[customer_id].append(item_id)

    if user_ids is not None:
        fallback_items = [] if fallback_items is None else list(fallback_items)
        for customer_id in user_ids:
            items = submission.setdefault(customer_id, [])
            for item_id in fallback_items:
                if len(items) >= k:
                    break
                if item_id not in items:
                    items.append(item_id)

    return submission


def save_submission_pickle(submission_dict, output_path):
    """
    Save submission dict thành pickle file
    """

    with open(output_path, "wb") as f:
        pickle.dump(submission_dict, f)

    print(f"Saved submission to: {output_path}")
