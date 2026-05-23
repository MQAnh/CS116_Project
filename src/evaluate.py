import joblib
import pandas as pd


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


import pickle


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
        else topk_df[["customer_id", "item_id"]].to_dict("records")
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
