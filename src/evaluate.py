import joblib
import pandas as pd


def predict_valid(valid_model_path, model_path, feature_columns_path=None):
    valid_df = pd.read_parquet(valid_model_path)
    X_valid = valid_df.drop(columns=["customer_id", "item_id", "target"])

    if feature_columns_path is not None:
        feature_cols = joblib.load(feature_columns_path)
        X_valid = X_valid[feature_cols]

    model = joblib.load(model_path)
    valid_df["score"] = model.predict_proba(X_valid)[:, 1]
    return valid_df


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


import pickle


def topk_to_submission_dict(topk_df):
    """
    Convert dataframe top-k prediction thành:

    {
        customer_id: [item1, item2, ...]
    }
    """

    submission = {}

    for row in topk_df.iter_rows(named=True):
        customer_id = row["customer_id"]
        item_id = row["item_id"]

        if customer_id not in submission:
            submission[customer_id] = []

        submission[customer_id].append(item_id)

    return submission


def save_submission_pickle(submission_dict, output_path):
    """
    Save submission dict thành pickle file
    """

    with open(output_path, "wb") as f:
        pickle.dump(submission_dict, f)

    print(f"Saved submission to: {output_path}")