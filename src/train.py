import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train_lgbm(train_model_path, model_path, feature_columns_path=None, importance_path=None):
    """Step 8 từ notebook: train LightGBM binary baseline."""
    train_df = pd.read_parquet(train_model_path)
    y = train_df["target"]
    X = train_df.drop(columns=["target"])
    feature_cols = list(X.columns)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y,
    )

    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_holdout, y_holdout)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

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
