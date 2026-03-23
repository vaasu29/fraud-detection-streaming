import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from src.preprocess import load_data, preprocess, save_scaler

def train():
    # ── Load & preprocess ──────────────────────────────────────────
    df = load_data("creditcard.csv")
    X, y, scaler = preprocess(df)
    save_scaler(scaler)

    # ── Train/test split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    # ── SMOTE — handle severe class imbalance ──────────────────────
    print("Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Class distribution: {pd.Series(y_train_bal).value_counts().to_dict()}")

    # ── MLflow experiment ──────────────────────────────────────────
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="xgboost-smote-v1"):

        # ── Model params ───────────────────────────────────────────
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 1,  # SMOTE handles balance
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
        }

        # ── Train ──────────────────────────────────────────────────
        model = XGBClassifier(**params)
        model.fit(X_train_bal, y_train_bal,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        # ── Evaluate ───────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_prob),
        }

        print("\n── Model Metrics ──────────────────────────")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # ── Log to MLflow ──────────────────────────────────────────
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model,
            artifact_path="fraud_model",
            registered_model_name="fraud-detection-xgboost"
        )

        # ── Save model locally ─────────────────────────────────────
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fraud_model.pkl")
        print("\nModel saved to models/fraud_model.pkl")

        run_id = mlflow.active_run().info.run_id
        print(f"MLflow Run ID: {run_id}")
        print("Training complete!")

if __name__ == "__main__":
    train()
