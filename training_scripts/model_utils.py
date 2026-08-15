import os
import json
import sys
import uuid
import glob
import time
from datetime import datetime
from typing import Optional, Union, Tuple
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

CURRENT_DIR = Path(__file__).resolve().parent

SAVED_MODELS_DIR = CURRENT_DIR.parent / "models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def calculate_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Calculates the 6 required classification evaluation metrics:
    1. Accuracy
    2. AUC Score
    3. Precision
    4. Recall
    5. F1 Score
    6. Matthews Correlation Coefficient (MCC Score)
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        "mcc_score": float(matthews_corrcoef(y_true, y_pred))
    }

    # Calculate ROC-AUC Score if probabilities are available
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 2:
                prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                metrics["auc_score"] = float(roc_auc_score(y_true, prob))
            else:
                metrics["auc_score"] = float(roc_auc_score(y_true, y_prob, multi_class='ovr'))
        except Exception:
            metrics["auc_score"] = None
    else:
        metrics["auc_score"] = None

    return metrics


def get_total_parameters(model) -> int:
    """Estimates parameter counts for Scikit-Learn classifiers."""
    if hasattr(model, "coef_"):
        return int(model.coef_.size + (model.intercept_.size if hasattr(model, "intercept_") else 0))
    elif hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    elif hasattr(model, "estimators_"):
        return int(sum(est.tree_.node_count for est in model.estimators_))
    elif hasattr(model, "_fit_X"):
        return int(model._fit_X.size)
    elif hasattr(model, "theta_"):
        return int(model.theta_.size + model.var_.size)
    elif hasattr(model, "feature_log_prob_"):
        return int(model.feature_log_prob_.size)
    return 0


def train_and_evaluate(model, model_name: str, train_x, train_y, test_x, test_y, verbose: bool = True) -> dict:
    """
    Functionality (i): Trains model, computes the 6 evaluation metrics,
    prints detailed logs, and saves model artifacts with a 4-digit UUID.
    """
    unique_id = str(uuid.uuid4().hex[:4])
    start_dt = datetime.now()
    start_time = time.time()

    if verbose:
        print("\n" + "=" * 80)
        print(f"🚀 [START] Training Pipeline for Model: '{model_name}' | ID: [{unique_id}]")
        print(f"⏰ Start Time: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        train_shape = train_x.shape if hasattr(train_x, 'shape') else (len(train_x),)
        test_shape = test_x.shape if hasattr(test_x, 'shape') else (len(test_x),)
        print(f"📊 Dataset Info -> Train set features: {train_shape} | Test set features: {test_shape}")
        print(f"⏳ Fitting model onto training data...")

    # 1. Fit Model
    model.fit(train_x, train_y)

    end_time = time.time()
    end_dt = datetime.now()
    duration = end_time - start_time

    # 2. Predictions & Probabilities
    preds = model.predict(test_x)
    probs = model.predict_proba(test_x) if hasattr(model, "predict_proba") else None

    # 3. Calculate 6 Required Metrics & Parameter Count
    metrics = calculate_metrics(test_y, preds, probs)
    param_count = get_total_parameters(model)

    # 4. Build Evaluation Report
    evaluation_report = {
        "model_id": unique_id,
        "model_name": model_name,
        "model_params": model.get_params(),
        "total_parameters": param_count,
        "training_start": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "training_end": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_time_seconds": round(duration, 4),
        "metrics": metrics
    }

    if verbose:
        print(f"✅ Training completed in {duration:.4f} seconds!")
        print(f"🧮 Estimated Total Model Parameters: {param_count:,}")
        print("\n📋 --- Training Evaluation Summary ---")
        for metric_name, val in metrics.items():
            val_str = f"{val:.4f}" if isinstance(val, (float, int)) and val is not None else "N/A"
            print(f"  • {metric_name.upper():<12}: {val_str}")

    # 5. Save Artifacts
    model_filename = f"{model_name}_{unique_id}.joblib"
    json_filename = f"{model_name}_{unique_id}.json"

    joblib.dump(model, os.path.join(SAVED_MODELS_DIR, model_filename))
    with open(os.path.join(SAVED_MODELS_DIR, json_filename), "w") as f:
        json.dump(evaluation_report, f, indent=4)

    if verbose:
        print(f"\n💾 Saved model binary : {model_filename}")
        print(f"💾 Saved metrics JSON : {json_filename}")
        print(f"🏁 [FINISHED] Execution complete for [{model_name}_{unique_id}]")
        print("=" * 80 + "\n")

    return evaluation_report


def predict_with_best_model(
        test_x_df: pd.DataFrame,
        raw_features_df: Optional[pd.DataFrame] = None,
        test_y: Optional[pd.Series] = None,
        model_name: Optional[str] = None,
        metric_key: str = "accuracy",
        target_name: str = "predicted_diagnosis",
        save_output: bool = True,
        custom_id: Optional[str] = None
) -> dict:
    """
    Runs predictions using test_x_df (scaled/preprocessed features), but returns
    the output dictionary mapped onto raw_features_df (original unscaled features)
    if provided.
    """
    json_files = glob.glob(os.path.join(SAVED_MODELS_DIR, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"No trained model evaluation JSON files found in '{SAVED_MODELS_DIR}' directory.")

    best_score = -float("inf")
    best_model_file = None
    best_model_info = None

    # Find Best Model
    for jf in json_files:
        with open(jf, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

            saved_model_name = data.get("model_name")

            if model_name is not None and saved_model_name != model_name:
                continue

            score = data.get("metrics", {}).get(metric_key)
            if score is not None and score > best_score:
                best_score = score
                best_model_info = data
                m_id = data["model_id"]
                m_name = data["model_name"]
                best_model_file = os.path.join(SAVED_MODELS_DIR, f"{m_name}_{m_id}.joblib")

    if best_model_file is None or not os.path.exists(best_model_file):
        filter_msg = f"for model '{model_name}' " if model_name else ""
        raise FileNotFoundError(f"Could not locate best model {filter_msg}based on metric '{metric_key}'.")

    # Load Model and Predict on Preprocessed Features (test_x_df)
    best_model = joblib.load(best_model_file)
    raw_predictions = best_model.predict(test_x_df)
    probabilities = best_model.predict_proba(test_x_df) if hasattr(best_model, "predict_proba") else None

    # Bi-directional label map for strings <-> integers
    str_to_int_map = {'M': 1, 'B': 0, '1': 1, '0': 0, 1: 1, 0: 0}
    int_to_str_map = {1: 'M', 0: 'B', '1': 'M', '0': 'B', 'M': 'M', 'B': 'B'}

    string_predictions = [int_to_str_map.get(pred, pred) for pred in raw_predictions]

    # Process test_y and calculate metrics IF valid ground truth is provided
    pred_metrics = None
    if test_y is not None:
        valid_y = test_y.dropna() if isinstance(test_y, (pd.Series, pd.DataFrame)) else pd.Series(test_y).dropna()
        if len(valid_y) > 0 and len(valid_y) == len(raw_predictions):
            try:
                numeric_test_y = [str_to_int_map[val] for val in valid_y if val in str_to_int_map]
                numeric_preds = [str_to_int_map.get(p, p) for p in raw_predictions]

                if len(numeric_test_y) == len(numeric_preds):
                    pred_metrics = calculate_metrics(numeric_test_y, numeric_preds, probabilities)
            except Exception as e:
                print(f"⚠️ Metric calculation skipped: {e}")

    # Use raw_features_df for the final output table if provided, otherwise fallback to test_x_df
    base_output_df = raw_features_df.copy() if raw_features_df is not None else test_x_df.copy()

    # Attach prediction and actual labels
    base_output_df[target_name] = string_predictions

    if test_y is not None:
        actual_labels = [int_to_str_map.get(val, val) if pd.notna(val) else None for val in test_y]
        base_output_df["actual_diagnosis"] = actual_labels

    # Build Response Structure
    result_json = {
        "model_used": f"{best_model_info['model_name']}_{best_model_info['model_id']}",
        "test_evaluation_metrics": pred_metrics,
        "predictions": {}
    }

    for idx, row in base_output_df.iterrows():
        result_json["predictions"][str(idx)] = row.to_dict()

    if save_output:
        run_id = custom_id if custom_id else str(uuid.uuid4().hex[:6])
        output_filename = f"test_results_{run_id}.json"
        with open(output_filename, "w") as f:
            json.dump(result_json, f, indent=4)

    return result_json