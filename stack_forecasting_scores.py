from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


KEY_COLUMNS = ["src", "dest", "timestamp"]


def read_score_file(path: Path, score_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    label_column = "y_true_future_fraud" if "y_true_future_fraud" in frame.columns else "y_true"
    score_column = "future_fraud_score" if "future_fraud_score" in frame.columns else "fraud_score"
    required = set(KEY_COLUMNS + [label_column, score_column])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    output = frame[KEY_COLUMNS + [label_column, score_column]].copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    output = output.rename(columns={label_column: "y_true", score_column: score_name})
    return output


def merge_score_files(paths: list[Path], split: str) -> pd.DataFrame:
    merged = None
    for index, run_dir in enumerate(paths):
        score_name = f"{run_dir.name or f'score_{index}'}_score"
        frame = read_score_file(run_dir / f"{split}_predictions.csv", score_name)
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=KEY_COLUMNS + ["y_true"], how="inner")
    if merged is None:
        raise ValueError("At least one run directory is required.")
    return merged


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "predicted_positive_rows": int(y_pred.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


def train_meta_model(kind: str, validation: pd.DataFrame, score_columns: list[str]):
    if kind == "logistic_regression":
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
    elif kind == "xgboost":
        from xgboost import XGBClassifier

        positives = max(int(validation["y_true"].sum()), 1)
        negatives = max(int(len(validation) - positives), 1)
        model = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=negatives / positives,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported meta-model: {kind}")
    model.fit(validation[score_columns], validation["y_true"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Stack forecasting and GNN score predictions.")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Dirs with validation/test prediction CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/forecasting_score_stack"))
    parser.add_argument("--meta-model", choices=["logistic_regression", "xgboost", "mean", "max"], default="xgboost")
    args = parser.parse_args()

    validation = merge_score_files(args.run_dirs, "validation")
    test = merge_score_files(args.run_dirs, "test")
    score_columns = [column for column in validation.columns if column.endswith("_score")]
    if args.meta_model == "mean":
        validation_prob = validation[score_columns].mean(axis=1).to_numpy()
        test_prob = test[score_columns].mean(axis=1).to_numpy()
        model = None
    elif args.meta_model == "max":
        validation_prob = validation[score_columns].max(axis=1).to_numpy()
        test_prob = test[score_columns].max(axis=1).to_numpy()
        model = None
    else:
        model = train_meta_model(args.meta_model, validation, score_columns)
        validation_prob = model.predict_proba(validation[score_columns])[:, 1]
        test_prob = model.predict_proba(test[score_columns])[:, 1]

    threshold = best_f1_threshold(validation["y_true"].to_numpy(), validation_prob)
    test_metrics = metrics(test["y_true"].to_numpy(), test_prob, threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, frame, probabilities in (
        ("validation", validation, validation_prob),
        ("test", test, test_prob),
    ):
        output = frame[KEY_COLUMNS + ["y_true"]].copy()
        output["fraud_score"] = probabilities
        output["y_pred"] = (probabilities >= threshold).astype(np.int64)
        output.to_csv(args.output_dir / f"{split_name}_predictions.csv", index=False)

    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(confusion_matrix(test["y_true"], (test_prob >= threshold).astype(np.int64))).to_csv(
        args.output_dir / "confusion_matrix.csv",
        index=False,
    )
    if model is not None:
        joblib.dump(model, args.output_dir / "meta_model.joblib")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
