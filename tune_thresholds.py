from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def infer_columns(frame: pd.DataFrame, label_column: str | None, score_column: str | None) -> tuple[str, str]:
    if label_column is None:
        for candidate in ("y_true", "y_true_future_fraud"):
            if candidate in frame.columns:
                label_column = candidate
                break
    if score_column is None:
        for candidate in ("fraud_score", "future_fraud_score"):
            if candidate in frame.columns:
                score_column = candidate
                break
    if label_column is None or score_column is None:
        raise ValueError("Could not infer label/score columns; pass --label-column and --score-column.")
    return label_column, score_column


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "flagged_rows": int(pred.sum()),
    }


def choose_operating_points(y_true: np.ndarray, scores: np.ndarray, precision_floors: list[float]) -> dict:
    print(f"[thresholds] computing precision-recall curve for {len(y_true)} rows", flush=True)
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        thresholds = np.array([0.5])
    print(f"[thresholds] evaluating {len(thresholds)} candidate thresholds", flush=True)
    candidates = [metrics_at_threshold(y_true, scores, float(threshold)) for threshold in thresholds]
    print("[thresholds] selecting operating points", flush=True)
    result = {
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        "best_f1": max(candidates, key=lambda row: row["f1"]),
        "best_f2": max(candidates, key=lambda row: row["f2"]),
    }
    for floor in precision_floors:
        eligible = [row for row in candidates if row["precision"] >= floor]
        result[f"best_recall_at_precision_{floor:g}"] = (
            max(eligible, key=lambda row: row["recall"]) if eligible else None
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune threshold operating points for prediction CSVs.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--label-column")
    parser.add_argument("--score-column")
    parser.add_argument("--precision-floor", type=float, action="append", default=[0.5, 0.6, 0.7])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    print(f"[thresholds] reading {args.predictions}", flush=True)
    frame = pd.read_csv(args.predictions)
    print(f"[thresholds] loaded rows={len(frame)} columns={len(frame.columns)}", flush=True)
    label_column, score_column = infer_columns(frame, args.label_column, args.score_column)
    print(f"[thresholds] using label_column={label_column} score_column={score_column}", flush=True)
    valid = frame[[label_column, score_column]].dropna()
    print(f"[thresholds] valid scored rows={len(valid)} dropped_rows={len(frame) - len(valid)}", flush=True)
    result = choose_operating_points(
        valid[label_column].astype(int).to_numpy(),
        valid[score_column].astype(float).to_numpy(),
        args.precision_floor,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output_json:
        print(f"[thresholds] writing {args.output_json}", flush=True)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload, encoding="utf-8")
        print("[thresholds] done", flush=True)


if __name__ == "__main__":
    main()
