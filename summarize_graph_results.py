from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize graph-model test metrics.")
    parser.add_argument("root", nargs="?", default="outputs", help="Directory to scan.")
    args = parser.parse_args()

    rows = []
    for metrics_path in Path(args.root).rglob("test_metrics.json"):
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        row = {"run": str(metrics_path.parent)}
        row.update(metrics)
        rows.append(row)

    if not rows:
        print(f"No test_metrics.json files found under {args.root}")
        return

    frame = pd.DataFrame(rows)
    sort_columns = [column for column in ["f1", "pr_auc", "roc_auc"] if column in frame.columns]
    frame = frame.sort_values(sort_columns, ascending=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
