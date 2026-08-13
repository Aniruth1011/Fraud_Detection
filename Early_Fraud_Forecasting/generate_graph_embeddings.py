from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate past-only entity graph embeddings for early fraud forecasting."
    )
    parser.add_argument(
        "--config",
        default="configs/early_fraud_forecasting.json",
        help="Forecasting config containing transaction paths and embedding settings.",
    )
    return parser.parse_args()


def read_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def build_entity_index(transactions: pd.DataFrame) -> pd.Index:
    return pd.Index(
        pd.concat([transactions["src"], transactions["dest"]], ignore_index=True)
        .dropna()
        .unique()
    )


def bucket_start(timestamp: pd.Series, frequency: str) -> pd.Series:
    return timestamp.dt.to_period(frequency).dt.start_time


def build_adjacency(
    transactions: pd.DataFrame,
    entities: pd.Index,
    weighting: str,
) -> sparse.csr_matrix:
    entity_codes = pd.Series(np.arange(len(entities)), index=entities)
    src_index = transactions["src"].map(entity_codes).to_numpy()
    dest_index = transactions["dest"].map(entity_codes).to_numpy()

    valid = ~pd.isna(src_index) & ~pd.isna(dest_index)
    src_index = src_index[valid].astype(int)
    dest_index = dest_index[valid].astype(int)

    if weighting == "amount":
        weights = pd.to_numeric(transactions.loc[valid, "amount"], errors="coerce")
        weights = np.log1p(weights.fillna(0.0).clip(lower=0.0)).to_numpy()
    else:
        weights = np.ones(len(src_index), dtype=float)

    row = np.concatenate([src_index, dest_index])
    col = np.concatenate([dest_index, src_index])
    value = np.concatenate([weights, weights])
    adjacency = sparse.coo_matrix(
        (value, (row, col)),
        shape=(len(entities), len(entities)),
    ).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def make_embeddings(
    adjacency: sparse.csr_matrix,
    dimensions: int,
    random_state: int,
) -> np.ndarray:
    if adjacency.nnz == 0:
        return np.zeros((adjacency.shape[0], dimensions), dtype=float)

    max_components = max(1, min(dimensions, adjacency.shape[0] - 1))
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    matrix = normalize(adjacency, norm="l2", axis=1)
    embeddings = svd.fit_transform(matrix)
    if embeddings.shape[1] < dimensions:
        padding = np.zeros((embeddings.shape[0], dimensions - embeddings.shape[1]))
        embeddings = np.hstack([embeddings, padding])
    return embeddings


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    embedding_config = config.get("graph_embeddings", {})

    output_path = Path(
        config.get(
            "entity_embeddings_path",
            "outputs/early_fraud_forecasting/graph_embeddings/entity_embeddings.csv",
        )
    )
    dimensions = int(embedding_config.get("dimensions", 32))
    frequency = embedding_config.get("snapshot_frequency", "W")
    min_history_days = int(embedding_config.get("min_history_days", 7))
    weighting = embedding_config.get("weighting", "count")
    random_state = int(embedding_config.get("random_state", 42))

    print(f"[data] reading transactions from {config['transactions_path']}")
    transactions = pd.read_csv(config["transactions_path"], low_memory=False)
    transactions = transactions.loc[transactions["edge_type"] == "transaction"].copy()
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="coerce")
    transactions = transactions.dropna(subset=["timestamp", "src", "dest"])
    transactions = transactions.sort_values("timestamp", kind="stable").reset_index(drop=True)
    transactions["current_is_fraud"] = parse_bool_series(transactions["is_fraudulent"])

    entities = build_entity_index(transactions)
    transactions["snapshot_start"] = bucket_start(transactions["timestamp"], frequency)
    snapshot_starts = pd.Index(transactions["snapshot_start"].drop_duplicates()).sort_values()
    min_timestamp = transactions["timestamp"].min()

    rows = []
    embedding_columns = [f"embedding_{index:03d}" for index in range(dimensions)]

    for snapshot_start in snapshot_starts:
        if snapshot_start - min_timestamp < pd.Timedelta(days=min_history_days):
            continue

        history = transactions.loc[transactions["timestamp"] < snapshot_start]
        if history.empty:
            continue

        print(f"[embed] building {frequency} snapshot as of {snapshot_start}")
        adjacency = build_adjacency(history, entities, weighting)
        embeddings = make_embeddings(adjacency, dimensions, random_state)

        snapshot = pd.DataFrame(embeddings, columns=embedding_columns)
        snapshot.insert(0, "as_of_timestamp", snapshot_start)
        snapshot.insert(0, "entity_id", entities.to_numpy())
        rows.append(snapshot)

    if not rows:
        raise RuntimeError("No embedding snapshots were created. Check date range settings.")

    output = pd.concat(rows, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"[embed] saved {len(output)} entity embeddings to {output_path}")


if __name__ == "__main__":
    main()
