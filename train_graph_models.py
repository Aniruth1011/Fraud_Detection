from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GINEConv,
    Linear,
    PNAConv,
    RGCNConv,
    to_hetero,
)
from torch_geometric.utils import degree


FORWARD_RELATION = ("node", "to", "node")
REVERSE_RELATION = ("node", "rev_to", "node")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Multi-GNN style graph fraud models.")
    parser.add_argument(
        "--config",
        default="configs/graph_models_li.json",
        help="Path to the graph-model JSON config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch_size from the JSON config.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        help="Override num_neighbors from the JSON config, e.g. --num-neighbors 10 10.",
    )
    return parser.parse_args()


def read_config(config_path: str) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_token(value):
    if pd.isna(value):
        return value
    if isinstance(value, str) and "." in value:
        return value.split(".", 1)[1]
    return value


def parse_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "false": 0, "1": 1, "0": 0})
        .fillna(0)
        .astype(int)
    )


def setup_logger(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def save_torch_artifact(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_torch_artifact(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def z_norm(train_tensor: torch.Tensor, other_tensors: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    mean = train_tensor.mean(0, keepdim=True)
    std = train_tensor.std(0, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    normalized_train = (train_tensor - mean) / std
    normalized_others = [(tensor - mean) / std for tensor in other_tensors]
    return normalized_train, normalized_others


def factorize_with_vocab(series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    values = series.fillna("__missing__").astype(str)
    categories = sorted(values.unique())
    vocab = {value: index for index, value in enumerate(categories)}
    return values.map(vocab).astype(int), vocab


def build_feature_frame(frame: pd.DataFrame, drop_columns: list[str]) -> pd.DataFrame:
    features = frame.drop(columns=drop_columns, errors="ignore").copy()
    for column in features.columns:
        if pd.api.types.is_bool_dtype(features[column]):
            features[column] = features[column].astype(int)
    features = pd.get_dummies(features, dummy_na=True)
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return features


def select_temporal_split_days(timestamps: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    daily_indices: list[np.ndarray] = []
    daily_counts: list[int] = []
    for day in range(n_days):
        left = day * 24 * 3600
        right = (day + 1) * 24 * 3600
        day_indices = np.where((timestamps >= left) & (timestamps < right))[0]
        daily_indices.append(day_indices)
        daily_counts.append(int(day_indices.shape[0]))

    split_target = [0.6, 0.2, 0.2]
    split_scores: dict[tuple[int, int], float] = {}
    totals = np.array(daily_counts)
    all_days = list(range(len(totals)))

    for start, end in itertools.combinations(all_days, 2):
        if end < start:
            continue
        split_totals = [totals[:start].sum(), totals[start:end].sum(), totals[end:].sum()]
        split_sum = max(np.sum(split_totals), 1)
        split_props = [value / split_sum for value in split_totals]
        split_errors = [abs(value - target) / target for value, target in zip(split_props, split_target)]
        split_scores[(start, end)] = max(split_errors)

    start, end = min(split_scores, key=split_scores.get)
    return list(range(start)), list(range(start, end)), list(range(end, len(totals)))


def build_ports(edge_index: torch.Tensor, timestamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    ts = timestamps.cpu().numpy()
    edges = pd.DataFrame({"src": src, "dest": dst, "timestamp": ts})

    incoming = (
        edges.groupby(["dest", "src"], sort=False)["timestamp"]
        .min()
        .reset_index()
        .sort_values(["dest", "timestamp", "src"], kind="stable")
    )
    incoming["in_port"] = incoming.groupby("dest").cumcount()
    outgoing = (
        edges.groupby(["src", "dest"], sort=False)["timestamp"]
        .min()
        .reset_index()
        .sort_values(["src", "timestamp", "dest"], kind="stable")
    )
    outgoing["out_port"] = outgoing.groupby("src").cumcount()

    enriched = edges.merge(incoming[["dest", "src", "in_port"]], on=["dest", "src"], how="left")
    enriched = enriched.merge(outgoing[["src", "dest", "out_port"]], on=["src", "dest"], how="left")
    in_ports = torch.tensor(enriched["in_port"].fillna(0).to_numpy(), dtype=torch.float32).view(-1, 1)
    out_ports = torch.tensor(enriched["out_port"].fillna(0).to_numpy(), dtype=torch.float32).view(-1, 1)
    return in_ports, out_ports


def build_time_deltas(edge_index: torch.Tensor, timestamps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    ts = timestamps.cpu().numpy()
    frame = pd.DataFrame({"edge_id": np.arange(len(ts)), "src": src, "dest": dst, "timestamp": ts})

    in_delta = np.zeros(len(ts), dtype=np.float32)
    out_delta = np.zeros(len(ts), dtype=np.float32)

    for _, group in frame.groupby("dest", sort=False):
        ordered = group.sort_values("timestamp", kind="stable")
        values = ordered["timestamp"].to_numpy()
        deltas = np.concatenate([[0], np.diff(values)]).astype(np.float32)
        in_delta[ordered["edge_id"].to_numpy()] = deltas

    for _, group in frame.groupby("src", sort=False):
        ordered = group.sort_values("timestamp", kind="stable")
        values = ordered["timestamp"].to_numpy()
        deltas = np.concatenate([[0], np.diff(values)]).astype(np.float32)
        out_delta[ordered["edge_id"].to_numpy()] = deltas

    return torch.tensor(in_delta).view(-1, 1), torch.tensor(out_delta).view(-1, 1)


def load_transactions(config: dict) -> pd.DataFrame:
    print(f"[graph-data] reading transactions from {config['transactions_path']}", flush=True)
    transactions = pd.read_csv(config["transactions_path"])
    print("[graph-data] filtering transaction edges", flush=True)
    transactions = transactions.loc[transactions["edge_type"] == "transaction"].copy()

    print("[graph-data] normalizing strings", flush=True)
    for column in transactions.columns:
        if transactions[column].dtype == "object":
            transactions[column] = transactions[column].map(normalize_token)

    print("[graph-data] building raw AML-style edge features", flush=True)
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="coerce")
    transactions = transactions.dropna(subset=["timestamp"]).reset_index(drop=True)
    transactions["timestamp_seconds"] = (
        transactions["timestamp"].astype("int64") // 10**9
    ).astype(np.int64)
    transactions["timestamp_seconds"] = (
        transactions["timestamp_seconds"] - transactions["timestamp_seconds"].min()
    )
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce").fillna(0.0)
    transactions["label"] = parse_bool_series(transactions["is_fraudulent"])
    transactions["currency_id"], _ = factorize_with_vocab(transactions["currency"])
    transactions["payment_format_id"], _ = factorize_with_vocab(transactions["transaction_type"])
    transactions = transactions.sort_values("timestamp_seconds", kind="stable").reset_index(drop=True)
    transactions["edge_uid"] = np.arange(len(transactions), dtype=np.int64)
    return transactions


def build_base_graph(config: dict) -> tuple[Data, pd.DataFrame]:
    transactions = load_transactions(config)

    print(f"[graph-data] reading nodes from {config['nodes_path']}", flush=True)
    nodes = pd.read_csv(config["nodes_path"])
    for column in nodes.columns:
        if nodes[column].dtype == "object":
            nodes[column] = nodes[column].map(normalize_token)

    node_ids = nodes["node_id"].astype(str).tolist()
    referenced = pd.Index(transactions["src"].astype(str)).append(pd.Index(transactions["dest"].astype(str)))
    missing_nodes = sorted(set(referenced.unique()) - set(node_ids))
    if missing_nodes:
        node_ids.extend(missing_nodes)

    node_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    transactions["src_index"] = transactions["src"].astype(str).map(node_to_index).astype(int)
    transactions["dest_index"] = transactions["dest"].astype(str).map(node_to_index).astype(int)

    x = torch.ones((len(node_ids), 1), dtype=torch.float32)
    edge_index = torch.tensor(
        transactions[["src_index", "dest_index"]].to_numpy().T,
        dtype=torch.long,
    )
    timestamps = torch.tensor(transactions["timestamp_seconds"].to_numpy(), dtype=torch.float32)

    edge_attr_parts = [
        torch.tensor(transactions["edge_uid"].to_numpy(), dtype=torch.float32).view(-1, 1),
        timestamps.view(-1, 1),
        torch.tensor(transactions["amount"].to_numpy(), dtype=torch.float32).view(-1, 1),
        torch.tensor(transactions["currency_id"].to_numpy(), dtype=torch.float32).view(-1, 1),
        torch.tensor(transactions["payment_format_id"].to_numpy(), dtype=torch.float32).view(-1, 1),
    ]

    if config.get("use_ports", True):
        print("[graph-data] adding port numbers", flush=True)
        in_ports, out_ports = build_ports(edge_index, timestamps)
        edge_attr_parts.extend([in_ports, out_ports])

    if config.get("use_time_deltas", True):
        print("[graph-data] adding time deltas", flush=True)
        in_delta, out_delta = build_time_deltas(edge_index, timestamps)
        edge_attr_parts.extend([in_delta, out_delta])

    edge_attr = torch.cat(edge_attr_parts, dim=1)
    edge_label = torch.tensor(transactions["label"].to_numpy(), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=edge_label,
        timestamps=timestamps,
        edge_uid=torch.tensor(transactions["edge_uid"].to_numpy(), dtype=torch.long),
    )
    return data, transactions


def build_full_hetero_graph(config: dict) -> tuple[HeteroData, dict[str, pd.DataFrame]]:
    print("[full-hetero] reading nodes and all edges")
    nodes = pd.read_csv(config["nodes_path"])
    transactions = pd.read_csv(config["transactions_path"])

    for frame in (nodes, transactions):
        for column in frame.columns:
            if frame[column].dtype == "object":
                frame[column] = frame[column].map(normalize_token)

    nodes["node_type"] = nodes["node_type"].fillna("UNKNOWN").astype(str)
    nodes["creation_date"] = pd.to_datetime(nodes["creation_date"], errors="coerce")
    nodes["creation_year"] = nodes["creation_date"].dt.year
    nodes["creation_month"] = nodes["creation_date"].dt.month
    nodes["incorporation_year"] = pd.to_numeric(nodes["incorporation_year"], errors="coerce")
    nodes["number_of_employees"] = pd.to_numeric(nodes["number_of_employees"], errors="coerce")
    nodes["risk_score"] = pd.to_numeric(nodes["risk_score"], errors="coerce")
    for column in ["is_fraudulent", "is_high_risk_category", "is_high_risk_country"]:
        if column in nodes.columns:
            nodes[column] = parse_bool_series(nodes[column])

    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="coerce")
    transactions["amount"] = pd.to_numeric(transactions["amount"], errors="coerce")
    transactions["ownership_percentage"] = pd.to_numeric(
        transactions["ownership_percentage"], errors="coerce"
    )
    transactions["transaction_label"] = parse_bool_series(transactions["is_fraudulent"])
    transactions["timestamp_seconds"] = (
        transactions["timestamp"].astype("int64", errors="ignore") // 10**9
    )
    valid_timestamps = transactions["timestamp_seconds"].dropna()
    if not valid_timestamps.empty:
        transactions["timestamp_seconds"] = transactions["timestamp_seconds"] - valid_timestamps.min()
    transactions["time_since_previous_transaction_seconds"] = pd.to_timedelta(
        transactions["time_since_previous_transaction"], errors="coerce"
    ).dt.total_seconds()

    node_type_to_storage = {
        "INDIVIDUAL": "individual",
        "BUSINESS": "business",
        "ACCOUNT": "account",
        "FINANCIAL_INSTITUTION": "institution",
        "INSTITUTION": "institution",
        "UNKNOWN": "unknown",
    }
    nodes["storage_type"] = nodes["node_type"].map(node_type_to_storage).fillna("unknown")

    hetero = HeteroData()
    node_lookup: dict[str, tuple[str, int]] = {}
    node_tables: dict[str, pd.DataFrame] = {}
    transaction_relations: list[tuple[str, str, str]] = []
    ownership_relations: list[tuple[str, str, str]] = []

    print("[full-hetero] building typed node tables")
    for storage_type, group in nodes.groupby("storage_type", sort=False):
        group = group.reset_index(drop=True).copy()
        feature_frame = build_feature_frame(
            group,
            drop_columns=["node_id", "node_type", "storage_type", "name", "creation_date"],
        )
        hetero[storage_type].x = torch.tensor(feature_frame.to_numpy(dtype=np.float32), dtype=torch.float32)
        node_tables[storage_type] = group
        for local_index, node_id in enumerate(group["node_id"].astype(str).tolist()):
            node_lookup[node_id] = (storage_type, local_index)

    print("[full-hetero] splitting ownership and transaction relations")
    ownership_edges = transactions.loc[transactions["edge_type"] == "ownership"].copy()
    transaction_edges = transactions.loc[transactions["edge_type"] == "transaction"].copy()

    relation_tables: dict[str, pd.DataFrame] = {}

    if not ownership_edges.empty:
        ownership_edges = ownership_edges.loc[
            ownership_edges["src"].astype(str).isin(node_lookup)
            & ownership_edges["dest"].astype(str).isin(node_lookup)
        ].copy()
        ownership_edges["src_type"] = ownership_edges["src"].astype(str).map(lambda value: node_lookup[value][0])
        ownership_edges["src_local"] = ownership_edges["src"].astype(str).map(lambda value: node_lookup[value][1])
        ownership_edges["dest_type"] = ownership_edges["dest"].astype(str).map(lambda value: node_lookup[value][0])
        ownership_edges["dest_local"] = ownership_edges["dest"].astype(str).map(lambda value: node_lookup[value][1])

        for (src_type, dest_type), group in ownership_edges.groupby(["src_type", "dest_type"], sort=False):
            relation = (src_type, "owns", dest_type)
            edge_attr_frame = build_feature_frame(
                group[
                    [
                        "ownership_percentage",
                        "currency",
                    ]
                ],
                drop_columns=[],
            )
            hetero[relation].edge_index = torch.tensor(
                group[["src_local", "dest_local"]].to_numpy().T,
                dtype=torch.long,
            )
            hetero[relation].edge_attr = torch.tensor(
                edge_attr_frame.to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            ownership_relations.append(relation)
            relation_tables[f"{src_type}__owns__{dest_type}"] = group.reset_index(drop=True)

    if not transaction_edges.empty:
        transaction_edges = transaction_edges.loc[
            transaction_edges["src"].astype(str).isin(node_lookup)
            & transaction_edges["dest"].astype(str).isin(node_lookup)
        ].copy()
        transaction_edges["src_type"] = transaction_edges["src"].astype(str).map(lambda value: node_lookup[value][0])
        transaction_edges["src_local"] = transaction_edges["src"].astype(str).map(lambda value: node_lookup[value][1])
        transaction_edges["dest_type"] = transaction_edges["dest"].astype(str).map(lambda value: node_lookup[value][0])
        transaction_edges["dest_local"] = transaction_edges["dest"].astype(str).map(lambda value: node_lookup[value][1])
        transaction_edges["transaction_uid"] = np.arange(len(transaction_edges), dtype=np.int64)

        for (src_type, dest_type), group in transaction_edges.groupby(["src_type", "dest_type"], sort=False):
            relation = (src_type, "transacts_to", dest_type)
            edge_attr_frame = build_feature_frame(
                group[
                    [
                        "amount",
                        "currency",
                        "transaction_type",
                        "ownership_percentage",
                        "time_since_previous_transaction_seconds",
                        "timestamp_seconds",
                    ]
                ],
                drop_columns=[],
            )
            hetero[relation].edge_index = torch.tensor(
                group[["src_local", "dest_local"]].to_numpy().T,
                dtype=torch.long,
            )
            hetero[relation].edge_attr = torch.tensor(
                edge_attr_frame.to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
            hetero[relation].y = torch.tensor(
                group["transaction_label"].fillna(0).to_numpy(),
                dtype=torch.long,
            )
            hetero[relation].timestamps = torch.tensor(
                group["timestamp_seconds"].fillna(0).to_numpy(),
                dtype=torch.float32,
            )
            hetero[relation].edge_uid = torch.tensor(
                group["transaction_uid"].to_numpy(),
                dtype=torch.long,
            )
            transaction_relations.append(relation)
            relation_tables[f"{src_type}__transacts_to__{dest_type}"] = group.reset_index(drop=True)

    metadata = {
        "node_tables": node_tables,
        "relation_tables": relation_tables,
        "transaction_relations": transaction_relations,
        "ownership_relations": ownership_relations,
    }
    return hetero, metadata


def load_or_build_base_graph(config: dict) -> tuple[Data, pd.DataFrame]:
    cache_path_raw = config.get("processed_graph_path")
    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        if cache_path.exists():
            print(f"[graph-data] loading cached graph from {cache_path}", flush=True)
            payload = load_torch_artifact(cache_path)
            return payload["data"], payload["transactions"]

    data, transactions = build_base_graph(config)

    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        print(f"[graph-data] saving cached graph to {cache_path}", flush=True)
        save_torch_artifact({"data": data, "transactions": transactions}, cache_path)

    return data, transactions


def load_or_build_full_hetero_graph(config: dict) -> tuple[HeteroData, dict[str, pd.DataFrame]]:
    cache_path_raw = config.get("processed_full_hetero_graph_path")
    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        if cache_path.exists():
            print(f"[full-hetero] loading cached graph from {cache_path}")
            payload = load_torch_artifact(cache_path)
            return payload["hetero"], payload["metadata"]

    hetero, metadata = build_full_hetero_graph(config)

    if cache_path_raw:
        cache_path = Path(cache_path_raw)
        print(f"[full-hetero] saving cached graph to {cache_path}")
        save_torch_artifact({"hetero": hetero, "metadata": metadata}, cache_path)

    return hetero, metadata


def split_temporally(data: Data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = data.timestamps.cpu().numpy()
    train_days, val_days, test_days = select_temporal_split_days(timestamps)
    day_indices: list[np.ndarray] = []
    n_days = int(timestamps.max() / (3600 * 24) + 1)
    for day in range(n_days):
        left = day * 24 * 3600
        right = (day + 1) * 24 * 3600
        day_indices.append(np.where((timestamps >= left) & (timestamps < right))[0])

    train_idx = np.concatenate([day_indices[day] for day in train_days]) if train_days else np.array([], dtype=int)
    val_idx = np.concatenate([day_indices[day] for day in val_days]) if val_days else np.array([], dtype=int)
    test_idx = np.concatenate([day_indices[day] for day in test_days]) if test_days else np.array([], dtype=int)
    return train_idx, val_idx, test_idx


def make_snapshot(data: Data, edge_indices: np.ndarray) -> Data:
    return Data(
        x=data.x.clone(),
        edge_index=data.edge_index[:, edge_indices].clone(),
        edge_attr=data.edge_attr[edge_indices].clone(),
        y=data.y[edge_indices].clone(),
        timestamps=data.timestamps[edge_indices].clone(),
        edge_uid=data.edge_uid[edge_indices].clone(),
    )


def create_hetero_snapshot(snapshot: Data, config: dict) -> HeteroData:
    hetero = HeteroData()
    hetero["node"].x = snapshot.x
    hetero[FORWARD_RELATION].edge_index = snapshot.edge_index
    hetero[FORWARD_RELATION].edge_attr = snapshot.edge_attr
    hetero[FORWARD_RELATION].y = snapshot.y
    hetero[FORWARD_RELATION].timestamps = snapshot.timestamps
    hetero[FORWARD_RELATION].edge_uid = snapshot.edge_uid

    hetero[REVERSE_RELATION].edge_index = snapshot.edge_index.flip(0)
    reverse_attr = snapshot.edge_attr.clone()
    if config.get("use_ports", True):
        reverse_attr[:, [5, 6]] = reverse_attr[:, [6, 5]]
    if config.get("use_time_deltas", True):
        delta_start = 7 if config.get("use_ports", True) else 5
        reverse_attr[:, [delta_start, delta_start + 1]] = reverse_attr[:, [delta_start + 1, delta_start]]
    hetero[REVERSE_RELATION].edge_attr = reverse_attr
    return hetero


def normalize_snapshots(train_snapshot: Data, val_snapshot: Data, test_snapshot: Data) -> tuple[Data, Data, Data]:
    train_snapshot.x, [val_snapshot.x, test_snapshot.x] = z_norm(
        train_snapshot.x,
        [val_snapshot.x, test_snapshot.x],
    )
    train_snapshot.edge_attr[:, 1:], [val_edges, test_edges] = z_norm(
        train_snapshot.edge_attr[:, 1:],
        [val_snapshot.edge_attr[:, 1:], test_snapshot.edge_attr[:, 1:]],
    )
    val_snapshot.edge_attr[:, 1:] = val_edges
    test_snapshot.edge_attr[:, 1:] = test_edges
    return train_snapshot, val_snapshot, test_snapshot


def build_datasets(config: dict):
    print("[graph-data] preparing temporal graph snapshots", flush=True)
    base_data, transactions = load_or_build_base_graph(config)
    print("[graph-data] splitting train/validation/test windows", flush=True)
    train_idx, val_idx, test_idx = split_temporally(base_data)

    print("[graph-data] cloning temporal snapshots", flush=True)
    train_snapshot = make_snapshot(base_data, train_idx)
    val_snapshot = make_snapshot(base_data, np.concatenate([train_idx, val_idx]))
    test_snapshot = make_snapshot(base_data, np.arange(base_data.edge_index.shape[1]))
    print("[graph-data] normalizing snapshot features", flush=True)
    train_snapshot, val_snapshot, test_snapshot = normalize_snapshots(
        train_snapshot, val_snapshot, test_snapshot
    )

    if config.get("use_reverse_mp", True):
        print("[graph-data] creating forward/reverse hetero snapshots", flush=True)
        train_data = create_hetero_snapshot(train_snapshot, config)
        val_data = create_hetero_snapshot(val_snapshot, config)
        test_data = create_hetero_snapshot(test_snapshot, config)
    else:
        train_data = train_snapshot
        val_data = val_snapshot
        test_data = test_snapshot

    print("[graph-data] dataset preparation complete", flush=True)
    return train_data, val_data, test_data, train_idx, val_idx, test_idx, transactions


def build_loaders(train_data, val_data, test_data, train_idx, val_idx, test_idx, config: dict):
    num_neighbors = config["num_neighbors"]
    batch_size = config["batch_size"]
    num_workers = int(config.get("num_workers", 2))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    if isinstance(train_data, HeteroData):
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors={FORWARD_RELATION: num_neighbors, REVERSE_RELATION: num_neighbors},
            edge_label_index=(FORWARD_RELATION, train_data[FORWARD_RELATION].edge_index),
            edge_label=train_data[FORWARD_RELATION].y,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors={FORWARD_RELATION: num_neighbors, REVERSE_RELATION: num_neighbors},
            edge_label_index=(FORWARD_RELATION, val_data[FORWARD_RELATION].edge_index[:, len(train_idx) :]),
            edge_label=val_data[FORWARD_RELATION].y[len(train_idx) :],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = LinkNeighborLoader(
            data=test_data,
            num_neighbors={FORWARD_RELATION: num_neighbors, REVERSE_RELATION: num_neighbors},
            edge_label_index=(FORWARD_RELATION, test_data[FORWARD_RELATION].edge_index[:, test_idx]),
            edge_label=test_data[FORWARD_RELATION].y[test_idx],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
    else:
        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=num_neighbors,
            edge_label_index=train_data.edge_index,
            edge_label=train_data.y,
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=num_neighbors,
            edge_label_index=val_data.edge_index[:, len(train_idx) :],
            edge_label=val_data.y[len(train_idx) :],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = LinkNeighborLoader(
            data=test_data,
            num_neighbors=num_neighbors,
            edge_label_index=test_data.edge_index[:, test_idx],
            edge_label=test_data.y[test_idx],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
    return train_loader, val_loader, test_loader


class GINEEdgeNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        final_dropout: float,
        edge_updates: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.edge_updates = edge_updates
        self.edge_mlps = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.norms.append(BatchNorm(hidden_dim))
            if self.edge_updates:
                self.edge_mlps.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        for layer_index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            updated = conv(x, edge_index, edge_attr)
            x = 0.5 * (x + F.relu(norm(updated)))
            x = self.dropout(x)
            if self.edge_updates:
                edge_update = self.edge_mlps[layer_index](torch.cat([x[src], x[dst], edge_attr], dim=1))
                edge_attr = 0.5 * (edge_attr + edge_update)
        edge_states = x[edge_index.T].reshape(-1, 2 * self.hidden_dim).relu()
        return self.readout(torch.cat([edge_states, edge_attr], dim=1))


class GATEdgeNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        final_dropout: float,
        heads: int,
        edge_updates: bool,
    ) -> None:
        super().__init__()
        hidden_dim = (hidden_dim // heads) * heads
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.edge_updates = edge_updates
        self.edge_mlps = nn.ModuleList()
        out_channels = hidden_dim // heads
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    out_channels,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(BatchNorm(hidden_dim))
            if self.edge_updates:
                self.edge_mlps.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        for layer_index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            updated = conv(x, edge_index, edge_attr)
            x = 0.5 * (x + F.relu(norm(updated)))
            x = self.dropout(x)
            if self.edge_updates:
                edge_update = self.edge_mlps[layer_index](torch.cat([x[src], x[dst], edge_attr], dim=1))
                edge_attr = 0.5 * (edge_attr + edge_update)
        edge_states = x[edge_index.T].reshape(-1, 2 * self.hidden_dim).relu()
        return self.readout(torch.cat([edge_states, edge_attr], dim=1))


class PNAEdgeNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        final_dropout: float,
        deg_hist: torch.Tensor,
        edge_updates: bool,
    ) -> None:
        super().__init__()
        hidden_dim = int((hidden_dim // 4) * 4)
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.edge_updates = edge_updates
        self.edge_mlps = nn.ModuleList()
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        for _ in range(num_layers):
            self.convs.append(
                PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg_hist,
                    edge_dim=hidden_dim,
                    towers=4,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False,
                )
            )
            self.norms.append(BatchNorm(hidden_dim))
            if self.edge_updates:
                self.edge_mlps.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        for layer_index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            updated = conv(x, edge_index, edge_attr)
            x = 0.5 * (x + F.relu(norm(updated)))
            x = self.dropout(x)
            if self.edge_updates:
                edge_update = self.edge_mlps[layer_index](torch.cat([x[src], x[dst], edge_attr], dim=1))
                edge_attr = 0.5 * (edge_attr + edge_update)
        edge_states = x[edge_index.T].reshape(-1, 2 * self.hidden_dim).relu()
        return self.readout(torch.cat([edge_states, edge_attr], dim=1))


class RGCNEdgeNet(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        final_dropout: float,
        num_relations: int,
        edge_updates: bool,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.edge_updates = edge_updates
        self.edge_mlps = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations))
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            if self.edge_updates:
                self.edge_mlps.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Sequential(
            Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(final_dropout),
            Linear(hidden_dim // 2, 2),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        relation_type = edge_attr[:, -1].long()
        edge_payload = self.edge_proj(edge_attr[:, :-1])
        x = self.node_proj(x)
        for layer_index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            updated = conv(x, edge_index, relation_type)
            x = 0.5 * (x + F.relu(norm(updated)))
            x = self.dropout(x)
            if self.edge_updates:
                edge_update = self.edge_mlps[layer_index](torch.cat([x[src], x[dst], edge_payload], dim=1))
                edge_payload = 0.5 * (edge_payload + edge_update)
        edge_states = x[edge_index.T].reshape(-1, 2 * self.hidden_dim).relu()
        return self.readout(torch.cat([edge_states, edge_payload], dim=1))


def build_model(model_name: str, sample_data, config: dict):
    hidden_dim = config["hidden_dim"]
    num_layers = config["num_layers"]
    dropout = config["dropout"]
    final_dropout = config["final_dropout"]
    edge_updates = bool(config.get("edge_updates", False))

    if isinstance(sample_data, HeteroData):
        node_dim = sample_data["node"].x.shape[1]
        edge_dim = sample_data[FORWARD_RELATION].edge_attr.shape[1] - 1
        if model_name == "pna":
            degree_index = torch.cat(
                [
                    sample_data[FORWARD_RELATION].edge_index[1],
                    sample_data[REVERSE_RELATION].edge_index[1],
                ]
            )
            deg_hist = torch.bincount(degree(degree_index, dtype=torch.long), minlength=1)
            base_model = PNAEdgeNet(node_dim, edge_dim, hidden_dim, num_layers, dropout, final_dropout, deg_hist, edge_updates)
        elif model_name == "gat":
            heads = config["models"][model_name].get("heads", 4)
            base_model = GATEdgeNet(node_dim, edge_dim, hidden_dim, num_layers, dropout, final_dropout, heads, edge_updates)
        elif model_name == "gin":
            base_model = GINEEdgeNet(node_dim, edge_dim, hidden_dim, num_layers, dropout, final_dropout, edge_updates)
        elif model_name == "rgcn":
            relation_count = int(sample_data[FORWARD_RELATION].edge_attr[:, -1].max().item()) + 1
            base_model = RGCNEdgeNet(node_dim, edge_dim + 1, hidden_dim, num_layers, dropout, final_dropout, relation_count, edge_updates)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return to_hetero(base_model, sample_data.metadata(), aggr="mean")

    node_dim = sample_data.x.shape[1]
    edge_dim = sample_data.edge_attr.shape[1] - 1
    if model_name == "gin":
        return GINEEdgeNet(node_dim, edge_dim, hidden_dim, num_layers, dropout, final_dropout, edge_updates)
    if model_name == "gat":
        heads = config["models"][model_name].get("heads", 4)
        return GATEdgeNet(node_dim, edge_dim, hidden_dim, num_layers, dropout, final_dropout, heads, edge_updates)
    if model_name == "pna":
        deg_hist = torch.bincount(degree(sample_data.edge_index[1], dtype=torch.long), minlength=1)
        return PNAEdgeNet(node_dim, edge_dim, hidden_dim, num_layers, dropout, final_dropout, deg_hist, edge_updates)
    if model_name == "rgcn":
        relation_count = int(sample_data.edge_attr[:, -1].max().item()) + 1
        return RGCNEdgeNet(node_dim, edge_dim + 1, hidden_dim, num_layers, dropout, final_dropout, relation_count, edge_updates)
    raise ValueError(f"Unsupported model: {model_name}")


def append_rgcn_relation(sample_data):
    if isinstance(sample_data, HeteroData):
        for relation in [FORWARD_RELATION, REVERSE_RELATION]:
            edge_attr = sample_data[relation].edge_attr
            relation_type = edge_attr[:, 3:4].clone()
            sample_data[relation].edge_attr = torch.cat([edge_attr, relation_type], dim=1)
    else:
        relation_type = sample_data.edge_attr[:, 3:4].clone()
        sample_data.edge_attr = torch.cat([sample_data.edge_attr, relation_type], dim=1)


def mask_seed_edges_homo(batch, loader_edge_uid: torch.Tensor, split_indices: torch.Tensor) -> torch.Tensor:
    batch_edge_indices = split_indices[batch.input_id.long()]
    if hasattr(batch, "e_id"):
        return torch.isin(batch.e_id.long(), batch_edge_indices, assume_unique=True)
    batch_edge_uids = loader_edge_uid[batch_edge_indices]
    return torch.isin(batch.edge_attr[:, 0].long(), batch_edge_uids, assume_unique=True)


def mask_seed_edges_hetero(batch, loader_edge_uid: torch.Tensor, split_indices: torch.Tensor) -> torch.Tensor:
    batch_edge_indices = split_indices[batch[FORWARD_RELATION].input_id.long()]
    if "e_id" in batch[FORWARD_RELATION]:
        return torch.isin(
            batch[FORWARD_RELATION].e_id.long(),
            batch_edge_indices,
            assume_unique=True,
        )
    batch_edge_uids = loader_edge_uid[batch_edge_indices]
    return torch.isin(batch[FORWARD_RELATION].edge_attr[:, 0].long(), batch_edge_uids, assume_unique=True)


def autocast_settings(device: torch.device) -> tuple[bool, torch.dtype | None]:
    if device.type != "cuda":
        return False, None
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def count_batch_nodes_edges(batch) -> tuple[int, int]:
    if isinstance(batch, HeteroData):
        node_count = sum(store.x.shape[0] for store in batch.node_stores if "x" in store)
        edge_count = sum(store.edge_index.shape[1] for store in batch.edge_stores if "edge_index" in store)
        return int(node_count), int(edge_count)
    return int(batch.x.shape[0]), int(batch.edge_index.shape[1])


def evaluate_homo(model, loader, split_indices: np.ndarray, device):
    model.eval()
    preds = []
    probs = []
    truth = []
    uids = []
    split_indices = torch.as_tensor(split_indices, dtype=torch.long, device=device)
    loader_edge_uid = loader.data.edge_attr[:, 0].long().to(device, non_blocking=True)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            mask = mask_seed_edges_homo(batch, loader_edge_uid, split_indices)
            edge_uids = batch.edge_attr[mask, 0].detach().cpu().numpy().astype(np.int64)
            batch.edge_attr = batch.edge_attr[:, 1:]
            logits = model(batch.x, batch.edge_index, batch.edge_attr)
            masked_logits = logits[mask]
            masked_truth = batch.y[mask]
            preds.append(masked_logits.argmax(dim=-1).detach().cpu())
            probs.append(torch.softmax(masked_logits, dim=-1)[:, 1].detach().cpu())
            truth.append(masked_truth.detach().cpu())
            uids.append(edge_uids)
    y_true = torch.cat(truth).numpy()
    y_pred = torch.cat(preds).numpy()
    y_prob = torch.cat(probs).numpy()
    edge_uids = np.concatenate(uids) if uids else np.array([], dtype=np.int64)
    return y_true, y_pred, y_prob, edge_uids


def evaluate_hetero(model, loader, split_indices: np.ndarray, device):
    model.eval()
    preds = []
    probs = []
    truth = []
    uids = []
    split_indices = torch.as_tensor(split_indices, dtype=torch.long, device=device)
    loader_edge_uid = loader.data[FORWARD_RELATION].edge_attr[:, 0].long().to(device, non_blocking=True)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            mask = mask_seed_edges_hetero(batch, loader_edge_uid, split_indices)
            edge_uids = batch[FORWARD_RELATION].edge_attr[mask, 0].detach().cpu().numpy().astype(np.int64)
            batch[FORWARD_RELATION].edge_attr = batch[FORWARD_RELATION].edge_attr[:, 1:]
            batch[REVERSE_RELATION].edge_attr = batch[REVERSE_RELATION].edge_attr[:, 1:]
            logits_dict = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            logits = logits_dict[FORWARD_RELATION]
            masked_logits = logits[mask]
            masked_truth = batch[FORWARD_RELATION].y[mask]
            preds.append(masked_logits.argmax(dim=-1).detach().cpu())
            probs.append(torch.softmax(masked_logits, dim=-1)[:, 1].detach().cpu())
            truth.append(masked_truth.detach().cpu())
            uids.append(edge_uids)
    y_true = torch.cat(truth).numpy()
    y_pred = torch.cat(preds).numpy()
    y_prob = torch.cat(probs).numpy()
    edge_uids = np.concatenate(uids) if uids else np.array([], dtype=np.int64)
    return y_true, y_pred, y_prob, edge_uids


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Fraud", "Fraud"])
    ax.set_yticklabels(["Not Fraud", "Fraud"])
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_model(model_name: str, config: dict, train_data, val_data, test_data, train_idx, val_idx, test_idx, transactions: pd.DataFrame) -> None:
    model_dir = Path(config["output_root"]) / config["experiment_name"] / model_name
    logger = setup_logger(model_dir / "training.log")
    logger.info("Starting %s", model_name)

    if model_name == "rgcn":
        append_rgcn_relation(train_data)
        append_rgcn_relation(val_data)
        append_rgcn_relation(test_data)

    logger.info("Building LinkNeighborLoader objects")
    train_loader, val_loader, test_loader = build_loaders(train_data, val_data, test_data, train_idx, val_idx, test_idx, config)
    logger.info(
        "Built loaders train_batches=%s validation_batches=%s test_batches=%s batch_size=%s num_neighbors=%s num_workers=%s",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        config["batch_size"],
        config["num_neighbors"],
        int(config.get("num_workers", 2)),
    )

    sample_data = train_data
    logger.info("Building %s model", model_name)
    model = build_model(model_name, sample_data, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info("Model ready on device=%s", device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(config["positive_class_weight"])], dtype=torch.float32, device=device)
    )
    train_idx_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    if isinstance(train_data, HeteroData):
        train_edge_uid = train_loader.data[FORWARD_RELATION].edge_attr[:, 0].long().to(device, non_blocking=True)
    else:
        train_edge_uid = train_loader.data.edge_attr[:, 0].long().to(device, non_blocking=True)
    amp_enabled, amp_dtype = autocast_settings(device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    logger.info("Mixed precision enabled=%s dtype=%s", amp_enabled, amp_dtype)

    best_state = None
    best_val_f1 = -1.0
    log_every_batches = int(config.get("log_every_batches", 10))

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = torch.zeros((), device=device)
        total_examples = 0
        epoch_started_at = time.perf_counter()
        last_batch_finished_at = epoch_started_at
        logger.info("epoch=%s starting training batches", epoch + 1)

        for batch_index, batch in enumerate(train_loader, start=1):
            batch_received_at = time.perf_counter()
            sample_elapsed = batch_received_at - last_batch_finished_at
            optimizer.zero_grad(set_to_none=True)
            if isinstance(train_data, HeteroData):
                node_count, edge_count = count_batch_nodes_edges(batch)
                transfer_started_at = time.perf_counter()
                batch = batch.to(device, non_blocking=True)
                transfer_elapsed = time.perf_counter() - transfer_started_at
                mask = mask_seed_edges_hetero(batch, train_edge_uid, train_idx_tensor)
                mask_elapsed = time.perf_counter() - batch_received_at - transfer_elapsed
                batch[FORWARD_RELATION].edge_attr = batch[FORWARD_RELATION].edge_attr[:, 1:]
                batch[REVERSE_RELATION].edge_attr = batch[REVERSE_RELATION].edge_attr[:, 1:]
                compute_started_at = time.perf_counter()
                context = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
                with context:
                    logits = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)[FORWARD_RELATION]
                    pred = logits[mask]
                    truth = batch[FORWARD_RELATION].y[mask]
                    loss = loss_fn(pred, truth)
            else:
                node_count, edge_count = count_batch_nodes_edges(batch)
                transfer_started_at = time.perf_counter()
                batch = batch.to(device, non_blocking=True)
                transfer_elapsed = time.perf_counter() - transfer_started_at
                mask = mask_seed_edges_homo(batch, train_edge_uid, train_idx_tensor)
                mask_elapsed = time.perf_counter() - batch_received_at - transfer_elapsed
                batch.edge_attr = batch.edge_attr[:, 1:]
                compute_started_at = time.perf_counter()
                context = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
                with context:
                    logits = model(batch.x, batch.edge_index, batch.edge_attr)
                    pred = logits[mask]
                    truth = batch.y[mask]
                    loss = loss_fn(pred, truth)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            compute_elapsed = time.perf_counter() - compute_started_at
            total_loss += loss.detach() * pred.numel()
            total_examples += pred.numel()
            if batch_index == 1 or (log_every_batches > 0 and batch_index % log_every_batches == 0):
                loss_value = float(loss.detach().cpu())
                logger.info(
                    "epoch=%s batch=%s/%s loss=%.6f seed_edges=%s sampled_nodes=%s sampled_edges=%s sample_sec=%.2f mask_sec=%.2f transfer_sec=%.2f compute_sec=%.2f elapsed_sec=%.1f",
                    epoch + 1,
                    batch_index,
                    len(train_loader),
                    loss_value,
                    int(pred.numel()),
                    node_count,
                    edge_count,
                    sample_elapsed,
                    mask_elapsed,
                    transfer_elapsed,
                    compute_elapsed,
                    time.perf_counter() - epoch_started_at,
                )
            last_batch_finished_at = time.perf_counter()

        logger.info("epoch=%s starting validation", epoch + 1)
        if isinstance(val_data, HeteroData):
            y_true, y_pred, y_prob, _ = evaluate_hetero(model, val_loader, val_idx, device)
        else:
            y_true, y_pred, y_prob, _ = evaluate_homo(model, val_loader, val_idx, device)
        val_metrics = compute_metrics(y_true, y_pred, y_prob)
        logger.info(
            "epoch=%s loss=%.6f val_f1=%.6f val_pr_auc=%.6f",
            epoch + 1,
            float((total_loss / max(total_examples, 1)).detach().cpu()),
            val_metrics["f1"],
            val_metrics["pr_auc"],
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    if isinstance(test_data, HeteroData):
        y_true, y_pred, y_prob, edge_uids = evaluate_hetero(model, test_loader, test_idx, device)
    else:
        y_true, y_pred, y_prob, edge_uids = evaluate_homo(model, test_loader, test_idx, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("test_metrics=%s", json.dumps(metrics))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": model_name,
            "state_dict": {key: value.numpy() for key, value in model.state_dict().items()},
            "config": config,
        },
        model_dir / "model.joblib",
    )

    prediction_rows = transactions.set_index("edge_uid").loc[edge_uids].reset_index()
    prediction_rows = prediction_rows[["edge_uid", "src", "dest", "timestamp"]].copy()
    prediction_rows["y_true"] = y_true
    prediction_rows["y_pred"] = y_pred
    prediction_rows["fraud_score"] = y_prob
    prediction_rows.to_csv(model_dir / "test_predictions.csv", index=False)
    save_json(metrics, model_dir / "test_metrics.json")
    save_confusion_matrix(y_true, y_pred, model_dir / "confusion_matrix.png")
    logger.info("Finished %s", model_name)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_neighbors is not None:
        config["num_neighbors"] = args.num_neighbors
    train_data, val_data, test_data, train_idx, val_idx, test_idx, transactions = build_datasets(config)
    for model_name, model_config in config["models"].items():
        if model_config.get("enabled", True):
            train_one_model(
                model_name,
                config,
                train_data,
                val_data,
                test_data,
                train_idx,
                val_idx,
                test_idx,
                transactions,
            )


if __name__ == "__main__":
    main()
