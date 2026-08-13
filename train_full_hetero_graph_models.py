from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.typing as pyg_typing
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
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from tqdm.auto import tqdm

from train_graph_models import autocast_settings, load_or_build_full_hetero_graph, read_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full heterogeneous graph fraud models.")
    parser.add_argument(
        "--config",
        default="configs/full_hetero_graph_li.json",
        help="Path to the full-hetero graph JSON config.",
    )
    parser.add_argument(
        "--output-root",
        help="Override output_root from the JSON config.",
    )
    parser.add_argument(
        "--experiment-name",
        help="Override experiment_name from the JSON config.",
    )
    parser.add_argument(
        "--processed-full-hetero-graph-path",
        help="Override processed_full_hetero_graph_path from the JSON config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare/load the hetero graph and relation splits, then exit before training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override DataLoader worker count. Use 0 to debug or avoid worker crashes.",
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
        help="Override sampled neighbors per layer, e.g. --num-neighbors 25 10.",
    )
    parser.add_argument(
        "--relation",
        "--relations",
        dest="relations",
        nargs="+",
        help="Train only selected relation names, e.g. account__transacts_to__account.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        help="Override positive_class_weight from the JSON config.",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.output_root is not None:
        config["output_root"] = args.output_root
    if args.experiment_name is not None:
        config["experiment_name"] = args.experiment_name
    if args.processed_full_hetero_graph_path is not None:
        config["processed_full_hetero_graph_path"] = args.processed_full_hetero_graph_path
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_neighbors is not None:
        config["num_neighbors"] = args.num_neighbors
    if args.relations is not None:
        config["relations"] = args.relations
    if args.positive_class_weight is not None:
        config["positive_class_weight"] = args.positive_class_weight
    return config


def select_temporal_split_days(timestamps: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    day_ids = np.floor_divide(np.maximum(timestamps, 0), 3600 * 24).astype(np.int64)
    if day_ids.size == 0:
        return [], [], []
    totals = np.bincount(day_ids)
    n_days = int(totals.shape[0])
    cumulative = np.cumsum(totals)
    total_edges = int(cumulative[-1]) if cumulative.size else 0
    if total_edges == 0:
        return [], [], []

    start = int(np.searchsorted(cumulative, total_edges * 0.6, side="left") + 1)
    end = int(np.searchsorted(cumulative, total_edges * 0.8, side="left") + 1)
    start = min(max(start, 1), max(n_days - 2, 1))
    end = min(max(end, start + 1), max(n_days - 1, start + 1))
    return list(range(start)), list(range(start, end)), list(range(end, len(totals)))


def compute_relation_splits(relation_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    if relation_frame.empty:
        return {
            "train": np.array([], dtype=int),
            "validation": np.array([], dtype=int),
            "test": np.array([], dtype=int),
        }
    timestamps = relation_frame["timestamp_seconds"].fillna(0).to_numpy()
    train_days, val_days, test_days = select_temporal_split_days(timestamps)
    day_ids = np.floor_divide(np.maximum(timestamps, 0), 3600 * 24).astype(np.int64)
    train_mask = np.isin(day_ids, train_days)
    val_mask = np.isin(day_ids, val_days)
    test_mask = np.isin(day_ids, test_days)
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def build_full_hetero_datasets(config: dict):
    print("[full-hetero] preparing cached/full hetero graph", flush=True)
    hetero, metadata = load_or_build_full_hetero_graph(config)
    transaction_relations = metadata.get("transaction_relations", [])
    print(
        f"[full-hetero] loaded graph with {len(hetero.node_types)} node types, "
        f"{len(hetero.edge_types)} edge types, {len(transaction_relations)} transaction relations",
        flush=True,
    )
    if not transaction_relations:
        raise ValueError("Full hetero graph metadata has no transaction_relations to train.")

    relation_splits: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for relation in transaction_relations:
        relation_key = "__".join(relation)
        if relation_key not in metadata["relation_tables"]:
            raise KeyError(f"Missing relation table for transaction relation {relation_key}")
        relation_splits[relation] = compute_relation_splits(metadata["relation_tables"][relation_key])
        split_sizes = {name: int(indices.shape[0]) for name, indices in relation_splits[relation].items()}
        positives = int(metadata["relation_tables"][relation_key]["transaction_label"].sum())
        print(
            f"[full-hetero] relation={relation_key} rows={len(metadata['relation_tables'][relation_key])} "
            f"positives={positives} splits={split_sizes}",
            flush=True,
        )
    normalize_full_hetero_features(hetero, metadata, relation_splits)
    return hetero, metadata, relation_splits


def z_normalize_tensor_(tensor: torch.Tensor, reference: torch.Tensor | None = None) -> None:
    if tensor.numel() == 0:
        return
    stats_tensor = reference if reference is not None and reference.numel() else tensor
    mean = stats_tensor.mean(dim=0, keepdim=True)
    std = stats_tensor.std(dim=0, keepdim=True)
    std = torch.where(std <= 1e-12, torch.ones_like(std), std)
    tensor.sub_(mean).div_(std)


def normalize_full_hetero_features(
    hetero: HeteroData,
    metadata: dict,
    relation_splits: dict[tuple[str, str, str], dict[str, np.ndarray]],
) -> None:
    print("[full-hetero] normalizing node and edge features", flush=True)
    for node_type in hetero.node_types:
        z_normalize_tensor_(hetero[node_type].x)

    transaction_relations = set(metadata.get("transaction_relations", []))
    for relation in hetero.edge_types:
        if not hasattr(hetero[relation], "edge_attr"):
            continue
        reference = None
        if relation in transaction_relations:
            train_idx = relation_splits[relation]["train"]
            if train_idx.size:
                reference = hetero[relation].edge_attr[torch.as_tensor(train_idx, dtype=torch.long)]
        z_normalize_tensor_(hetero[relation].edge_attr, reference)


def build_relation_loader(data: HeteroData, relation, edge_indices: np.ndarray, config: dict, shuffle: bool):
    if edge_indices.size == 0:
        raise ValueError(f"Relation {'__'.join(relation)} has no edges for requested split.")
    edge_index = data[relation].edge_index[:, edge_indices]
    edge_label = data[relation].y[edge_indices]
    input_id = torch.as_tensor(edge_indices, dtype=torch.long)
    num_workers = int(config.get("num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    def attach_edge_label_attr(batch: HeteroData) -> HeteroData:
        batch[relation].edge_label_attr = data[relation].edge_attr[batch[relation].input_id]
        return batch

    num_neighbors = config["num_neighbors"]
    if isinstance(num_neighbors, int):
        num_neighbors = [num_neighbors] * config["num_layers"]
    return LinkNeighborLoader(
        data=data,
        num_neighbors={key: num_neighbors for key in data.edge_types},
        edge_label_index=(relation, edge_index),
        edge_label=edge_label,
        input_id=input_id,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        transform=attach_edge_label_attr,
        **loader_kwargs,
    )


class FullHeteroEdgeModel(nn.Module):
    def __init__(self, data: HeteroData, hidden_dim: int, num_layers: int, dropout: float, final_dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.node_projections = nn.ModuleDict()
        for node_type in data.node_types:
            self.node_projections[node_type] = Linear(data[node_type].x.shape[1], hidden_dim)

        self.edge_projections = nn.ModuleDict()
        self.edge_classifiers = nn.ModuleDict()
        self.relation_to_key: dict[tuple[str, str, str], str] = {}
        for relation in data.edge_types:
            relation_key = "__".join(relation)
            self.relation_to_key[relation] = relation_key
            self.edge_projections[relation_key] = Linear(data[relation].edge_attr.shape[1], hidden_dim)
            if relation[1] == "transacts_to":
                self.edge_classifiers[relation_key] = nn.Sequential(
                    Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(final_dropout),
                    Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(final_dropout),
                    Linear(hidden_dim // 2, 2),
                )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            relation_modules = {}
            for relation in data.edge_types:
                relation_modules[relation] = SAGEConv((hidden_dim, hidden_dim), hidden_dim)
            self.convs.append(HeteroConv(relation_modules, aggr="sum"))

    def encode_nodes(self, x_dict, edge_index_dict):
        embeddings = {
            node_type: self.node_projections[node_type](features)
            for node_type, features in x_dict.items()
        }
        for conv in self.convs:
            active_edge_index_dict = {
                relation: edge_index
                for relation, edge_index in edge_index_dict.items()
                if relation[0] in embeddings and relation[2] in embeddings
            }
            if not active_edge_index_dict:
                break
            updated = conv(embeddings, active_edge_index_dict)
            next_embeddings = dict(embeddings)
            for node_type, value in updated.items():
                next_embeddings[node_type] = F.dropout(
                    F.relu(value),
                    p=self.dropout,
                    training=self.training,
                )
            embeddings = next_embeddings
        return embeddings

    def classify_relation(self, embeddings, relation, edge_index, edge_attr):
        relation_key = self.relation_to_key[relation]
        src_type, _, dest_type = relation
        src = embeddings[src_type][edge_index[0]]
        dst = embeddings[dest_type][edge_index[1]]
        edge_emb = self.edge_projections[relation_key](edge_attr)
        features = torch.cat([src, dst, edge_emb], dim=1)
        return self.edge_classifiers[relation_key](features)

    def forward(self, batch: HeteroData, relation):
        embeddings = self.encode_nodes(batch.x_dict, batch.edge_index_dict)
        return self.classify_relation(
            embeddings,
            relation,
            batch[relation].edge_label_index,
            batch[relation].edge_label_attr,
        )


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
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


def save_model_checkpoint(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def assert_neighbor_sampler_available() -> None:
    if pyg_typing.WITH_PYG_LIB or pyg_typing.WITH_TORCH_SPARSE:
        return
    raise ImportError(
        "Full hetero graph training uses LinkNeighborLoader, which requires either "
        "pyg-lib or torch-sparse. In this environment both are unavailable or failed "
        "to import. Reinstall a torch-compatible torch-sparse/pyg-lib build before "
        "starting training."
    )


def evaluate_relation(model, data: HeteroData, relation, edge_indices: np.ndarray, config: dict, device):
    loader = build_relation_loader(data, relation, edge_indices, config, shuffle=False)
    model.eval()
    preds = []
    probs = []
    truth = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            logits = model(batch, relation)
            pred = logits.argmax(dim=-1)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            target = batch[relation].edge_label
            preds.append(pred.detach().cpu())
            probs.append(prob.detach().cpu())
            truth.append(target.detach().cpu())
    y_true = torch.cat(truth).numpy()
    y_pred = torch.cat(preds).numpy()
    y_prob = torch.cat(probs).numpy()
    return y_true, y_pred, y_prob


def train_relation_model(relation, relation_frame: pd.DataFrame, data: HeteroData, split_indices: dict[str, np.ndarray], config: dict) -> None:
    relation_name = "__".join(relation)
    model_dir = Path(config["output_root"]) / config["experiment_name"] / relation_name
    logger = setup_logger(model_dir / "training.log")
    logger.info("Starting relation %s", relation_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullHeteroEdgeModel(
        data=data,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        final_dropout=config["final_dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(config["positive_class_weight"])], dtype=torch.float32, device=device)
    )
    amp_enabled, amp_dtype = autocast_settings(device)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    logger.info("Mixed precision enabled=%s dtype=%s", amp_enabled, amp_dtype)

    train_loader = build_relation_loader(data, relation, split_indices["train"], config, shuffle=True)
    best_state = None
    best_val_f1 = -1.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopping_patience = int(config.get("early_stopping_patience", 0))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
    checkpoint_every_epochs = int(config.get("checkpoint_every_epochs", 2))
    use_tqdm = bool(config.get("use_tqdm", True))

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = torch.zeros((), device=device)
        total_examples = 0
        epoch_batches = tqdm(
            train_loader,
            desc=f"{relation_name} epoch {epoch + 1}/{config['epochs']}",
            unit="batch",
            leave=False,
            disable=not use_tqdm,
        )
        for batch in epoch_batches:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=amp_dtype) if amp_enabled else nullcontext()
            with context:
                logits = model(batch, relation)
                target = batch[relation].edge_label
                loss = loss_fn(logits, target)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.detach() * target.numel()
            total_examples += target.numel()
            epoch_batches.set_postfix(edges=int(target.numel()))

        y_true, y_pred, y_prob = evaluate_relation(
            model, data, relation, split_indices["validation"], config, device
        )
        val_threshold = best_f1_threshold(y_true, y_prob)
        val_metrics = compute_metrics(y_true, y_prob, val_threshold)
        logger.info(
            "epoch=%s loss=%.6f val_f1=%.6f val_pr_auc=%.6f val_threshold=%.6f",
            epoch + 1,
            float((total_loss / max(total_examples, 1)).detach().cpu()),
            val_metrics["f1"],
            val_metrics["pr_auc"],
            val_metrics["threshold"],
        )
        if checkpoint_every_epochs > 0 and (epoch + 1) % checkpoint_every_epochs == 0:
            checkpoint_path = model_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            save_model_checkpoint(
                {
                    "relation": relation,
                    "epoch": epoch + 1,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "config": config,
                    "validation_metrics": val_metrics,
                },
                checkpoint_path,
            )
            logger.info("saved checkpoint=%s", checkpoint_path)

        if val_metrics["f1"] > best_val_f1 + early_stopping_min_delta:
            best_val_f1 = val_metrics["f1"]
            best_threshold = val_threshold
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            best_checkpoint_path = model_dir / "best_checkpoint.pt"
            save_model_checkpoint(
                {
                    "relation": relation,
                    "epoch": best_epoch,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "config": config,
                    "validation_metrics": val_metrics,
                    "threshold": best_threshold,
                },
                best_checkpoint_path,
            )
            logger.info("saved best_checkpoint=%s", best_checkpoint_path)
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "early stopping at epoch=%s best_epoch=%s best_val_f1=%.6f",
                    epoch + 1,
                    best_epoch,
                    best_val_f1,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_true, _, val_prob = evaluate_relation(
        model, data, relation, split_indices["validation"], config, device
    )
    val_pred = (val_prob >= best_threshold).astype(np.int64)

    y_true, y_pred, y_prob = evaluate_relation(
        model, data, relation, split_indices["test"], config, device
    )
    y_pred = (y_prob >= best_threshold).astype(np.int64)
    metrics = compute_metrics(y_true, y_prob, best_threshold)
    logger.info("test_metrics=%s", json.dumps(metrics))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "relation": relation,
            "state_dict": {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
            "config": config,
            "threshold": best_threshold,
        },
        model_dir / "model.joblib",
    )

    prediction_rows = relation_frame.iloc[split_indices["test"]][["src", "dest", "timestamp"]].copy()
    prediction_rows["y_true"] = y_true
    prediction_rows["y_pred"] = y_pred
    prediction_rows["fraud_score"] = y_prob
    prediction_rows.to_csv(model_dir / "test_predictions.csv", index=False)
    validation_rows = relation_frame.iloc[split_indices["validation"]][["src", "dest", "timestamp"]].copy()
    validation_rows["y_true"] = val_true
    validation_rows["y_pred"] = val_pred
    validation_rows["fraud_score"] = val_prob
    validation_rows.to_csv(model_dir / "validation_predictions.csv", index=False)
    save_json(metrics, model_dir / "test_metrics.json")
    save_confusion_matrix(y_true, y_pred, model_dir / "confusion_matrix.png")
    logger.info("Finished relation %s", relation_name)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    config = apply_cli_overrides(config, args)
    hetero, metadata, relation_splits = build_full_hetero_datasets(config)
    if args.dry_run:
        print("[full-hetero] dry run complete; exiting before training", flush=True)
        return
    assert_neighbor_sampler_available()
    selected_relations = set(config.get("relations", []))
    for relation in metadata["transaction_relations"]:
        relation_key = "__".join(relation)
        if selected_relations and relation_key not in selected_relations:
            print(f"[full-hetero] skipping relation={relation_key}", flush=True)
            continue
        train_relation_model(
            relation,
            metadata["relation_tables"][relation_key],
            hetero,
            relation_splits[relation],
            config,
        )


if __name__ == "__main__":
    main()
