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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, Linear, SAGEConv

from train_graph_models import autocast_settings, load_or_build_full_hetero_graph, read_config, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full heterogeneous graph fraud models.")
    parser.add_argument(
        "--config",
        default="configs/full_hetero_graph_li.json",
        help="Path to the full-hetero graph JSON config.",
    )
    return parser.parse_args()


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

    for start in all_days:
        for end in all_days:
            if end < start:
                continue
            split_totals = [totals[:start].sum(), totals[start:end].sum(), totals[end:].sum()]
            split_sum = max(np.sum(split_totals), 1)
            split_props = [value / split_sum for value in split_totals]
            split_errors = [abs(value - target) / target for value, target in zip(split_props, split_target)]
            split_scores[(start, end)] = max(split_errors)

    start, end = min(split_scores, key=split_scores.get)
    return list(range(start)), list(range(start, end)), list(range(end, len(totals)))


def compute_relation_splits(relation_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    timestamps = relation_frame["timestamp_seconds"].fillna(0).to_numpy()
    train_days, val_days, test_days = select_temporal_split_days(timestamps)
    day_indices: list[np.ndarray] = []
    n_days = int(timestamps.max() / (3600 * 24) + 1) if len(timestamps) else 0
    for day in range(n_days):
        left = day * 24 * 3600
        right = (day + 1) * 24 * 3600
        day_indices.append(np.where((timestamps >= left) & (timestamps < right))[0])

    train_idx = np.concatenate([day_indices[day] for day in train_days]) if train_days else np.array([], dtype=int)
    val_idx = np.concatenate([day_indices[day] for day in val_days]) if val_days else np.array([], dtype=int)
    test_idx = np.concatenate([day_indices[day] for day in test_days]) if test_days else np.array([], dtype=int)
    return {"train": train_idx, "validation": val_idx, "test": test_idx}


def build_full_hetero_datasets(config: dict):
    hetero, metadata = load_or_build_full_hetero_graph(config)
    relation_splits: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for relation in metadata["transaction_relations"]:
        relation_key = "__".join(relation)
        relation_splits[relation] = compute_relation_splits(metadata["relation_tables"][relation_key])
    return hetero, metadata, relation_splits


def build_relation_loader(data: HeteroData, relation, edge_indices: np.ndarray, config: dict, shuffle: bool):
    edge_index = data[relation].edge_index[:, edge_indices]
    edge_label = data[relation].y[edge_indices]
    return LinkNeighborLoader(
        data=data,
        num_neighbors={key: [config["num_neighbors"]] * config["num_layers"] for key in data.edge_types},
        edge_label_index=(relation, edge_index),
        edge_label=edge_label,
        batch_size=config["batch_size"],
        shuffle=shuffle,
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
            updated = conv(embeddings, edge_index_dict)
            embeddings = {
                node_type: F.dropout(F.relu(value), p=self.dropout, training=self.training)
                for node_type, value in updated.items()
            }
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
            batch[relation].edge_attr,
        )


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


def save_model_checkpoint(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


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
    checkpoint_every_epochs = int(config.get("checkpoint_every_epochs", 2))

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = torch.zeros((), device=device)
        total_examples = 0
        for batch in train_loader:
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

        y_true, y_pred, y_prob = evaluate_relation(
            model, data, relation, split_indices["validation"], config, device
        )
        val_metrics = compute_metrics(y_true, y_pred, y_prob)
        logger.info(
            "epoch=%s loss=%.6f val_f1=%.6f val_pr_auc=%.6f",
            epoch + 1,
            float((total_loss / max(total_examples, 1)).detach().cpu()),
            val_metrics["f1"],
            val_metrics["pr_auc"],
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

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, y_prob = evaluate_relation(
        model, data, relation, split_indices["test"], config, device
    )
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info("test_metrics=%s", json.dumps(metrics))

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "relation": relation,
            "state_dict": {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()},
            "config": config,
        },
        model_dir / "model.joblib",
    )

    prediction_rows = relation_frame.iloc[split_indices["test"]][["src", "dest", "timestamp"]].copy()
    prediction_rows["y_true"] = y_true
    prediction_rows["y_pred"] = y_pred
    prediction_rows["fraud_score"] = y_prob
    prediction_rows.to_csv(model_dir / "test_predictions.csv", index=False)
    save_json(metrics, model_dir / "test_metrics.json")
    save_confusion_matrix(y_true, y_pred, model_dir / "confusion_matrix.png")
    logger.info("Finished relation %s", relation_name)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    hetero, metadata, relation_splits = build_full_hetero_datasets(config)
    for relation in metadata["transaction_relations"]:
        relation_key = "__".join(relation)
        train_relation_model(
            relation,
            metadata["relation_tables"][relation_key],
            hetero,
            relation_splits[relation],
            config,
        )


if __name__ == "__main__":
    main()
