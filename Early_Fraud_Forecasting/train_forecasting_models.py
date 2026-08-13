from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_COLUMNS = [
    "src_risk_score",
    "dest_risk_score",
    "src_occupation",
    "dest_occupation",
    "src_is_high_risk_category",
    "dest_is_high_risk_category",
    "src_is_high_risk_country",
    "dest_is_high_risk_country",
]

IDENTIFIER_COLUMNS = ["src", "dest", "src_institution_id", "dest_institution_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train early fraud forecasting models from transaction history."
    )
    parser.add_argument(
        "--config",
        default="configs/early_fraud_forecasting.json",
        help="Path to the early fraud forecasting JSON config.",
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


def seconds_since(series: pd.Series, reference: pd.Series) -> pd.Series:
    return (reference - series).dt.total_seconds()


def compute_next_fraud_time(
    entity: pd.Series,
    timestamp: pd.Series,
    fraud_events: pd.DataFrame,
) -> pd.Series:
    result = np.full(len(entity), np.datetime64("NaT"), dtype="datetime64[ns]")
    if fraud_events.empty:
        return pd.Series(result, index=entity.index)

    event_lookup = {
        key: group["fraud_timestamp"].to_numpy(dtype="datetime64[ns]")
        for key, group in fraud_events.groupby("entity", sort=False)
    }
    entity_values = entity.to_numpy()
    timestamp_values = timestamp.to_numpy(dtype="datetime64[ns]")
    grouped_indices = pd.Series(np.arange(len(entity_values))).groupby(entity_values, sort=False)

    for key, row_indices in grouped_indices:
        events = event_lookup.get(key)
        if events is None or len(events) == 0:
            continue
        indices = row_indices.to_numpy()
        positions = np.searchsorted(events, timestamp_values[indices], side="right")
        valid = positions < len(events)
        result[indices[valid]] = events[positions[valid]]

    return pd.Series(result, index=entity.index)


def build_future_target(
    transactions: pd.DataFrame, forecast_horizon_days: int
) -> tuple[pd.Series, pd.Series]:
    fraud_transactions = transactions.loc[transactions["current_is_fraud"] == 1]
    fraud_events = pd.concat(
        [
            fraud_transactions[["src", "timestamp"]].rename(
                columns={"src": "entity", "timestamp": "fraud_timestamp"}
            ),
            fraud_transactions[["dest", "timestamp"]].rename(
                columns={"dest": "entity", "timestamp": "fraud_timestamp"}
            ),
        ],
        ignore_index=True,
    ).dropna()
    fraud_events = fraud_events.sort_values(["entity", "fraud_timestamp"], kind="stable")

    next_src_fraud = compute_next_fraud_time(
        transactions["src"], transactions["timestamp"], fraud_events
    )
    next_dest_fraud = compute_next_fraud_time(
        transactions["dest"], transactions["timestamp"], fraud_events
    )
    next_any_fraud = next_src_fraud.combine(next_dest_fraud, min, fill_value=pd.NaT)
    horizon = pd.Timedelta(days=forecast_horizon_days)
    lead_time = next_any_fraud - transactions["timestamp"]
    target = lead_time.notna() & lead_time.gt(pd.Timedelta(0)) & lead_time.le(horizon)
    return target.astype(int), lead_time.dt.total_seconds() / 86400.0


def add_time_features(transactions: pd.DataFrame) -> None:
    transactions["transaction_hour"] = transactions["timestamp"].dt.hour
    transactions["transaction_dayofweek"] = transactions["timestamp"].dt.dayofweek
    transactions["transaction_day"] = transactions["timestamp"].dt.day
    transactions["transaction_month"] = transactions["timestamp"].dt.month
    transactions["transaction_is_weekend"] = (
        transactions["transaction_dayofweek"].fillna(-1).ge(5).astype(int)
    )
    transactions["time_since_previous_transaction_seconds"] = pd.to_timedelta(
        transactions["time_since_previous_transaction"], errors="coerce"
    ).dt.total_seconds()
    ownership_start = pd.to_datetime(
        transactions["ownership_start_date"], errors="coerce"
    )
    transactions["ownership_start_date_days_before_txn"] = (
        transactions["timestamp"] - ownership_start
    ).dt.total_seconds() / 86400.0


def add_history_features(transactions: pd.DataFrame, group_columns: list[str], prefix: str) -> None:
    group = transactions.groupby(group_columns, sort=False)
    prior_count = group.cumcount()
    amount = pd.to_numeric(transactions["amount"], errors="coerce").fillna(0.0)
    current_fraud = transactions["current_is_fraud"].fillna(0).astype(int)

    prior_amount_sum = group["amount_numeric"].cumsum() - amount
    prior_fraud_count = group["current_is_fraud"].cumsum() - current_fraud
    previous_timestamp = group["timestamp"].shift(1)

    transactions[f"{prefix}_prior_txn_count"] = prior_count
    transactions[f"{prefix}_prior_amount_sum"] = prior_amount_sum
    transactions[f"{prefix}_prior_amount_mean"] = (
        prior_amount_sum / prior_count.replace(0, np.nan)
    )
    transactions[f"{prefix}_prior_fraud_count"] = prior_fraud_count
    transactions[f"{prefix}_seconds_since_previous_txn"] = seconds_since(
        previous_timestamp, transactions["timestamp"]
    )


def add_entity_rolling_features(transactions: pd.DataFrame, entity_column: str, prefix: str) -> None:
    amount = transactions["amount_numeric"].fillna(0.0)
    current_fraud = transactions["current_is_fraud"].fillna(0).astype(int)
    entity_frame = pd.DataFrame(
        {
            "entity": transactions[entity_column].astype(str),
            "timestamp": transactions["timestamp"],
            "amount": amount,
            "fraud": current_fraud,
            "row_index": np.arange(len(transactions)),
        }
    ).sort_values(["entity", "timestamp", "row_index"], kind="stable")

    for window in ("1D", "7D", "30D"):
        rolled = (
            entity_frame.set_index("timestamp")
            .groupby("entity", sort=False)[["amount", "fraud"]]
            .rolling(window, closed="left")
            .agg({"amount": ["count", "sum", "mean", "max"], "fraud": ["sum"]})
        )
        rolled.columns = [
            f"{prefix}_{window.lower()}_txn_count",
            f"{prefix}_{window.lower()}_amount_sum",
            f"{prefix}_{window.lower()}_amount_mean",
            f"{prefix}_{window.lower()}_amount_max",
            f"{prefix}_{window.lower()}_fraud_count",
        ]
        rolled = rolled.reset_index(level=0, drop=True).reset_index()
        rolled["row_index"] = entity_frame["row_index"].to_numpy()
        rolled = rolled.set_index("row_index").sort_index()
        for column in rolled.columns:
            if column != "timestamp":
                transactions[column] = rolled[column]


def prepare_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    for column in nodes.columns:
        if nodes[column].dtype == "object":
            nodes[column] = nodes[column].map(normalize_token)

    nodes["creation_date"] = pd.to_datetime(nodes["creation_date"], errors="coerce")
    nodes["node_creation_year"] = nodes["creation_date"].dt.year
    nodes["node_creation_month"] = nodes["creation_date"].dt.month
    nodes["incorporation_year"] = pd.to_numeric(nodes["incorporation_year"], errors="coerce")
    nodes["risk_score"] = pd.to_numeric(nodes["risk_score"], errors="coerce")
    for column in ["is_high_risk_category", "is_high_risk_country"]:
        if column in nodes.columns:
            nodes[column] = (
                nodes[column].astype(str).str.strip().str.lower().map(
                    {"true": 1, "false": 0, "1": 1, "0": 0}
                )
            )
    return nodes


def attach_entity_embeddings(
    data: pd.DataFrame,
    embeddings_path: str,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    embeddings = pd.read_csv(embeddings_path)
    required_columns = {"entity_id", "as_of_timestamp"}
    missing_columns = required_columns - set(embeddings.columns)
    if missing_columns:
        raise ValueError(
            "Embedding file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    embeddings["as_of_timestamp"] = pd.to_datetime(
        embeddings["as_of_timestamp"], errors="coerce"
    )
    embeddings = embeddings.dropna(subset=["entity_id", "as_of_timestamp"])
    embedding_columns = [
        column
        for column in embeddings.columns
        if column.startswith("embedding_")
    ]
    if not embedding_columns:
        raise ValueError("Embedding file does not contain embedding_* columns.")

    embeddings = embeddings.sort_values(["as_of_timestamp", "entity_id"], kind="stable")

    def merge_side(frame: pd.DataFrame, entity_column: str, prefix: str) -> pd.DataFrame:
        left = frame.reset_index().rename(
            columns={entity_column: "entity_id", timestamp_column: "as_of_timestamp"}
        )
        left = left.sort_values(["as_of_timestamp", "entity_id"], kind="stable")
        merged = pd.merge_asof(
            left,
            embeddings,
            on="as_of_timestamp",
            by="entity_id",
            direction="backward",
        )
        rename_columns = {
            column: f"{prefix}_{column}"
            for column in embedding_columns
        }
        merged = merged.rename(columns=rename_columns).set_index("index")
        return merged[[rename_columns[column] for column in embedding_columns]].sort_index()

    print(f"[data] joining graph embeddings from {embeddings_path}")
    src_embeddings = merge_side(data, "src", "src")
    dest_embeddings = merge_side(data, "dest", "dest")
    data = data.join(src_embeddings).join(dest_embeddings)

    src_columns = [f"src_{column}" for column in embedding_columns]
    dest_columns = [f"dest_{column}" for column in embedding_columns]
    data["src_dest_embedding_dot"] = (
        data[src_columns].to_numpy(dtype=float) * data[dest_columns].to_numpy(dtype=float)
    ).sum(axis=1)
    data["src_dest_embedding_l2"] = np.sqrt(
        np.square(
            data[src_columns].to_numpy(dtype=float)
            - data[dest_columns].to_numpy(dtype=float)
        ).sum(axis=1)
    )
    return data


def attach_gnn_score_features(data: pd.DataFrame, score_configs: list[dict]) -> pd.DataFrame:
    if not score_configs:
        return data

    base_keys = data[["src", "dest", "timestamp"]].copy()
    base_keys["timestamp"] = pd.to_datetime(base_keys["timestamp"], errors="coerce")
    for index, score_config in enumerate(score_configs):
        path = Path(score_config["path"])
        name = score_config.get("name", path.parent.name or f"gnn_{index}")
        score_column = score_config.get("score_column", "fraud_score")
        frame = pd.read_csv(path)
        required = {"src", "dest", "timestamp", score_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"GNN score file {path} is missing columns: {', '.join(sorted(missing))}")
        scores = frame[["src", "dest", "timestamp", score_column]].copy()
        scores["timestamp"] = pd.to_datetime(scores["timestamp"], errors="coerce")
        scores = scores.rename(columns={score_column: f"{name}_score"})
        merged = base_keys.merge(scores, on=["src", "dest", "timestamp"], how="left")
        data[f"{name}_score"] = merged[f"{name}_score"]
    return data


def load_and_prepare_data(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    processed_data_path = Path(config["processed_data_path"])
    processed_target_path = Path(config["processed_target_path"])
    processed_keys_path = Path(config["processed_keys_path"])

    if (
        processed_data_path.exists()
        and processed_target_path.exists()
        and processed_keys_path.exists()
        and not config.get("rebuild_processed", False)
    ):
        print(f"[data] loading cached features from {processed_data_path}")
        features = pd.read_pickle(processed_data_path)
        print(f"[data] loading cached target from {processed_target_path}")
        target = pd.read_pickle(processed_target_path)
        print(f"[data] loading cached keys from {processed_keys_path}")
        keys = pd.read_pickle(processed_keys_path)
        return features, target, keys

    print(f"[data] reading transactions from {config['transactions_path']}")
    transactions = pd.read_csv(config["transactions_path"], low_memory=False)
    print(f"[data] reading nodes from {config['nodes_path']}")
    nodes = pd.read_csv(config["nodes_path"], low_memory=False)

    print("[data] filtering transaction rows")
    transactions = transactions.loc[transactions["edge_type"] == "transaction"].copy()
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="coerce")
    transactions = transactions.dropna(subset=["timestamp"]).sort_values(
        "timestamp", kind="stable"
    )
    transactions = transactions.reset_index(drop=True)
    transactions["current_is_fraud"] = parse_bool_series(transactions["is_fraudulent"])
    transactions["amount_numeric"] = pd.to_numeric(transactions["amount"], errors="coerce")

    print("[data] normalizing transaction string tokens")
    for column in transactions.columns:
        if transactions[column].dtype == "object":
            transactions[column] = transactions[column].map(normalize_token)

    print("[data] building future fraud target")
    target, lead_time_days = build_future_target(
        transactions, int(config["forecast_horizon_days"])
    )
    keys = transactions[["src", "dest", "timestamp", "current_is_fraud"]].copy()
    keys["forecast_horizon_days"] = int(config["forecast_horizon_days"])
    keys["future_fraud_lead_time_days"] = lead_time_days

    print("[data] creating as-of transaction and history features")
    add_time_features(transactions)
    add_history_features(transactions, ["src"], "src")
    add_history_features(transactions, ["dest"], "dest")
    add_history_features(transactions, ["src", "dest"], "pair")
    if config.get("add_rolling_features", False):
        print("[data] adding temporal leakage-safe rolling entity features")
        add_entity_rolling_features(transactions, "src", "src")
        add_entity_rolling_features(transactions, "dest", "dest")

    if not config.get("include_current_fraud_observations", False):
        print("[data] excluding currently fraudulent rows from forecasting observations")
        keep_mask = transactions["current_is_fraud"] == 0
        transactions = transactions.loc[keep_mask].reset_index(drop=True)
        target = target.loc[keep_mask].reset_index(drop=True)
        keys = keys.loc[keep_mask].reset_index(drop=True)
    else:
        target = target.reset_index(drop=True)
        keys = keys.reset_index(drop=True)

    print("[data] creating node features")
    nodes = prepare_nodes(nodes)
    src_nodes = nodes.drop(columns=["name", "is_fraudulent"], errors="ignore").add_prefix("src_")
    src_nodes = src_nodes.rename(columns={"src_node_id": "src"})
    dest_nodes = nodes.drop(columns=["name", "is_fraudulent"], errors="ignore").add_prefix("dest_")
    dest_nodes = dest_nodes.rename(columns={"dest_node_id": "dest"})

    print("[data] joining source and destination node features")
    data = transactions.merge(src_nodes, on="src", how="left")
    data = data.merge(dest_nodes, on="dest", how="left")

    print("[data] creating pairwise features")
    data["src_dest_same_country"] = (
        data["src_country_code"].fillna("__missing__")
        == data["dest_country_code"].fillna("__missing__")
    ).astype(int)
    data["src_dest_same_currency"] = (
        data["src_currency"].fillna("__missing__")
        == data["dest_currency"].fillna("__missing__")
    ).astype(int)
    data["src_dest_risk_score_gap"] = (
        pd.to_numeric(data["src_risk_score"], errors="coerce")
        - pd.to_numeric(data["dest_risk_score"], errors="coerce")
    )
    data["amount_to_src_risk"] = pd.to_numeric(data["amount"], errors="coerce") / (
        pd.to_numeric(data["src_risk_score"], errors="coerce").replace({0: np.nan})
    )
    data["amount_to_dest_risk"] = pd.to_numeric(data["amount"], errors="coerce") / (
        pd.to_numeric(data["dest_risk_score"], errors="coerce").replace({0: np.nan})
    )

    embeddings_path = config.get("entity_embeddings_path")
    if embeddings_path:
        data = attach_entity_embeddings(data, embeddings_path)
    data = attach_gnn_score_features(data, config.get("gnn_score_features", []))

    print("[data] dropping unused columns from final forecasting feature table")
    features = data.drop(
        columns=[
            "edge_type",
            "is_fraudulent",
            "current_is_fraud",
            "timestamp",
            "ownership_start_date",
            "time_since_previous_transaction",
            "amount_numeric",
            "src_creation_date",
            "dest_creation_date",
        ],
        errors="ignore",
    )

    print(f"[data] saving processed features to {processed_data_path}")
    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_pickle(processed_data_path)
    print(f"[data] saving processed target to {processed_target_path}")
    target.to_pickle(processed_target_path)
    print(f"[data] saving processed keys to {processed_keys_path}")
    keys.to_pickle(processed_keys_path)
    print(
        "[data] processed forecasting data ready: "
        f"{len(features)} rows, {int(target.sum())} positives"
    )
    return features, target, keys


def split_data(
    features: pd.DataFrame, target: pd.Series, keys: pd.DataFrame, config: dict
) -> dict[str, pd.DataFrame | pd.Series]:
    total_rows = len(features)
    train_end = int(total_rows * config["train_fraction"])
    validation_end = train_end + int(total_rows * config["validation_fraction"])
    return {
        "X_train": features.iloc[:train_end].reset_index(drop=True),
        "y_train": target.iloc[:train_end].reset_index(drop=True),
        "X_validation": features.iloc[train_end:validation_end].reset_index(drop=True),
        "y_validation": target.iloc[train_end:validation_end].reset_index(drop=True),
        "X_test": features.iloc[validation_end:].reset_index(drop=True),
        "y_test": target.iloc[validation_end:].reset_index(drop=True),
        "keys_validation": keys.iloc[train_end:validation_end].reset_index(drop=True),
        "keys_test": keys.iloc[validation_end:].reset_index(drop=True),
    }


def build_preprocessor(X_train: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical_columns = list(X_train.select_dtypes(include=["object"]).columns)
    numeric_columns = [column for column in X_train.columns if column not in categorical_columns]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_model(model_name: str, model_params: dict):
    model_params = dict(model_params)
    if model_name == "logistic_regression":
        return LogisticRegression(**model_params)
    if model_name == "random_forest":
        return RandomForestClassifier(**model_params)
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**model_params)
    if model_name == "isolation_forest":
        return IsolationForest(**model_params)
    if model_name == "local_outlier_factor":
        return LocalOutlierFactor(**model_params)
    raise ValueError(f"Unsupported model: {model_name}")


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


def save_json(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Future Fraud", "Future Fraud"])
    ax.set_yticklabels(["No Future Fraud", "Future Fraud"])
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def add_threshold_metrics(
    metrics: dict,
    y_true: pd.Series,
    scores: pd.Series | None,
    thresholds: list[float],
) -> None:
    if scores is None:
        return
    threshold_metrics = {}
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        threshold_metrics[str(threshold)] = {
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "flagged_rows": int(predictions.sum()),
        }
    metrics["threshold_metrics"] = threshold_metrics


def best_f1_threshold(y_true: pd.Series, scores: pd.Series | np.ndarray | None) -> float:
    if scores is None or y_true.nunique() < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_binary_metrics(
    y_true: pd.Series,
    predictions: pd.Series,
    scores: pd.Series | None,
) -> dict:
    metrics = {
        "rows": int(len(y_true)),
        "positive_rows": int(y_true.sum()),
        "predicted_positive_rows": int(predictions.sum()),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }
    if scores is not None and y_true.nunique() > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    return metrics


def apply_auto_model_params(model_name: str, model_config: dict, y_train: pd.Series) -> dict:
    model_config = dict(model_config)
    params = dict(model_config["params"])
    if model_name == "xgboost" and params.get("scale_pos_weight") == "auto":
        positives = max(int(y_train.sum()), 1)
        negatives = max(int(len(y_train) - y_train.sum()), 1)
        params["scale_pos_weight"] = negatives / positives
    model_config["params"] = params
    return model_config


def prepare_experiment_features(
    features: pd.DataFrame, experiment_config: dict
) -> pd.DataFrame:
    experiment_features = features.copy()
    if experiment_config.get("drop_identifier_features", True):
        print("[experiment] dropping identifier features")
        experiment_features = experiment_features.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    if experiment_config.get("drop_leakage_features", False):
        print("[experiment] dropping leakage-prone features")
        experiment_features = experiment_features.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    return experiment_features


def train_and_save_one_model(
    model_name: str,
    model_config: dict,
    split: dict[str, pd.DataFrame | pd.Series],
    output_root: Path,
    thresholds: list[float],
) -> None:
    model_dir = output_root / model_name
    logger = setup_logger(model_dir / "training.log")
    logger.info("Starting %s", model_name)

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_validation = split["X_validation"]
    y_validation = split["y_validation"]
    X_test = split["X_test"]
    y_test = split["y_test"]
    keys_validation = split["keys_validation"].copy()
    keys_test = split["keys_test"].copy()

    scale_numeric = model_name == "logistic_regression"
    preprocessor = build_preprocessor(X_train, scale_numeric=scale_numeric)
    model_config = apply_auto_model_params(model_name, model_config, y_train)
    model = build_model(model_name, model_config["params"])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_config["type"] == "unsupervised":
        normal_mask = y_train == 0
        pipeline.fit(X_train.loc[normal_mask].reset_index(drop=True))
        raw_validation_predictions = pipeline.predict(X_validation)
        validation_predictions = pd.Series(np.where(np.asarray(raw_validation_predictions) == -1, 1, 0))
        raw_test_predictions = pipeline.predict(X_test)
        predictions = pd.Series(np.where(np.asarray(raw_test_predictions) == -1, 1, 0))
        if hasattr(pipeline, "decision_function"):
            validation_scores = pd.Series(-np.asarray(pipeline.decision_function(X_validation)))
            scores = pd.Series(-np.asarray(pipeline.decision_function(X_test)))
        elif hasattr(pipeline, "score_samples"):
            validation_scores = pd.Series(-np.asarray(pipeline.score_samples(X_validation)))
            scores = pd.Series(-np.asarray(pipeline.score_samples(X_test)))
        else:
            validation_scores = None
            scores = None
    else:
        pipeline.fit(X_train, y_train)
        if hasattr(pipeline, "predict_proba"):
            validation_scores = pd.Series(pipeline.predict_proba(X_validation)[:, 1])
            scores = pd.Series(pipeline.predict_proba(X_test)[:, 1])
        elif hasattr(pipeline, "decision_function"):
            validation_scores = pd.Series(pipeline.decision_function(X_validation))
            scores = pd.Series(pipeline.decision_function(X_test))
        else:
            validation_scores = None
            scores = None
        threshold = best_f1_threshold(y_validation, validation_scores)
        validation_predictions = pd.Series((validation_scores >= threshold).astype(int)) if validation_scores is not None else pd.Series(pipeline.predict(X_validation)).astype(int)
        predictions = pd.Series((scores >= threshold).astype(int)) if scores is not None else pd.Series(pipeline.predict(X_test)).astype(int)

    if model_config["type"] == "unsupervised":
        threshold = None

    validation_metrics = compute_binary_metrics(y_validation, validation_predictions, validation_scores)
    metrics = compute_binary_metrics(y_test, predictions, scores)
    metrics = {f"test_{key}": value for key, value in metrics.items()}
    metrics["validation"] = validation_metrics
    if threshold is not None:
        metrics["validation_best_f1_threshold"] = float(threshold)
    add_threshold_metrics(metrics, y_test, scores, thresholds)

    save_confusion_matrix(y_test, predictions, model_dir / "confusion_matrix.png")
    logger.info("Saved confusion matrix")
    joblib.dump(pipeline, model_dir / "model.joblib")
    logger.info("Saved model")

    results = keys_test.copy()
    results["y_true_future_fraud"] = y_test
    results["y_pred_future_fraud"] = predictions
    if scores is not None:
        results["future_fraud_score"] = scores
    results.to_csv(model_dir / "test_predictions.csv", index=False)
    logger.info("Saved test predictions")

    validation_results = keys_validation.copy()
    validation_results["y_true_future_fraud"] = y_validation
    validation_results["y_pred_future_fraud"] = validation_predictions
    if validation_scores is not None:
        validation_results["future_fraud_score"] = validation_scores
    validation_results.to_csv(model_dir / "validation_predictions.csv", index=False)
    logger.info("Saved validation predictions")

    save_json(metrics, model_dir / "test_metrics.json")
    logger.info("Saved test metrics")


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    features, target, keys = load_and_prepare_data(config)
    experiments = config.get(
        "experiments",
        [
            {
                "name": config["experiment_name"],
                "drop_identifier_features": True,
                "drop_leakage_features": False,
            }
        ],
    )

    for experiment_config in experiments:
        experiment_name = experiment_config["name"]
        print(f"[experiment] starting {experiment_name}")
        experiment_features = prepare_experiment_features(features, experiment_config)
        split = split_data(experiment_features, target, keys, config)
        output_root = Path(config["output_root"]) / experiment_name
        thresholds = [float(value) for value in config.get("score_thresholds", [0.5])]

        for model_name, model_config in config["models"].items():
            if model_config.get("enabled", True):
                train_and_save_one_model(
                    model_name, model_config, split, output_root, thresholds
                )


if __name__ == "__main__":
    main()
