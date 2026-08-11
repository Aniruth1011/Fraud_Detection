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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical fraud models.")
    parser.add_argument(
        "--config",
        default="configs/classical_models.json",
        help="Path to the classical-model JSON config.",
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
        .astype(int)
    )


def load_and_prepare_data(config: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    processed_data_path = Path(config["processed_data_path"])
    processed_target_path = Path(config["processed_target_path"])
    processed_keys_path = Path(config["processed_keys_path"])

    if (
        processed_data_path.exists()
        and processed_target_path.exists()
        and processed_keys_path.exists()
    ):
        print(f"[data] loading cached features from {processed_data_path}")
        features = pd.read_pickle(processed_data_path)
        print(f"[data] loading cached target from {processed_target_path}")
        target = pd.read_pickle(processed_target_path)
        print(f"[data] loading cached keys from {processed_keys_path}")
        keys = pd.read_pickle(processed_keys_path)
        datetime_columns = list(features.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
        if datetime_columns:
            print(
                "[data] cached features still contain datetime columns, rebuilding cache: "
                + ", ".join(datetime_columns)
            )
        else:
            print(f"[data] cached processed data loaded: {len(features)} rows")
            return features, target, keys

    print(f"[data] reading transactions from {config['transactions_path']}")
    transactions = pd.read_csv(config["transactions_path"])
    print(f"[data] reading nodes from {config['nodes_path']}")
    nodes = pd.read_csv(config["nodes_path"])

    print("[data] filtering transaction rows")
    transactions = transactions.loc[transactions["edge_type"] == "transaction"].copy()
    print("[data] building target column")
    target = parse_bool_series(transactions["is_fraudulent"])
    print("[data] capturing transaction keys")
    keys = transactions[["src", "dest", "timestamp"]].copy()

    print("[data] normalizing transaction string tokens")
    for column in transactions.columns:
        if transactions[column].dtype == "object":
            transactions[column] = transactions[column].map(normalize_token)

    print("[data] normalizing node string tokens")
    for column in nodes.columns:
        if nodes[column].dtype == "object":
            nodes[column] = nodes[column].map(normalize_token)

    print("[data] creating transaction time features")
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="coerce")
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

    print("[data] creating node features")
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

    print("[data] preparing source and destination node tables")
    src_nodes = nodes.drop(columns=["name", "is_fraudulent"], errors="ignore").add_prefix("src_")
    src_nodes = src_nodes.rename(columns={"src_node_id": "src"})
    dest_nodes = nodes.drop(columns=["name", "is_fraudulent"], errors="ignore").add_prefix("dest_")
    dest_nodes = dest_nodes.rename(columns={"dest_node_id": "dest"})

    print("[data] joining source node features")
    data = transactions.merge(src_nodes, on="src", how="left")
    print("[data] joining destination node features")
    data = data.merge(dest_nodes, on="dest", how="left")

    print("[data] creating pairwise fraud features")
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
        data["src_risk_score"].replace({0: np.nan})
    )
    data["amount_to_dest_risk"] = pd.to_numeric(data["amount"], errors="coerce") / (
        data["dest_risk_score"].replace({0: np.nan})
    )

    print("[data] sorting by timestamp")
    order = data["timestamp"].sort_values(kind="stable").index
    data = data.loc[order].reset_index(drop=True)
    target = target.loc[order].reset_index(drop=True)
    keys = keys.loc[order].reset_index(drop=True)

    print("[data] dropping unused columns from final feature table")
    features = data.drop(
        columns=[
            "edge_type",
            "is_fraudulent",
            "timestamp",
            "ownership_start_date",
            "time_since_previous_transaction",
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
    print(f"[data] processed data ready: {len(features)} rows")
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
        "X_test": features.iloc[validation_end:].reset_index(drop=True),
        "y_test": target.iloc[validation_end:].reset_index(drop=True),
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


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
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


def prepare_experiment_features(
    features: pd.DataFrame, experiment_config: dict
) -> pd.DataFrame:
    experiment_features = features.copy()
    if experiment_config.get("drop_leakage_features", False):
        print("[experiment] dropping leakage-prone features")
        experiment_features = experiment_features.drop(
            columns=LEAKAGE_COLUMNS,
            errors="ignore",
        )
    return experiment_features


def train_and_save_one_model(
    model_name: str,
    model_config: dict,
    split: dict[str, pd.DataFrame | pd.Series],
    output_root: Path,
) -> None:
    model_dir = output_root / model_name
    logger = setup_logger(model_dir / "training.log")
    logger.info("Starting %s", model_name)

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_test = split["X_test"]
    y_test = split["y_test"]
    keys_test = split["keys_test"].copy()

    scale_numeric = model_name == "logistic_regression"
    preprocessor = build_preprocessor(X_train, scale_numeric=scale_numeric)
    model = build_model(model_name, model_config["params"])
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    if model_config["type"] == "unsupervised":
        normal_mask = y_train == 0
        pipeline.fit(X_train.loc[normal_mask].reset_index(drop=True))
        raw_predictions = pipeline.predict(X_test)
        predictions = pd.Series(np.where(np.asarray(raw_predictions) == -1, 1, 0))
        if hasattr(pipeline, "decision_function"):
            scores = pd.Series(-np.asarray(pipeline.decision_function(X_test)))
        elif hasattr(pipeline, "score_samples"):
            scores = pd.Series(-np.asarray(pipeline.score_samples(X_test)))
        else:
            scores = None
        metrics = {
            "test_rows": int(len(y_test)),
            "predicted_anomalies": int(predictions.sum()),
        }
    else:
        pipeline.fit(X_train, y_train)
        predictions = pd.Series(pipeline.predict(X_test)).astype(int)
        if hasattr(pipeline, "predict_proba"):
            scores = pd.Series(pipeline.predict_proba(X_test)[:, 1])
        elif hasattr(pipeline, "decision_function"):
            scores = pd.Series(pipeline.decision_function(X_test))
        else:
            scores = None
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
        }
        if scores is not None:
            metrics["pr_auc"] = float(average_precision_score(y_test, scores))
            metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
        save_confusion_matrix(y_test, predictions, model_dir / "confusion_matrix.png")
        logger.info("Saved confusion matrix")

    joblib.dump(pipeline, model_dir / "model.joblib")
    logger.info("Saved model")

    results = keys_test.copy()
    results["y_true"] = y_test
    results["y_pred"] = predictions
    if scores is not None:
        results["fraud_score"] = scores
    results.to_csv(model_dir / "test_predictions.csv", index=False)
    logger.info("Saved test predictions")

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

        for model_name, model_config in config["models"].items():
            if model_config.get("enabled", True):
                train_and_save_one_model(model_name, model_config, split, output_root)


if __name__ == "__main__":
    main()
