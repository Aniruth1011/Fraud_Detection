# Graph-Based Fraud Detection and Early Fraud Forecasting

This project builds an AML platform for two related tasks on the LI transaction
dataset:

1. **Transaction fraud detection:** classify the current transaction as
   fraudulent.
2. **Early fraud forecasting:** predict whether an entity involved in the
   current transaction will be involved in fraud within the next 30 days.

The implementation includes temporal graph construction, graph neural networks,
past-only graph embeddings, leakage-aware forecasting models, batch scoring,
audit persistence, and drift monitoring.

## Data And Evaluation

The input is a time-ordered transaction graph containing individuals, accounts,
and transaction edges. Transaction amounts, timestamps, ports, time deltas,
node attributes, and edge attributes are used as model inputs.

All primary evaluation uses chronological splits rather than random splits:

- **Training window:** earliest events
- **Validation window:** later events used for checkpoint and threshold selection
- **Test window:** final unseen events used only for final metrics

This prevents future transactions from leaking into earlier predictions. Fraud
labels are used for training and evaluation only; they are excluded from serving
features.

## Transaction Fraud Detection

### Graph construction

The transaction data is converted into a temporal graph. The pipeline prepares
graph snapshots, normalizes features using training data, and masks the seed
edges during neighborhood evaluation. Reverse message-passing edges are added
so information can flow in both directions while preserving the temporal split.

### Models evaluated

The graph-model runner supports PNA, GIN, GAT, and relation-aware graph models.
The main LI experiments used:

- PNA and GIN edge classifiers
- Two message-passing layers with hidden dimension 64
- Edge updates and dropout
- Node features, port features, and transaction time deltas
- Reverse message passing
- `LinkNeighborLoader` mini-batch neighborhood sampling
- Two-hop neighbor sampling with `[50, 25]` neighbors
- Positive-class weighting of 20 to address class imbalance
- Validation-selected operating thresholds
- Early stopping based on validation F1

PNA was selected because it provided the strongest meaningful graph-detection
trade-off on the LI test split.

### Best LI transaction-detection result

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tuned PNA, best checkpoint | 0.999856 | 0.800000 | 0.301887 | 0.438356 | 0.341243 | 0.991104 | 0.834733 |

The deployed checkpoint is:

```text
outputs/li_pna_tuned/pna/best_checkpoint.pt
```

The checkpoint was selected at epoch 11 using validation performance and
evaluated on the held-out test split.

## Early Fraud Forecasting

Early forecasting is a different prediction problem from current-transaction
detection. At event time `t`, the model uses information available up to `t` to
predict future fraud within a 30-day horizon.

### Graph-embedding approach

Past-only entity graphs are generated from historical transactions. The graph
embeddings are computed from historical adjacency snapshots using an SVD-based
embedding pipeline. These embeddings are combined with transaction and entity
features and passed to forecasting models.

Label-derived features such as prior fraud counts are excluded from the
serving-compatible model. This prevents training-serving skew and target
leakage.

The forecasting experiments included:

- Graph-embedding XGBoost
- Graph-embedding random forest
- Logistic regression baselines
- Isolation Forest and other anomaly baselines
- An early-forecasting PNA graph model

### Best observed early-forecasting results

The strongest observed early PNA experiment achieved:

| Model | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Early PNA, best observed run | 0.987456 | 0.827032 | 0.378952 | 0.519751 | 0.402225 | 0.832595 | 0.859258 |

The strongest serving-compatible no-leakage graph-embedding XGBoost result
recorded in the archived runs was:

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| Graph-embedding XGBoost, no leakage | 0.652763 | 0.385472 | 0.484711 | 0.296132 | 0.835203 | 0.925664 |

The no-leakage XGBoost artifact is the safer serving reference because its
features are available at prediction time. The early PNA result is retained as
the strongest experimental forecasting result and requires the corresponding
PNA serving integration before production use.

## Platform Deployment

`run_platform.py` provides local batch scoring and Kafka-ready execution. The
platform performs:

- Event-time ordered transaction replay
- PNA transaction scoring
- Early-risk scoring
- Risk-decision construction
- Label resolution after the forecast horizon is available
- SQLite ledger persistence
- Runtime throughput and latency reporting
- Drift evaluation and drift-event persistence
- Optional Kafka publication

### Recorded throughput

The restored LI PNA checkpoint processed a 50,000-event detection batch with:

| Measure | Result |
|---|---:|
| Input events | 50,000 |
| Predictions | 50,000 |
| Prediction coverage | 100% |
| Detection scoring time | 18.905 seconds |
| Detection throughput | 2,644 events/second |
| Missing predictions | 0 |
| Ordering | Correct |

The full 1,000-event local phase covering detection, early scoring, decisions,
labels, drift, and SQLite writes completed successfully. Its measured overall
throughput was approximately 5.07 events/second because ledger persistence was
the dominant cost. Detection scoring alone ran at approximately 58.46
events/second in that full phase.

The reported event-time-to-processing-time lag is large when replaying historical
2025 data in 2026. That is replay age, not model inference latency.

## Drift Monitoring

The monitoring layer compares a baseline reference window with a current window.
It supports:

- PSI for numerical feature and prediction-score drift
- KS statistic and p-value for numerical distributions
- Wasserstein distance for magnitude of distribution shift
- JS divergence for categorical distributions
- PR-AUC and recall at a fixed false-positive rate
- Brier score for probability calibration
- Detection rate and false-alarm rate
- Time-to-detect for ordered event streams

The runtime drift monitor also evaluates amount distributions, graph size
ratios, prediction-score shifts, positive-rate shifts, and labeled F1 drops.
Warning and confirmed states are persisted across monitoring windows.

In the 50,000-event epoch-11 comparison, natural prediction-score drift was
small:

```text
Prediction-score PSI:       0.066459
KS statistic:               0.062140
Wasserstein distance:       0.001462
```

A controlled score-shift simulation was 
implemented to verify that the drift pipeline raises PSI above the `0.10`
warning level when synthetic output drift is introduced.

## Important Artifacts

```text
outputs/li_pna_tuned/pna/best_checkpoint.pt
outputs/early_fraud_forecasting_graph_embeddings_boosted/li_early_fraud_forecast_30d_graph_embeddings_boosted_no_leakage/xgboost/model.joblib
outputs/early_fraud_forecasting/graph_embeddings/li_entity_embeddings.csv
outputs/platform_test/detection_predictions_50k.jsonl
outputs/platform_test/detection_summary_50k.json
scripts/evaluate_drift_metrics.py
scripts/simulate_prediction_drift.py
configs/platform.yaml
```

## Reproducibility

Use the project environment and keep user-site packages disabled:

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/matplotlib
PY=/home/aniruth/.conda/envs/fraud-gnn/bin/python
```

The main platform entry point is:

```bash
$PY run_platform.py --help
```

The saved metric archive contains historical comparison results and caveats for
older experiments:

```text
MODEL_METRICS_ARCHIVE.md
```
