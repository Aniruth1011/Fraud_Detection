# Early Fraud Forecasting

This package is separate from the transaction-level fraud classifiers.

The target is forward-looking: for each non-fraudulent transaction observed at time
`t`, the model predicts whether either endpoint will be involved in a fraudulent
transaction within `forecast_horizon_days`.

Run:

```bash
python Early_Fraud_Forecasting/train_plain_forecasting.py --config configs/early_fraud_forecasting.json
```

Optional graph embeddings:

```bash
python Early_Fraud_Forecasting/generate_graph_embeddings.py --config configs/early_fraud_forecasting_graph_embeddings.json
python Early_Fraud_Forecasting/train_graph_embedding_forecasting.py --config configs/early_fraud_forecasting_graph_embeddings.json
```

Plain outputs are written under `outputs/early_fraud_forecasting_plain`.
Graph-embedding outputs are written under `outputs/early_fraud_forecasting_graph_embeddings`.
