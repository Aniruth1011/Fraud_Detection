# Model Metrics Archive

Generated: 2026-08-12

This file preserves the important model-result information before moving or deleting bulky files under `outputs/`.

## Best Results

- Best meaningful overall F1: `0.484711`
  - Run: boosted no-leakage early fraud forecasting with graph embeddings, XGBoost
  - Path: `outputs/early_fraud_forecasting_graph_embeddings_boosted/li_early_fraud_forecast_30d_graph_embeddings_boosted_no_leakage/xgboost/test_metrics.json`
  - Precision: `0.652763`
  - Recall: `0.385472`
  - PR-AUC: `0.296132`
  - ROC-AUC: `0.835203`
  - Validation-selected threshold: `0.925664`

- Best meaningful fraud-detection graph F1: `0.438356`
  - Run: tuned LI PNA graph model
  - Path: `outputs/graph_models/li_pna_tuned/pna/training.log`
  - Precision: `0.800000`
  - Recall: `0.301887`
  - PR-AUC: `0.341243`
  - ROC-AUC: `0.991104`
  - Threshold: `0.834733`

- Raw best F1, not meaningful: `1.000000`
  - Run: full hetero `individual__transacts_to__account`
  - Reason to ignore: only 30 rows and all were positive, so F1=1 is not evidence of a useful model.

## Completed Runs

| Run | Source | F1 | Precision | Recall | PR-AUC | ROC-AUC | Threshold | Rows | Positives | Pred + |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `outputs/full_hetero_graph_models/li_full_hetero_graph_tuned/individual__transacts_to__account/test_metrics.json` | json | 1.000000 | 1.000000 | 1.000000 | 1.000000 | nan | 1.000000 |  |  |  |
| `outputs/early_fraud_forecasting_graph_embeddings_boosted/li_early_fraud_forecast_30d_graph_embeddings_boosted_no_leakage/xgboost/test_metrics.json` | json | 0.484711 | 0.652763 | 0.385472 | 0.296132 | 0.835203 | 0.925664 | 956479 | 6374 | 3764 |
| `outputs/early_fraud_forecasting_graph_embeddings/li_early_fraud_forecast_30d_graph_embeddings/xgboost/test_metrics.json` | json | 0.480430 | 0.657758 | 0.378412 | 0.288093 | 0.772195 |  | 956479 | 6374 | 3667 |
| `outputs/early_fraud_forecasting/li_early_fraud_forecast_30d/xgboost/test_metrics.json` | json | 0.477270 | 0.652079 | 0.376373 | 0.214173 | 0.732852 |  | 956479 | 6374 | 3679 |
| `outputs/early_fraud_forecasting_graph_embeddings_boosted/li_early_fraud_forecast_30d_graph_embeddings_boosted_no_leakage/random_forest/test_metrics.json` | json | 0.459815 | 0.678189 | 0.347819 | 0.289135 | 0.867507 | 0.583956 | 956479 | 6374 | 3269 |
| `outputs/graph_models/li_pna_tuned/pna/training.log` | log | 0.438356 | 0.800000 | 0.301887 | 0.341243 | 0.991104 | 0.834733 |  |  |  |
| `outputs/graph_models/li_graph_models/gin/training.log` | log | 0.404494 | 0.818182 | 0.268657 | 0.307080 | 0.920022 |  |  |  |  |
| `outputs/early_fraud_forecasting_graph_embeddings/li_early_fraud_forecast_30d_graph_embeddings/random_forest/test_metrics.json` | json | 0.392075 | 0.463034 | 0.339975 | 0.272607 | 0.860324 |  | 956479 | 6374 | 4680 |
| `outputs/graph_models/li_gin_tuned/gin/training.log` | log | 0.302326 | 1.000000 | 0.178082 | 0.299022 | 0.993439 | 0.929801 |  |  |  |
| `outputs/graph_models/li_graph_models/pna/training.log` | log | 0.187500 | 0.428571 | 0.120000 | 0.080089 | 0.983236 |  |  |  |  |
| `outputs/classical_models/li_classical_tabular/logistic_regression/test_metrics.json` | json | 0.109850 | 0.059801 | 0.673679 | 0.052024 | 0.945920 |  |  |  |  |
| `outputs/early_fraud_forecasting/li_early_fraud_forecast_30d/random_forest/test_metrics.json` | json | 0.079747 | 0.043193 | 0.518826 | 0.224392 | 0.809053 |  | 956479 | 6374 | 76563 |
| `outputs/classical_models/li_classical_tabular/xgboost/test_metrics.json` | json | 0.051917 | 0.460843 | 0.027508 | 0.749608 | 0.998331 |  |  |  |  |
| `outputs/early_fraud_forecasting_graph_embeddings/li_early_fraud_forecast_30d_graph_embeddings/logistic_regression/test_metrics.json` | json | 0.044203 | 0.022845 | 0.679636 | 0.267084 | 0.832386 |  | 956479 | 6374 | 189629 |
| `outputs/early_fraud_forecasting/li_early_fraud_forecast_30d/logistic_regression/test_metrics.json` | json | 0.038244 | 0.019622 | 0.750392 | 0.253449 | 0.813950 |  | 956479 | 6374 | 243759 |
| `outputs/classical_models/li_classical_tabular/random_forest/test_metrics.json` | json | 0.025413 | 0.108992 | 0.014383 | 0.100207 | 0.960992 |  |  |  |  |
| `outputs/full_hetero_graph_models/li_full_hetero_graph_tuned/account__transacts_to__account/test_metrics.json` | json | 0.000398 | 0.000199 | 1.000000 | 0.000199 | 0.500000 | 0.507678 |  |  |  |
| `outputs/early_fraud_forecasting/li_early_fraud_forecast_30d/isolation_forest/test_metrics.json` | json | 0.000000 | 0.000000 | 0.000000 | 0.006131 | 0.483049 |  | 956479 | 6374 | 0 |
| `outputs/early_fraud_forecasting_graph_embeddings/li_early_fraud_forecast_30d_graph_embeddings/isolation_forest/test_metrics.json` | json | 0.000000 | 0.000000 | 0.000000 | 0.008025 | 0.537006 |  | 956479 | 6374 | 0 |
| `outputs/graph_models/li_graph_models/gat/training.log` | log | 0.000000 | 0.000000 | 0.000000 | 0.002597 | 0.922795 |  |  |  |  |
| `outputs/classical_models/li_classical_tabular/isolation_forest/test_metrics.json` | json |  |  |  |  |  |  | 957885 |  |  |

## Boosted Forecasting Validation Metrics

- XGBoost boosted no-leakage:
  - Validation F1: `0.429802`
  - Validation precision: `0.968132`
  - Validation recall: `0.276214`
  - Validation PR-AUC: `0.333133`
  - Validation ROC-AUC: `0.795193`
  - Validation best F1 threshold: `0.925664`

- Random forest boosted no-leakage:
  - Validation F1: `0.413227`
  - Validation precision: `0.886583`
  - Validation recall: `0.269394`
  - Validation PR-AUC: `0.338501`
  - Validation ROC-AUC: `0.809559`
  - Validation best F1 threshold: `0.583956`

## Notes And Caveats

- Hetero `individual__transacts_to__account` F1=`1.0` should be ignored because the relation has only 30 rows and all are positive.
- Hetero `account__transacts_to__account` was effectively useless: F1=`0.000398`, PR-AUC near prevalence, ROC-AUC=`0.5`.
- Balanced negative-sampling GIN was poor before interruption/stop: validation F1 stayed around `0.005` to `0.007`.
- Old tuned GIN/PNA graph runs printed valid test metrics but crashed while saving artifacts in earlier code; their metrics are preserved from `training.log`.
- HI PNA run existed at `outputs/graph_models/hi_graph_models_tuned_best/pna/training.log`, but at archive time it had only started epoch 1 and had no completed test metrics yet.
- Threshold reports were not present under `outputs/` at archive time.

## Output Folder Sizes At Archive Time

| Folder | Size |
|---|---:|
| `outputs/full_hetero_graph_models` | 2.0G |
| `outputs/classical_models` | 2.9G |
| `outputs/early_fraud_forecasting` | 4.6G |
| `outputs/graph_models` | 5.7G |
| `outputs/early_fraud_forecasting_graph_embeddings` | 6.8G |
| `outputs/early_fraud_forecasting_graph_embeddings_boosted` | 8.5G |

## Best Commands To Recreate Current Preferred Runs

LI boosted no-leakage early forecasting:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib-forecast python Early_Fraud_Forecasting/train_graph_embedding_forecasting.py --config configs/early_fraud_forecasting_graph_embeddings_boosted.json
```

HI graph embeddings for forecasting:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib-forecast python Early_Fraud_Forecasting/generate_graph_embeddings.py --config configs/early_fraud_forecasting_graph_embeddings_boosted_hi_xgboost.json
```

HI boosted no-leakage early forecasting, XGBoost only:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib-forecast python Early_Fraud_Forecasting/train_graph_embedding_forecasting.py --config configs/early_fraud_forecasting_graph_embeddings_boosted_hi_xgboost.json
```

HI tuned PNA graph model:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib-graph python train_graph_models.py --config configs/graph_models_hi_tuned_best.json --models pna --num-workers 0
```
