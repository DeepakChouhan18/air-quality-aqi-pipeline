# Model Comparison Report

## Regression Models

| model            | split    |   rmse |    mae |     r2 |
|:-----------------|:---------|-------:|-------:|-------:|
| Persistence val  | baseline | 10.977 |  8.508 |  0.539 |
| Persistence test | baseline | 13.423 |  9.610 |  0.657 |
| MA7 val          | baseline | 19.820 | 16.401 | -0.502 |
| MA7 test         | baseline | 17.826 | 12.184 |  0.395 |
| Ridge            | val      | 16.534 | 12.665 | -0.045 |
| Ridge            | test     | 20.001 | 16.249 |  0.239 |
| RandomForest     | val      | 20.195 | 15.169 | -0.559 |
| RandomForest     | test     | 22.149 | 15.928 |  0.067 |
| XGBoost          | val      | 16.379 | 12.534 | -0.026 |
| XGBoost          | test     | 16.230 | 12.162 |  0.499 |

## Classification Models

| model              | split   |   accuracy |   macro_f1 |
|:-------------------|:--------|-----------:|-----------:|
| LogisticRegression | val     |      0.810 |      0.468 |
| LogisticRegression | test    |      0.889 |      0.577 |
| RandomForest       | val     |      0.873 |      0.825 |
| RandomForest       | test    |      0.905 |      0.825 |
| XGBoost            | val     |      0.857 |      0.775 |
| XGBoost            | test    |      0.794 |      0.734 |

> Metrics computed on chronological val/test splits only. No data leakage. Random seed = 42.
