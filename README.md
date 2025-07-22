# Ames-Housing-Price-Modeling-An-End-to-End-Regression-Ensemble-Pipeline

Project Overview
Predict the sale prices of 1,460 residential homes in Ames, Iowa using 79 explanatory variables.

Metric: Root-Mean-Squared-Error on log-transformed prices (RMSLE for CV; raw RMSE shown on leaderboard)

Final Public Score: 14,315.85

Rank: 239 / ~6,000

Data
Training set: train.csv (1,460 rows × 81 cols)

Test set: test.csv (1,459 rows × 80 cols)

Data description: data_description.txt

Target: SalePrice (dollars)

Key feature types:

Numerical: e.g. GrLivArea, TotalBsmtSF, LotArea

Categorical: e.g. Neighborhood, ExterQual, HouseStyle

Temporal: YearBuilt, YrSold, MoSold

Binary flags: e.g. CentralAir

Exploratory Data Analysis
Target distribution is right-skewed → applied log1p transform.

Missing values concentrated in PoolQC, MiscFeature, Alley, Fence, etc.

Strong correlations with log-price:

OverallQual (ρ≈0.80)

GrLivArea (ρ≈0.71)

TotalBsmtSF (ρ≈0.61)

Outliers: very large GrLivArea values handled by log-transform.

Preprocessing & Feature Engineering
Imputation

Numericals → median

Categoricals → "Missing"

Target transform: y = log1p(SalePrice)

Skew correction: log1p on numeric features with |skew| > 0.75

New features:

TotalSF = GrLivArea + TotalBsmtSF

HouseAge = YrSold − YearBuilt

Encoding & scaling via a ColumnTransformer + Pipeline:

python
Copy
Edit
num: [MedianImputer → StandardScaler]  
cat: [ConstantImputer("Missing") → OneHotEncoder]
Modeling & Validation
All models evaluated with 5-fold CV on y = log1p(SalePrice) and scored by neg_root_mean_squared_error.

Model	CV RMSE (log‐price)
Baseline Ridge	0.1396 ± 0.0252
Ridge + Feature Engineering	0.1265 ± 0.0156
LightGBM (default)	0.1284 ± 0.0069
LightGBM (tuned)	0.1285 ± 0.0072
Ensemble (Ridge + LGBM)	0.1258 ± 0.0063

Baseline Ridge Regression
python
Copy
Edit
from sklearn.linear_model import RidgeCV
pipeline = Pipeline([("prep", preprocessor), ("model", RidgeCV(alphas=[0.1,1,10]))])
cv = cross_val_score(pipeline, X, y, cv=5, scoring="neg_root_mean_squared_error")
Ridge + Feature Engineering
Applied skew-log transforms + TotalSF, HouseAge. CV RMSE ↓ 0.1265.

LightGBM (Default)
python
Copy
Edit
from lightgbm import LGBMRegressor
pipeline = Pipeline([("prep", preprocessor), ("model", LGBMRegressor(n_estimators=10000, learning_rate=0.01))])
LightGBM (Tuned)
RandomizedSearchCV over
n_estimators, learning_rate, num_leaves, colsample_bytree, subsample, min_child_samples, reg_alpha, reg_lambda.

Best params:

json
Copy
Edit
{
  "n_estimators": 10000,
  "learning_rate": 0.00103,
  "num_leaves": 16,
  "colsample_bytree": 0.65,
  "subsample": 0.89,
  "min_child_samples": 5,
  "reg_alpha": 1.22e-08,
  "reg_lambda": 1.58e-06
}
Ensemble
Blended predictions:

python
Copy
Edit
preds = 0.4*preds_ridge + 0.6*preds_lgbm
Final Results
Public Leaderboard RMSE: 14,315.85

Rank: 239 / ~3,000

Submission files in /submissions:

submission_baseline.csv

submission_lgbm_final.csv

submission_ensemble_final.csv

Project Structure
css
Copy
Edit
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Baseline_Ridge.ipynb
│   └── 04_LightGBM_Tuning.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   └── modeling.py
├── submissions/
│   ├── submission_baseline.csv
│   ├── submission_lgbm_final.csv
│   └── submission_ensemble_final.csv
└── README.md
How to Run
Clone repo & install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Place train.csv, test.csv in /data.

Run notebooks in order:

01_EDA.ipynb

02_Feature_Engineering.ipynb

03_Baseline_Ridge.ipynb

04_LightGBM_Tuning.ipynb

Generated submission files will appear in /submissions.

Conclusions & Next Steps
Feature engineering drove the biggest RMSE drop.

Ensembling Ridge and LightGBM gave the best CV performance.

Future improvements:

Target-encode high-cardinality categoricals

Add CatBoost and XGBoost to ensemble

Stack meta-learners

Incorporate spatial and interaction features

This report summarizes my approach and results on the Ames Housing competition. Feel free to explore the code and data in the notebooks!











