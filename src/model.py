import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import compose, impute, pipeline, preprocessing
from sklearn.model_selection import GridSearchCV, GroupKFold

from src.utils import save_submission

np.random.seed(421)

# Load datasets
mrb = pd.read_csv("data/processed/mrb_features.gzip", compression="gzip")
sales_train = pd.read_csv("data/raw/sales_train.csv")
submission_sample = pd.read_csv("data/submission_sample.csv")

mask = ~np.isin(sales_train["brand"], ["brand_3", "brand_3_market", "brand_12_market"])
sales_train = sales_train[mask]

# Create training dataset
train_df = sales_train.merge(mrb, on=["region", "month", "brand"], how="left")
train_df = train_df.loc[train_df["sales"] > 0]
train_df["sales"] = np.log1p(train_df["sales"])

X = train_df.drop(["month", "sales", "date_month"], axis=1)
y = np.log1p(train_df["sales"])

numerical_names = X.select_dtypes(exclude=["object"]).columns
categorical_names = X.select_dtypes(include=["object"]).columns
categorical_names = np.append(
    categorical_names.values,
    [
        "year_number",
        "month_number",
        "days_in_month",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "quarter",
    ],
)

numeric_transformer = pipeline.Pipeline(
    [
        ("imputer", impute.SimpleImputer(strategy="mean")),
        ("scaler", preprocessing.StandardScaler()),
    ]
)

categorical_transformer = pipeline.Pipeline(
    [
        ("imputer", impute.SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", preprocessing.OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

preprocessor = compose.ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_names),
        ("cat", categorical_transformer, categorical_names),
    ]
)

group_kfold = GroupKFold(n_splits=5)

model = lgb.LGBMRegressor()

e = pipeline.Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", model),
    ]
)

# More parameters @ https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
parameters = {
    "regressor__boosting_type": ["gbdt"],
    "regressor__colsample_bytree": [0.1, 0.2],
    "regressor__num_leaves": [5, 7],
    "regressor__max_depth": [4],
    "regressor__n_estimators": [50],
    "regressor__objective": ["regression", "mape"],
    "regressor__n_jobs": [-1],
    "regressor__random_state": [42],
}

cv_groups = group_kfold.split(X, y, X["region"].values)
clf = GridSearchCV(e, parameters, cv=cv_groups, n_jobs=-1)

clf.fit(X, y)

print(clf.best_params_)
print(clf.best_score_)

e = clf.best_estimator_

scoring = {
    "mae": "neg_mean_absolute_error",
    "mse": "neg_mean_squared_error",
    "r2": "r2",
}

# Send submission
evaluation_df = submission_sample[
    ["month", "region", "brand", "sales", "lower", "upper"]
].copy()

test_df = evaluation_df.merge(mrb, on=["region", "month", "brand"], how="left")

s = clf.best_estimator_.fit(X, y)

y_pred = np.expm1(
    s.predict(test_df.drop(["month", "date_month", "sales", "lower", "upper"], axis=1))
)

evaluation_df["sales"] = y_pred
evaluation_df["lower"] = y_pred * 0.9
evaluation_df["upper"] = y_pred * 1.1

save_submission(evaluation_df)
