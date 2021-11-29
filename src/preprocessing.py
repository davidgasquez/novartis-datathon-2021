"""
Run all the preprocessing steps for the data.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tsfresh import extract_features


def reduce_features(dataframe, n_components=0.95, name="brand"):
    """Summarize features by PCA."""
    X = dataframe.drop([name], axis=1).dropna(axis=1)

    pca = PCA(n_components=n_components, svd_solver="full")

    pca_best_95 = pca.fit_transform(X.apply(zscore).dropna(axis=1))
    pca_bests = pd.DataFrame(
        pca_best_95,
        columns=["pca_" + name + "_" + str(i) for i in range(pca_best_95.shape[1])],
    )
    pca_bests[name] = dataframe[name]

    return pca_bests


def get_min_max(s):
    rs = s.values.reshape(-1, 1)
    se = MinMaxScaler()
    se.fit_transform(rs)
    return se.scale_[0], se.min_[0]


def get_sigma_and_mean(s):
    rs = s.values.reshape(-1, 1)
    se = StandardScaler()
    se.fit_transform(rs)
    return se.scale_[0], se.mean_[0]


sales_train = pd.read_csv("data/raw/sales_train.csv")
prioritized_hcps = pd.read_csv("data/raw/hcps.csv")
regions_hcps = pd.read_csv("data/raw/regions_hcps.csv")
regions = pd.read_csv("data/raw/regions.csv")
rtes = pd.read_csv("data/raw/rtes.csv")
activity = pd.read_csv("data/raw/activity.csv")
submission_sample = pd.read_csv("data/submission_sample.csv")

# Main dataset to add all the features
mrb = sales_train.drop(["sales"], axis=1).merge(
    submission_sample.drop(["sales", "lower", "upper"], axis=1),
    on=["month", "region", "brand"],
    how="outer",
)

# Keep only the brands we're interested in
mrb = mrb[np.isin(mrb["brand"], ["brand_1", "brand_2"])]


# Brand 3 Sales and Markets Values
brand_and_market_sales_metrics = (
    sales_train[~np.isin(sales_train["brand"], ["brand_1", "brand_2"])]
    .pivot_table(
        "sales", index=["month", "region"], columns=["brand"], aggfunc=("mean")
    )
    .reset_index()
)

mrb = mrb.merge(brand_and_market_sales_metrics, on=["region", "month"], how="left")

# Custom features for brands across regions
brand_sales = mrb.merge(sales_train, on=["month", "region", "brand"], how="left")
brand_monthly_penetration_sum = (
    (
        brand_sales.groupby(["month", "brand"])["sales"].sum()
        / brand_sales.groupby(["month", "brand"])["brand_12_market"].sum()
    )
    .reset_index()
    .rename({0: "brand_penetration_sum"}, axis=1)
)

brand_total_sales_per_month = (
    sales_train.groupby(["month", "brand"])["sales"]
    .sum()
    .rename("brand_total_sales_per_month")
)

brand_mean_sales_per_month = (
    sales_train.groupby(["month", "brand"])["sales"]
    .mean()
    .rename("brand_mean_sales_per_month")
)

brand_min_sales_per_month = (
    sales_train.groupby(["month", "brand"])["sales"]
    .min()
    .rename("brand_min_sales_per_month")
)

brand_max_sales_per_month = (
    sales_train.groupby(["month", "brand"])["sales"]
    .max()
    .rename("brand_max_sales_per_month")
)

mrb = mrb.merge(brand_monthly_penetration_sum, on=["brand", "month"], how="left")
mrb = mrb.merge(brand_total_sales_per_month, on=["brand", "month"], how="left")
mrb = mrb.merge(brand_mean_sales_per_month, on=["brand", "month"], how="left")
mrb = mrb.merge(brand_min_sales_per_month, on=["brand", "month"], how="left")
mrb = mrb.merge(brand_max_sales_per_month, on=["brand", "month"], how="left")

mrb["brand_3_market_penetration"] = mrb["brand_3"] / mrb["brand_3_market"]
mrb["brand_12vs3_market"] = mrb["brand_12_market"] / mrb["brand_3_market"]


# Brand 1 and 2 Facts

# Extract features from the brand agregated by regions
brand_extracted_features = extract_features(
    sales_train, column_id="brand", column_sort="month", column_value="sales"
)

brand_extracted_features_df = brand_extracted_features.reset_index()
brand_extracted_features_df = brand_extracted_features_df.rename(
    {"index": "brand"}, axis=1
)

region_extracted_features = extract_features(
    sales_train[
        np.isin(sales_train["brand"], ["brand_12_market", "brand_3_market", "brand_3"])
    ],
    column_id="region",
    column_sort="month",
    column_kind="brand",
    column_value="sales",
)

region_extracted_features_df = region_extracted_features.reset_index()
region_extracted_features_df = region_extracted_features_df.rename(
    {"index": "region"}, axis=1
)

brand_extracted_features_best_df = reduce_features(
    brand_extracted_features_df, name="brand"
)
mrb = mrb.merge(brand_extracted_features_best_df, on=["brand"], how="left")

region_extracted_features_best_df = reduce_features(
    region_extracted_features_df, name="region", n_components=10
)
mrb = mrb.merge(region_extracted_features_best_df, on=["region"], how="left")


# Scale parameters for brands
min_max_sales = (
    sales_train.groupby(["brand", "month"])["sales"].apply(get_min_max).reset_index()
)

min_max_sales["mm_sales_scale"] = min_max_sales["sales"].apply(lambda x: x[0])
min_max_sales["mm_sales_min"] = min_max_sales["sales"].apply(lambda x: x[1])
min_max_sales.drop("sales", axis=1, inplace=True)

mrb = mrb.merge(min_max_sales, on=["brand", "month"], how="left")

ss_sales = (
    sales_train.groupby(["brand", "month"])["sales"]
    .apply(get_sigma_and_mean)
    .reset_index()
)

ss_sales["ss_sales_scale"] = ss_sales["sales"].apply(lambda x: x[0])
ss_sales["ss_sales_min"] = ss_sales["sales"].apply(lambda x: x[1])
ss_sales.drop("sales", axis=1, inplace=True)

mrb = mrb.merge(ss_sales, on=["brand", "month"], how="left")

mm_brand_1_scale = sales_train.groupby(["brand"])["sales"].apply(get_min_max)[
    "brand_1"
][0]
mm_brand_1_min = sales_train.groupby(["brand"])["sales"].apply(get_min_max)["brand_1"][
    1
]
mm_brand_2_scale = sales_train.groupby(["brand"])["sales"].apply(get_min_max)[
    "brand_2"
][0]
mm_brand_2_min = sales_train.groupby(["brand"])["sales"].apply(get_min_max)["brand_2"][
    1
]

mrb.loc[mrb["brand"] == "brand_1", "mm_brand_scale"] = mm_brand_1_scale
mrb.loc[mrb["brand"] == "brand_1", "mm_brand_min"] = mm_brand_1_min
mrb.loc[mrb["brand"] == "brand_2", "mm_brand_scale"] = mm_brand_2_scale
mrb.loc[mrb["brand"] == "brand_2", "mm_brand_min"] = mm_brand_2_min

ss_brand_1_scale = sales_train.groupby(["brand"])["sales"].apply(get_sigma_and_mean)[
    "brand_1"
][0]
ss_brand_1_mean = sales_train.groupby(["brand"])["sales"].apply(get_sigma_and_mean)[
    "brand_1"
][1]
ss_brand_2_scale = sales_train.groupby(["brand"])["sales"].apply(get_sigma_and_mean)[
    "brand_2"
][0]
ss_brand_2_mean = sales_train.groupby(["brand"])["sales"].apply(get_sigma_and_mean)[
    "brand_2"
][1]

mrb.loc[mrb["brand"] == "brand_1", "ss_brand_scale"] = ss_brand_1_scale
mrb.loc[mrb["brand"] == "brand_1", "ss_brand_mean"] = ss_brand_1_mean
mrb.loc[mrb["brand"] == "brand_2", "ss_brand_scale"] = ss_brand_2_scale
mrb.loc[mrb["brand"] == "brand_2", "ss_brand_mean"] = ss_brand_2_mean

mrb.loc[mrb["brand"] == "brand_1", "ss_brand_scale"] = ss_brand_1_scale
mrb.loc[mrb["brand"] == "brand_1", "ss_brand_mean"] = ss_brand_1_mean
mrb.loc[mrb["brand"] == "brand_2", "ss_brand_scale"] = ss_brand_2_scale
mrb.loc[mrb["brand"] == "brand_2", "ss_brand_mean"] = ss_brand_2_mean

ss_region_brand_3_scale = (
    sales_train[sales_train["brand"] == "brand_3"]
    .groupby(["region"])["sales"]
    .apply(get_sigma_and_mean)
    .apply(lambda x: x[0])
    .rename("ss_region_brand_3_scale")
)
ss_region_brand_3_mean = (
    sales_train[sales_train["brand"] == "brand_3"]
    .groupby(["region"])["sales"]
    .apply(get_sigma_and_mean)
    .apply(lambda x: x[0])
    .rename("ss_region_brand_3_mean")
)

mrb = mrb.merge(ss_region_brand_3_scale, on=["region"], how="left")
mrb = mrb.merge(ss_region_brand_3_mean, on=["region"], how="left")

ss_brand_12_market_scale = (
    sales_train[sales_train["brand"] == "brand_12_market"]
    .groupby(["region"])["sales"]
    .apply(get_sigma_and_mean)
    .apply(lambda x: x[0])
    .rename("ss_region_brand_12_market_scale")
)
ss_brand_12_market_mean = (
    sales_train[sales_train["brand"] == "brand_12_market"]
    .groupby(["region"])["sales"]
    .apply(get_sigma_and_mean)
    .apply(lambda x: x[0])
    .rename("ss_region_brand_12_market_mean")
)
mrb = mrb.merge(ss_brand_12_market_scale, on=["region"], how="left")
mrb = mrb.merge(ss_brand_12_market_mean, on=["region"], how="left")

mrb["scale_brand_vs_market"] = (
    mrb["ss_brand_mean"] / mrb["ss_region_brand_12_market_mean"]
)


# Date Features
mrb["date_month"] = pd.to_datetime(mrb["month"])
mrb["month_number"] = mrb["date_month"].dt.month
mrb["year_number"] = mrb["date_month"].dt.year
mrb["days_in_month"] = mrb["date_month"].dt.days_in_month
mrb["is_quarter_start"] = mrb["date_month"].dt.is_quarter_start
mrb["is_quarter_end"] = mrb["date_month"].dt.is_quarter_end
mrb["is_year_start"] = mrb["date_month"].dt.is_quarter_start
mrb["is_year_end"] = mrb["date_month"].dt.is_year_end
mrb["quarter"] = mrb["date_month"].dt.quarter


# Average Cumulative Sales at Month

brand_sales = sales_train[np.isin(sales_train["brand"], ["brand_1", "brand_2"])]

brand_sales_aggs = (
    brand_sales.pivot_table(
        "sales", index=["month", "brand"], aggfunc=("mean", "min", "max", "sum")
    )
    .reset_index()
    .add_prefix("brand_sales_aggs_")
    .rename(
        {"brand_sales_aggs_month": "month", "brand_sales_aggs_brand": "brand"}, axis=1
    )
)
mrb = mrb.merge(brand_sales_aggs, on=["month", "brand"], how="left")


# Region features
regions["pci_growth"] = regions["pci18"] - regions["pci16"]
regions["density"] = regions["population"] / regions["area"]
mrb = mrb.merge(regions, on=["region"], how="left")

regions_hcps.drop(["area", "pci16", "pci18"], axis=1, inplace=True)
mrb = mrb.merge(regions_hcps, on=["region"], how="left")


# HCPs features
region_tier_count = (
    prioritized_hcps.pivot_table(
        "hcp", index=["region"], columns=["tier"], aggfunc=pd.Series.nunique
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .add_prefix("hcps_priority_")
    .rename({"hcps_priority_region": "region"}, axis=1)
)

mrb = mrb.merge(region_tier_count, on=["region"], how="left")

region_specialty_count = (
    prioritized_hcps.pivot_table(
        "hcp", index=["region"], columns=["specialty"], aggfunc=pd.Series.nunique
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .add_prefix("HCP region count ")
    .rename({"HCP region count region": "region"}, axis=1)
)
mrb = mrb.merge(region_specialty_count, on=["region"], how="left")


# RTEs
rtes["month"] = rtes["time_sent"].apply(lambda x: x[:7])

email_type_count = (
    rtes.pivot_table(
        "content_id",
        index=["region", "brand", "month"],
        columns=["email_type"],
        aggfunc=pd.Series.count,
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .add_prefix("email_type_count_")
    .rename(
        {
            "email_type_count_region": "region",
            "email_type_count_brand": "brand",
            "email_type_count_month": "month",
        },
        axis=1,
    )
)
mrb = mrb.merge(email_type_count, on=["region", "brand", "month"], how="left")

hcp_speciality_count = (
    rtes.pivot_table(
        "hcp",
        index=["region", "brand", "month"],
        columns=["specialty"],
        aggfunc=pd.Series.nunique,
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .add_prefix("hcp_speciality_count_")
    .rename(
        {
            "hcp_speciality_count_region": "region",
            "hcp_speciality_count_brand": "brand",
            "hcp_speciality_count_month": "month",
        },
        axis=1,
    )
)
mrb = mrb.merge(hcp_speciality_count, on=["region", "brand", "month"], how="left")

email_type_opening = (
    rtes.pivot_table(
        "no. openings",
        index=["region", "brand", "month"],
        columns=["email_type"],
        aggfunc=pd.Series.sum,
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .add_prefix("email_type_opening_")
    .rename(
        {
            "email_type_opening_region": "region",
            "email_type_opening_brand": "brand",
            "email_type_opening_month": "month",
        },
        axis=1,
    )
)
mrb = mrb.merge(email_type_opening, on=["region", "brand", "month"], how="left")


## Activity Features

activity_channel_sum = (
    activity.pivot_table(
        "count",
        index=["region", "brand", "month"],
        columns=["channel"],
        aggfunc=pd.Series.sum,
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .add_prefix("activity_channel_sum_")
    .rename(
        {
            "activity_channel_sum_region": "region",
            "activity_channel_sum_brand": "brand",
            "activity_channel_sum_month": "month",
        },
        axis=1,
    )
)
mrb = mrb.merge(activity_channel_sum, on=["region", "brand", "month"], how="left")

hcp_activity = (
    activity.pivot_table(
        "hcp",
        index=["region", "brand", "month"],
        columns=["specialty"],
        aggfunc=pd.Series.nunique,
    )
    .fillna(0)
    .astype(int)
    .reset_index()
    .add_prefix("hcp_activity_")
    .rename(
        {
            "hcp_activity_region": "region",
            "hcp_activity_brand": "brand",
            "hcp_activity_month": "month",
        },
        axis=1,
    )
)
mrb = mrb.merge(hcp_activity, on=["region", "brand", "month"], how="left")


# Brand exponential fit parameter

exp_fit = pd.read_csv("data/processed/exp_fit.csv")
mrb = mrb.merge(exp_fit, on="brand", how="left")


# Month Brand Stats

month_brand_stats = pd.read_csv("data/processed/month_brand_stats.csv")
mrb = mrb.merge(month_brand_stats, on=["brand", "month"], how="left")


# Lags


def random_noise(dataframe, scale=0.5):
    return np.random.normal(scale=scale, size=(len(dataframe),))


def lag_features(dataframe, column, periods):
    lags = np.arange(1, periods + 1)
    for lag in lags:
        dataframe[column + "_lag_" + str(lag)] = dataframe.groupby(["region", "brand"])[
            column
        ].transform(
            lambda x: x.shift(lag)
        )  # + random_noise(dataframe)
    return dataframe


lag_columns = [
    "brand_3",
    "brand_12_market",
    "brand_3_market",
    "brand_penetration_sum",
    "activity_channel_sum_f2f",
    "activity_channel_sum_other",
    "activity_channel_sum_other",
    "activity_channel_sum_phone",
    "activity_channel_sum_video",
    "brand_max_sales_per_month",
]

for c in lag_columns:
    mrb = lag_features(mrb, c, 5)


# Save file
mrb.to_csv("data/processed/mrb_features.gzip", index=None, compression="gzip")
