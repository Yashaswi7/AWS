import os

import boto3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Upload to S3 bucket
import sagemaker

# Splitting data into train-test dataset using StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit

housing = pd.read_csv("s3://housing-1717/housing.csv")


columns = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
    "median_house_value",
]

housing = housing[columns]

# Created "income_cat" variable based on median_income
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

strat_train_set.to_csv("train_data_without_header.csv", header=False, index=False)
strat_test_set.to_csv("test_data_without_header.csv", header=False, index=False)


files = ["train_data_without_header.csv", "test_data_without_header.csv"]
session = sagemaker.Session()

for file in files:
    url = session.upload_data(
        file, bucket="housing-1717", key_prefix="housing/input-datasets"
    )
print(url)
