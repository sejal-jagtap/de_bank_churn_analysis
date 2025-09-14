import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split

from bank_churn_analysis import(
    load_data,
    preprocess_data,
    split_features_labels,
    train_random_forest,
    train_logistic_regression,
    predict_and_evaluate,
)

@pytest.fixture(scope="module")  #runs once per test file
def churn_data():
    return load_data("Customer-Churn_Records.csv")

@pytest.fixture(scope="module") 
def processed_data(churn_data):

