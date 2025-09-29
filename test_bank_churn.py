import pytest
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split

# import functions from main py script
from bank_churn_analysis import (
    load_data,
    preprocess_data,
    split_features_labels,
    train_random_forest,
    train_logistic_regression,
    predict_and_evaluate_model,
)


@pytest.fixture(scope="module")  # runs once per test file
def churn_data():
    # load data
    df = load_data()
    df_processed = preprocess_data(df)
    return df_processed


@pytest.fixture(scope="module")
def processed_data(churn_data):
    # Split the preprocessed data into features and labels, and then further into training and test setsS
    features, labels = split_features_labels(churn_data)
    return train_test_split(features, labels, test_size=0.2, random_state=20)


def test_load_data(churn_data):
    # check if csv loads correctly
    assert isinstance(churn_data, pd.DataFrame)
    assert not churn_data.empty
    assert "Exited" in churn_data.columns


def test_preprocess_data(churn_data):
    # check if preprocess removes categorical columns and adds dummies - one hot encoding
    # df_processed = preprocess_data(churn_data)
    assert "Geography" not in churn_data.columns
    assert "Gender" not in churn_data.columns
    assert "Card Type" not in churn_data.columns
    # new/dummy columns must exists
    assert any(c in churn_data.columns for c in ["France", "Spain", "Germany"])
    assert any(c in churn_data.columns for c in ["Male"])
    assert churn_data.isnull().sum().sum() == 0


def test_split_features_labels(processed_data):
    # ensure that feature/label splitting and train,test split works correctly
    train_features, test_features, train_labels, test_labels = processed_data
    # row match labels
    assert train_features.shape[0] == train_labels.shape[0]
    assert test_features.shape[0] == test_labels.shape[0]
    # label column should not be in feature
    assert "Exited" not in train_features.columns


def test_random_forest_training(processed_data):
    # train random forest model and ensure predictions match test length set
    train_features, test_features, train_labels, test_labels = processed_data
    model = train_random_forest(train_features, train_labels)
    prediction = predict_and_evaluate_model(model, test_features, test_labels)
    # ensure predictions align with test label
    assert len(prediction) == len(test_labels)


def test_logistic_regression_training(processed_data):
    # train logistic regression model and ensure predictions match test length set
    train_features, test_features, train_labels, test_labels = processed_data
    model, scaler = train_logistic_regression(train_features, train_labels)
    prediction = predict_and_evaluate_model(model, test_features, test_labels, scaler)
    # ensure predictions align with test label
    assert len(prediction) == len(test_labels)
