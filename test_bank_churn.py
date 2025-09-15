import pytest
import os
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
    #load data 
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "Customer-Churn-Records.csv")
    assert os.path.exists(file_path), f"CSV not found at {file_path}"
    return pd.read_csv(file_path)
    

@pytest.fixture(scope="module") 
def processed_data(churn_data):
    #returns features, labels after preprocessing
    df =load_data()
    df_processed = preprocess_data(df)
    features, labels = split_features_labels(df_processed)
    return train_test_split(features, labels, test_size=0.2, random_state=20)

def test_load_data(churn_data):
    #check if csv loads correctly
    assert isinstance(churn_data, pd.DataFrame)
    assert not churn_data.empty
    assert "Exited" in churn_data.columns

def test_preprocess_data(churn_data):
    #check if preprocess removes categorical columns and adds dummies - one hot encoding
    df_processed = preprocess_data(churn_data)
    assert "Geography" not in df_processed.columns
    assert "Gender" not in df_processed.columns
    assert "Card Type" not in df_processed.columns
    #new/dummy columns must exists
    assert any(c in  df_processed.columns for c in ["France","Spain","Germany"])
    assert any(c in  df_processed.columns for c in ["Male"])
    assert df_processed.isnull().sum().sum() == 0

def test_split_features_labels(processed_data):

    train_features, test_features, train_labels, test_labels = processed_data
    assert train_features.shape[0] ==  train_labels.shape[0]
    assert test_features.shape[0] ==  test_labels.shape[0]
    assert "Exited" not in train_features.columns

def test_random_forest_training(processed_data):
    train_features, test_features, train_labels, test_labels = processed_data
    model = train_random_forest(train_features, train_labels)
    prediction = predict_and_evaluate(model, test_features, test_labels)
    assert len(prediction) == len(test_labels)

def test_logistic_regression_raining(processed_data):
    train_features, test_features, train_labels, test_labels = processed_data
    model, scaler = train_logistic_regression(train_features, train_labels)
    prediction = predict_and_evaluate(model, test_features, test_labels, scaler)
    assert len(prediction) == len(test_labels)

'''
$ pytest -v test_bank_churn.py
================================================== test session starts ===================================================
platform win32 -- Python 3.12.11, pytest-8.4.2, pluggy-1.6.0 -- C:\Users\sejal\miniforge3\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\sejal\Dropbox\PC\Desktop\Fall_2025_Sem1\DE\de_bank_churn_analysis
plugins: anyio-4.7.0
collected 5 items                                                                                                         

test_bank_churn.py::test_load_data PASSED                                                                           [ 20%] 
test_bank_churn.py::test_preprocess_data PASSED                                                                     [ 40%] 
test_bank_churn.py::test_split_features_labels PASSED                                                               [ 60%] 
test_bank_churn.py::test_random_forest_training PASSED                                                              [ 80%] 
test_bank_churn.py::test_logistic_regression_raining PASSED                                                         [100%] 

=================================================== 5 passed in 3.22s ==================================================== 

'''