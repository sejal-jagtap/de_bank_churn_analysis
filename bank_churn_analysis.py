from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data from the csv file
# Source: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data


# Load data and drop unnecessary columns
def load_data(filepath="Customer-Churn-Records.csv"):
    df = pd.read_csv(filepath)
    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
    return df


# Explore the distribution of gender


def plot_gender_distribution(
    df,
    figsize: tuple = (5, 3),
    title: str = "Distribution of Gender",
    xlabel: str = "Female = 0 , Male = 1",
    ylabel: str = "Number of Customers",
    palette: dict = None,  # type: ignore
):
    if palette is None:
        palette = {"Female": "yellow", "Male": "lightblue"}

    plt.figure(figsize=figsize)
    sns.countplot(x="Gender", data=df, palette=palette)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Explore the distribution of age


def plot_age_distribution(
    df,
    figsize: tuple = (5, 3),
    title: str = "Distribution of Age",
    ylabel: str = "Number of Customers",
    legend: bool = False,
):
    plt.figure(figsize=figsize)
    sns.histplot(x="Age", data=df, bins=25, legend=legend)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()


# Explore if the customer has credit card?


def plot_credit_card_distribution(
    df,
    figsize: tuple = (5, 3),
    title: str = " Customer Has a Credit Card ?",
    xlabel: str = " No = 0 , Yes = 1 ",
    ylabel: str = "Number of Customers",
    hue: str = "HasCrCard",
    legend: bool = False,
    palette: dict = None,  # type: ignore
):
    if palette is None:
        palette = {0: "red", 1: "green"}

    plt.figure(figsize=figsize)
    sns.countplot(x="HasCrCard", data=df, hue=hue, palette=palette, legend=legend)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


# Check if there's any relation between individual columns and the output.
# eg.See if gender/geography has any effect on customer churn.


def plot_churn_by_category(
    df,
    category: str,
    figsize: tuple = (5, 3),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Number of Customers",
    legend_labels: list = ["No", "Yes"],
):
    counts = df.groupby(["Gender", "Exited"]).Exited.count().unstack()
    category_exited = counts.plot(kind="bar", stacked=True, figsize=figsize)
    category_exited.set_xlabel(xlabel or category)
    category_exited.set_ylabel(ylabel)
    category_exited.set_title(title or f"Customer Churn by {category}")
    category_exited.legend(title="Exited", labels=legend_labels)
    plt.show()


# --------------------------Data Preprocessing--------------------------------------------------------------------
# Since Geography and Gender columns are categorical columns ,
# We convert categorical data into numerical data using one-hot encoding scheme.

# 1. Drop columns "Geography" and "Gender"
# 2 : Create one hot coded columns for 'Geography' and 'Gender' and 'Card Type'
# 3 : Concatenate our churn data set with the one hot encoded vectors

# get_dummies() converts categorical value into binary(0 or 1),
# iloc[:,1:] - all rows, but drops first column to avoid redundancy


def preprocess_data(df):
    temp = df.drop(["Geography", "Gender", "Card Type"], axis=1)
    Geography = pd.get_dummies(df.Geography).iloc[:, 1:]
    Gender = pd.get_dummies(df.Gender).iloc[:, 1:]
    CardType = pd.get_dummies(df["Card Type"]).iloc[:, 1:]
    df_processed = pd.concat([temp, Geography, Gender, CardType], axis=1)
    return df_processed


# ------------------------------------Model Training and Test sets------------------------------------------------

# 1 : Divide the data into labels and feature set.

# feature set - all the columns except 'Exited'. Since value in the column 'Exited' is to be predicted.
# label set   - 'Exited' column


def split_features_labels(df):
    features = df.drop(["Exited"], axis=1)
    labels = df["Exited"]
    return features, labels


# ----------------------------------------ML Models----------------------------------------------------

# 1. Random forest
# n_estimators:Number of decision trees in the forest.
#             More trees → usually better performance but slower training
# fit() - trains the model, learns patters in training data.
# predict() - predicts labels for unseen data, returns an array 'predicted_lables' with values 0/1 for churn,
# compared to actual test labels to evaluate performance.


def train_random_forest(
    train_features, train_labels, n_estimators=200, random_state=10
):
    model = rfc(n_estimators=n_estimators, random_state=random_state)
    model.fit(train_features, train_labels)
    return model


# 2.Logistic regression:

# from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    train_features, train_labels, max_iter=1000, random_state=20
):
    scaler = StandardScaler()
    train_features_scaled = pd.DataFrame(
        scaler.fit_transform(train_features), columns=train_features.columns
    )
    model = LogisticRegression(
        solver="saga", max_iter=max_iter, random_state=random_state
    )
    model.fit(train_features_scaled, train_labels)
    return model, scaler


# Predict and Evaluate:


def predict_and_evaluate_model(model, test_features, test_labels, scaler=None):
    if scaler:
        test_features = pd.DataFrame(
            scaler.transform(test_features), columns=test_features.columns
        )

    prediction = model.predict(test_features)
    print("Classification Report: \n", classification_report(test_labels, prediction))
    print("Confusion Matrix: \n", confusion_matrix(test_labels, prediction))
    print("Accuracy score: \n", accuracy_score(test_labels, prediction))
    return prediction


def churn_analysis_training():
    # load data
    churn_data = load_data()

    # Explore data
    plot_gender_distribution(churn_data)
    plot_age_distribution(churn_data)
    plot_credit_card_distribution(churn_data)
    plot_churn_by_category(churn_data, category="Gender")
    plot_churn_by_category(churn_data, category="Geography")

    # Data Preprocessing
    preprocessed_churn_data = preprocess_data(churn_data)
    features, labels = split_features_labels(preprocessed_churn_data)

    # Train and Test
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=20
    )

    # Random Forest
    print("\n---- Random Forest ------")
    rf_model = train_random_forest(train_features, train_labels)
    predict_and_evaluate_model(rf_model, test_features, test_labels)

    # Logistic Regression
    print("\n---- Logistic Regression ------")
    lr_model, scaler = train_logistic_regression(train_features, train_labels)
    predict_and_evaluate_model(lr_model, test_features, test_labels, scaler)


if __name__ == "__main__":
    churn_analysis_training()


"""
RESULTS:

---- Random Forest ------
Classification Report: 
               precision    recall  f1-score   support

           0       1.00      1.00      1.00      1623
           1       0.99      1.00      1.00       377

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000

Confusion Matrix:
 [[1620    3]
 [   0  377]]
Accuracy score:
 0.9985

---- Logistic Regression ------
Classification Report: 
               precision    recall  f1-score   support

           0       1.00      1.00      1.00      1623
           1       0.99      1.00      1.00       377

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000

Confusion Matrix:
 [[1620    3]
 [   0  377]]
Accuracy score:
 0.9985
"""
