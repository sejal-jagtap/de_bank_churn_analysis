# flake8: noqa
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import StandardScaler

# Read data from the csv file
# Source: https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn/data

#Load data and drop unnecessary columns
def load_data(filepath="Customer-Churn-Records.csv"):
    df = pd.read_csv(filepath)
    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
    return df

# Explore the distribution of gender

def plot_gender_distribution(df):
    plt.figure(figsize=(5, 3))
    sns.countplot(
    x="Gender",
    data=df,
    palette={"Female": "yellow", "Male": "lightblue"})
    plt.title("Distribution of Gender")
    plt.xlabel("Female = 0 , Male = 1")
    plt.ylabel('Number of Customers')
    plt.show()

# Explore the distribution of age

def plot_age_distribution(df):
    plt.figure(figsize=(5, 3))
    sns.histplot(x="Age", data=df, bins=25)
    plt.ylabel("Number of Customers")
    plt.title("Distribution of Age")
    plt.show()

# Explore if the customer has credit card?

def plot_credit_card_distribution(df):
    plt.figure(figsize=(5, 3))
    sns.countplot(x="HasCrCard", data=df, hue="HasCrCard", palette={0: "red", 1: "green"}, legend= False)
    plt.title(" Customer Has a Credit Card ?")
    plt.xlabel(" No = 0 , Yes = 1 ")
    plt.ylabel("Number of Customers")
    plt.show()


# Check if there's any relation between individual columns and the output.
# eg.See if gender has any effect on customer churn.

# 1.Gender and Customer Churn Relationship
# target variable : Exited

def plot_churn_by_gender(df):
    counts = df.groupby(["Gender", "Exited"]).Exited.count().unstack()
    gender_exited = counts.plot(kind="bar", stacked=True, figsize=(5,3))
    gender_exited.set_xlabel("Gender")
    gender_exited.set_ylabel("Number of Customers")
    gender_exited.set_title("Customer Churn by Gender")
    gender_exited.legend(title="Exited",labels=["No","Yes"])
    plt.show()

# 2. Geography and Customer Churn Relationship
# target variable : Exited

def plot_churn_by_geography(df):
    counts = df.groupby(["Geography", "Exited"]).Exited.count().unstack()
    geo_exited = counts.plot(kind="bar", stacked=True, figsize=(5,3))
    geo_exited.set_xlabel("Geography")
    geo_exited.set_ylabel("Number of Customers")
    geo_exited.set_title("Customer Churn by Geography")
    geo_exited.legend(title="Exited",labels=["No","Yes"])
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
    Geography = pd.get_dummies(df.Geography).iloc[:,1:]
    Gender = pd.get_dummies(df.Gender).iloc[:,1:]
    CardType= pd.get_dummies(df["Card Type"]).iloc[:,1:]
    df_processed = pd.concat([temp,Geography,Gender,CardType], axis = 1)
    return df_processed

# ------------------------------------Model Training and Test sets------------------------------------------------

# Training set: To train the model (ML algorithm)
# Test Set: Model will be evaluated on test set.

# 1 : Divide the data into labels and feature set.

# feature set - all the columns except 'Exited'. Since value in the column 'Exited' is to be predicted.
# label set   - 'Exited' column


def split_features_labels(df):
    features = df.drop(["Exited"], axis=1)
    labels = df["Exited"]
    return features, labels

# Check if we still have any categorical values
#features.select_dtypes(include="object").columns

#Train ML Models:

# Divide training and test

# test will consist of 20% of the total dataset.
# using train_test_split from package - sklearn.model_selection   (import statement at the beginning)
# train_test_split shuffles the data and outputs 4 arrays/dataframes
# input args - features: churn_data excluding exited, label: target column, test_size: 20% for test,
# random_state= seed to ensure reproducibilty



# 1. Random forest
# n_estimators:Number of decision trees in the forest.
#             More trees → usually better performance but slower training
# fit() - trains the model, learns patters in training data.
# predict() - predicts labels for unseen data, returns an array 'predicted_lables' with values 0/1 for churn,
# compared to actual test labels to evaluate performance.

def train_random_forest(train_features,train_labels,n_estimators=200,random_state=10):
    model = rfc(n_estimators=n_estimators, random_state=random_state)
    model.fit(train_features,train_labels)
    return model

# The most commonly used metrics are precision and recall, F1 measure, accuracy and confusion matrix.
# The Scikit Learn library contains classes that can be used to calculate these metrics.

# 2.Logistic regression:

# from sklearn.linear_model import LogisticRegression

def train_logistic_regression(train_features,train_labels,max_iter=1000,random_state=20):
    scaler = StandardScaler()
    train_features_scaled = pd.DataFrame(scaler.fit_transform(train_features), columns= train_features.columns)
    model = LogisticRegression(solver='saga',max_iter=max_iter,random_state=random_state)
    model.fit(train_features_scaled, train_labels)
    return model,scaler

# Logistic Regression model accuracy: 99.85%

#Predict and Evaluate:

def predict_and_evaluate(model, test_features, test_labels, scaler=None):
    if scaler:
        test_features = pd.DataFrame(scaler.transform(test_features), columns = test_features.columns)

    prediction = model.predict(test_features)
    print("Classification Report: \n", classification_report(test_labels, prediction))
    print("Confusion Matrix: \n", confusion_matrix(test_labels, prediction))
    print("Accuracy score: \n", accuracy_score(test_labels, prediction))
    return prediction

def analysis():
    #load data
    churn_data = load_data()

    #Explore data
    plot_gender_distribution(churn_data)
    plot_age_distribution(churn_data)
    plot_credit_card_distribution(churn_data)
    plot_churn_by_gender(churn_data)
    plot_churn_by_geography(churn_data)

    #Data Preprocessing
    preprocessed_churn_data= preprocess_data(churn_data)
    features, labels = split_features_labels(preprocessed_churn_data)
    
    #Train and Test
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,labels,test_size=0.2,random_state=20
    )

    #Random Forest
    print("\n---- Random Forest ------")
    rf_model = train_random_forest(train_features,train_labels)
    rf_predictions= predict_and_evaluate(rf_model,test_features, test_labels)

    #Logistic Regression
    print("\n---- Logistic Regression ------")
    lr_model, scaler = train_logistic_regression(train_features, train_labels)
    lr_predictions = predict_and_evaluate(lr_model, test_features, test_labels, scaler)

if __name__ == "__main__":
    analysis()


'''
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
'''